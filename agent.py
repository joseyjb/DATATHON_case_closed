import os
from flask import Flask, request, jsonify
from threading import Lock
from collections import deque

# --- Required Game and Utilities Imports ---
from case_closed_game import Game, Direction, GameResult
from pathfinding_utils import PathfindingUtils

# --- PyTorch Model Imports ---
import torch
from pytorch_model import CaseClosedNet
from pytorch_integration import StateEncoder, NeuralMoveSelector

# --- Flask API Server Setup ---
app = Flask(__name__)
GLOBAL_GAME = Game()
LAST_POSTED_STATE = {}
game_lock = Lock()

PARTICIPANT = "ParticipantX"
AGENT_NAME = "AgentX-PyTorch"

# --- Pathfinding Utilities Setup ---
pathfinder = PathfindingUtils(board_height=18, board_width=20)

# --- PyTorch Model Initialization ---
model = CaseClosedNet()
encoder = StateEncoder()
neural_selector = NeuralMoveSelector(model, encoder, use_stochastic=False)
model_path = "case_closed_model.pth"
if os.path.exists(model_path):
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        print(f"✓ Loaded model weights from {model_path}")
    except Exception as e:
        print(f"✗ Could not load model: {e}\nUsing randomly initialized model!")

# --- Health/Info Endpoint ---
@app.route("/", methods=["GET"])
def info():
    """Basic health/info endpoint used by the judge to check connectivity.
    Returns participant and agent_name (so Judge.check_latency can create Agent objects).
    """
    return jsonify({"participant": PARTICIPANT, "agent_name": AGENT_NAME}), 200

# --- Update Local Game State From Judge POST ---
def _update_local_game_from_post(data: dict):
    """Update the local GLOBAL_GAME using the JSON posted by the judge.
    The judge posts a dictionary with keys matching the Judge.send_state payload.
    """
    with game_lock:
        LAST_POSTED_STATE.clear()
        LAST_POSTED_STATE.update(data)
        if "board" in data:
            GLOBAL_GAME.board.grid = data["board"]
        if "agent1_trail" in data:
            GLOBAL_GAME.agent1.trail = deque(tuple(p) for p in data["agent1_trail"])
        if "agent2_trail" in data:
            GLOBAL_GAME.agent2.trail = deque(tuple(p) for p in data["agent2_trail"])
        if "agent1_length" in data:
            GLOBAL_GAME.agent1.length = int(data["agent1_length"])
        if "agent2_length" in data:
            GLOBAL_GAME.agent2.length = int(data["agent2_length"])
        if "agent1_alive" in data:
            GLOBAL_GAME.agent1.alive = bool(data["agent1_alive"])
        if "agent2_alive" in data:
            GLOBAL_GAME.agent2.alive = bool(data["agent2_alive"])
        if "agent1_boosts" in data:
            GLOBAL_GAME.agent1.boosts_remaining = int(data["agent1_boosts"])
        if "agent2_boosts" in data:
            GLOBAL_GAME.agent2.boosts_remaining = int(data["agent2_boosts"])
        if "turn_count" in data:
            GLOBAL_GAME.turns = int(data["turn_count"])

# --- State Receive Endpoint ---
@app.route("/send-state", methods=["POST"])
def receive_state():
    """Judge calls this to push the current game state to the agent server.
    The agent should update its local representation and return 200.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "no json body"}), 400
    _update_local_game_from_post(data)
    return jsonify({"status": "state received"}), 200

# --- Move Decision Logic ---
@app.route("/send-move", methods=["GET"])
def send_move():
    """Judge calls this (GET) to request the agent's move for the current tick.
    Query params: player_number, attempt_number, random_moves_left, turn_count.
    Return format: {"move": "DIRECTION"} or {"move": "DIRECTION:BOOST"}
    """
    player_number = request.args.get("player_number", default=1, type=int)
    with game_lock:
        state = dict(LAST_POSTED_STATE)
        my_agent = GLOBAL_GAME.agent1 if player_number == 1 else GLOBAL_GAME.agent2
        boosts_remaining = my_agent.boosts_remaining
    # --- YOUR CODE GOES HERE ---
    my_trail = state.get("agent1_trail" if player_number == 1 else "agent2_trail", [])
    other_trail = state.get("agent2_trail" if player_number == 1 else "agent1_trail", [])
    turn_count = state.get("turn_count", 0)
    if not my_trail or len(my_trail) == 0:
        return jsonify({"move": "RIGHT"}), 200
    current_dir = pathfinder.get_current_direction(my_trail)
    try:
        state["player_number"] = player_number
        board_state, features = encoder.encode_state(state)
        valid_mask = neural_selector.get_valid_actions_mask(
            current_dir, my_trail, other_trail, boosts_remaining
        )
        action_idx, probs, value = model.predict_move(
            board_state, features, valid_mask
        )
        use_boost = action_idx == 4 and boosts_remaining > 0
        direction_map = ["UP", "DOWN", "LEFT", "RIGHT"]
        direction = direction_map[action_idx if action_idx < 4 else 0]
        move = f"{direction}:BOOST" if use_boost else direction
        next_pos = pathfinder.get_next_position(tuple(my_trail[-1]), direction)
        is_safe = pathfinder.is_safe_position(next_pos, my_trail, other_trail)
        if not is_safe:
            move = direction  # fallback is just direction (could improve)
    except Exception as e:
        print(f"PyTorch error: {e}")
        move = "RIGHT"  # fallback if model errors
    # --- END OF YOUR CODE ---
    return jsonify({"move": move}), 200

# --- Endgame Finalization Endpoint ---
@app.route("/end", methods=["POST"])
def end_game():
    """Judge notifies agent that the match finished and provides final state.
    We update local state for record-keeping and return OK.
    """
    data = request.get_json()
    if data:
        _update_local_game_from_post(data)
    return jsonify({"status": "acknowledged"}), 200

# --- Main Driver ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5008"))
    app.run(host="0.0.0.0", port=port, debug=True)
