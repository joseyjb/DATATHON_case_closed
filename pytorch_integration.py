"""
PyTorch Integration for Case Closed Agent
Encodes game state and uses neural network for move prediction.
"""

import numpy as np
import torch
from typing import Tuple, List, Dict
from collections import deque

class StateEncoder:
    """Encodes game state into neural network input format."""
    
    def __init__(self, board_height=18, board_width=20):
        self.height = board_height
        self.width = board_width
    
    def encode_board(self, my_trail: List[Tuple[int, int]], 
                    other_trail: List[Tuple[int, int]]) -> np.ndarray:
        """
        Encode board state as 3-channel image.
        
        Channels:
        0: Empty cells (1 where empty, 0 where occupied)
        1: My trail (1 where my trail exists)
        2: Opponent trail (1 where opponent trail exists)
        
        Returns:
            board_state: (height, width, 3) numpy array
        """
        board = np.zeros((self.height, self.width, 3), dtype=np.float32)
        
        # Channel 0: Empty cells (start with all 1s)
        board[:, :, 0] = 1.0
        
        # Channel 1: My trail
        for x, y in my_trail:
            x_norm = x % self.width
            y_norm = y % self.height
            board[y_norm, x_norm, 1] = 1.0
            board[y_norm, x_norm, 0] = 0.0  # Not empty
        
        # Channel 2: Opponent trail
        for x, y in other_trail:
            x_norm = x % self.width
            y_norm = y % self.height
            board[y_norm, x_norm, 2] = 1.0
            board[y_norm, x_norm, 0] = 0.0  # Not empty
        
        return board
    
    def encode_features(self, state: Dict) -> np.ndarray:
        """
        Encode additional game features.
        
        Features:
        - My position (x, y normalized)
        - Opponent position (x, y normalized)
        - Turn count (normalized to [0, 1])
        - My boosts remaining (normalized to [0, 1])
        - Opponent boosts remaining (normalized to [0, 1])
        - My trail length (normalized)
        - Opponent trail length (normalized)
        - Space occupancy ratio
        
        Returns:
            features: (10,) numpy array
        """
        features = np.zeros(10, dtype=np.float32)
        
        # Position features
        player_num = state.get("player_number", 1)
        my_trail = state.get(f"agent{player_num}_trail", [])
        other_trail = state.get(f"agent{3-player_num}_trail", [])
        
        if my_trail:
            my_pos = my_trail[-1]
            features[0] = my_pos[0] / self.width
            features[1] = my_pos[1] / self.height
        
        if other_trail:
            opp_pos = other_trail[-1]
            features[2] = opp_pos[0] / self.width
            features[3] = opp_pos[1] / self.height
        
        # Game state features
        features[4] = state.get("turn_count", 0) / 200.0  # Normalized turn
        features[5] = state.get(f"agent{player_num}_boosts", 3) / 3.0  # My boosts
        features[6] = state.get(f"agent{3-player_num}_boosts", 3) / 3.0  # Opp boosts
        
        # Trail length features (log scale for better distribution)
        my_length = state.get(f"agent{player_num}_length", 2)
        opp_length = state.get(f"agent{3-player_num}_length", 2)
        features[7] = np.log1p(my_length) / np.log1p(200)
        features[8] = np.log1p(opp_length) / np.log1p(200)
        
        # Space occupancy ratio
        total_cells = self.height * self.width
        occupied = my_length + opp_length
        features[9] = occupied / total_cells
        
        return features
    
    def encode_state(self, state: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Complete state encoding for neural network.
        
        Returns:
            board_state: (height, width, 3) array
            features: (10,) array
        """
        player_num = state.get("player_number", 1)
        my_trail = state.get(f"agent{player_num}_trail", [])
        other_trail = state.get(f"agent{3-player_num}_trail", [])
        
        board_state = self.encode_board(my_trail, other_trail)
        features = self.encode_features(state)
        
        return board_state, features


class NeuralMoveSelector:
    """Uses neural network to select moves."""
    
    def __init__(self, model, state_encoder, use_stochastic=False):
        """
        Args:
            model: PyTorch model (CaseClosedNet)
            state_encoder: StateEncoder instance
            use_stochastic: If True, sample from policy; if False, use greedy
        """
        self.model = model
        self.encoder = state_encoder
        self.use_stochastic = use_stochastic
        
        self.direction_map = ["UP", "DOWN", "LEFT", "RIGHT", "BOOST"]
    
    def get_valid_actions_mask(self, current_dir: str, my_trail: List, 
                               other_trail: List, boosts_remaining: int) -> np.ndarray:
        """
        Create mask for valid actions.
        
        Returns:
            mask: (5,) boolean array [UP, DOWN, LEFT, RIGHT, BOOST]
        """
        mask = np.ones(5, dtype=np.float32)
        
        # Can't go opposite direction
        opposites = {"UP": 1, "DOWN": 0, "LEFT": 3, "RIGHT": 2}  # Indices
        if current_dir in opposites:
            mask[opposites[current_dir]] = 0.0
        
        # Check if each direction is safe
        if my_trail:
            current_pos = my_trail[-1]
            my_trail_set = set(tuple(p) for p in my_trail)
            other_trail_set = set(tuple(p) for p in other_trail)
            occupied = my_trail_set | other_trail_set
            
            directions = [
                (current_pos[0], current_pos[1] - 1),  # UP
                (current_pos[0], current_pos[1] + 1),  # DOWN
                (current_pos[0] - 1, current_pos[1]),  # LEFT
                (current_pos[0] + 1, current_pos[1]),  # RIGHT
            ]
            
            for i, next_pos in enumerate(directions):
                # Normalize for torus
                next_pos = (next_pos[0] % self.encoder.width, 
                           next_pos[1] % self.encoder.height)
                if next_pos in occupied:
                    mask[i] = 0.0
        
        # Can't boost if no boosts remaining
        if boosts_remaining <= 0:
            mask[4] = 0.0
        
        # Ensure at least one action is valid
        if mask[:4].sum() == 0:
            # Emergency: allow all directions except opposite
            mask[:4] = 1.0
            if current_dir in opposites:
                mask[opposites[current_dir]] = 0.0
        
        return mask
    
    def select_move(self, state: Dict, current_dir: str) -> Tuple[str, bool]:
        """
        Select best move using neural network.
        
        Returns:
            direction: "UP", "DOWN", "LEFT", or "RIGHT"
            use_boost: Whether to use a speed boost
        """
        # Encode state
        board_state, features = self.encoder.encode_state(state)
        
        # Get valid actions
        player_num = state.get("player_number", 1)
        my_trail = state.get(f"agent{player_num}_trail", [])
        other_trail = state.get(f"agent{3-player_num}_trail", [])
        boosts = state.get(f"agent{player_num}_boosts", 3)
        
        valid_mask = self.get_valid_actions_mask(current_dir, my_trail, 
                                                 other_trail, boosts)
        
        # Get model prediction
        action_idx, probs, value = self.model.predict_move(
            board_state, features, valid_mask
        )
        
        # Decode action
        use_boost = False
        if action_idx == 4:  # BOOST selected
            use_boost = True
            # Pick best direction from remaining actions
            direction_probs = probs[:4] * valid_mask[:4]
            if direction_probs.sum() > 0:
                action_idx = np.argmax(direction_probs)
            else:
                action_idx = 0  # Default to UP
        
        direction = self.direction_map[action_idx]
        
        return direction, use_boost


class HybridMoveSelector:
    """
    Combines neural network with heuristic fallback.
    Uses NN when confident, falls back to heuristic when uncertain.
    """
    
    def __init__(self, neural_selector, heuristic_selector, confidence_threshold=0.6):
        self.neural = neural_selector
        self.heuristic = heuristic_selector
        self.threshold = confidence_threshold
    
    def select_move(self, state: Dict, current_dir: str) -> Tuple[str, bool]:
        """
        Select move using hybrid approach.
        
        If neural network is confident (max prob > threshold), use it.
        Otherwise, fall back to heuristic.
        """
        # Try neural network first
        board_state, features = self.neural.encoder.encode_state(state)
        player_num = state.get("player_number", 1)
        my_trail = state.get(f"agent{player_num}_trail", [])
        other_trail = state.get(f"agent{3-player_num}_trail", [])
        boosts = state.get(f"agent{player_num}_boosts", 3)
        
        valid_mask = self.neural.get_valid_actions_mask(
            current_dir, my_trail, other_trail, boosts
        )
        
        action_idx, probs, value = self.neural.model.predict_move(
            board_state, features, valid_mask
        )
        
        max_prob = probs.max()
        
        if max_prob >= self.threshold:
            # Neural network is confident, use its prediction
            use_boost = action_idx == 4
            if use_boost:
                direction_probs = probs[:4] * valid_mask[:4]
                action_idx = np.argmax(direction_probs) if direction_probs.sum() > 0 else 0
            direction = self.neural.direction_map[action_idx]
            return direction, use_boost
        else:
            # Fall back to heuristic
            return self.heuristic.select_move(state, current_dir)


# Example usage
if __name__ == "__main__":
    from pytorch_model import CaseClosedNet
    
    # Create components
    model = CaseClosedNet()
    encoder = StateEncoder()
    selector = NeuralMoveSelector(model, encoder)
    
    # Test state
    test_state = {
        "player_number": 1,
        "agent1_trail": [(1, 2), (2, 2), (3, 2)],
        "agent2_trail": [(17, 15), (16, 15)],
        "agent1_boosts": 3,
        "agent2_boosts": 3,
        "agent1_length": 3,
        "agent2_length": 2,
        "turn_count": 5,
    }
    
    # Encode state
    board, features = encoder.encode_state(test_state)
    print(f"Board shape: {board.shape}")
    print(f"Features shape: {features.shape}")
    print(f"Features: {features}")
    
    # Select move
    direction, use_boost = selector.select_move(test_state, "RIGHT")
    print(f"Selected move: {direction}, Boost: {use_boost}")
