"""
Training script for Case Closed PyTorch model using self-play.
Agents play against each other and learn from wins/losses.
"""

import torch
import numpy as np
from typing import List, Tuple
import random
from collections import deque
import pickle

from pytorch_model import CaseClosedNet, ExperienceBuffer, ModelTrainer, save_model, load_model
from pytorch_integration import StateEncoder
from case_closed_game import Game, Direction, GameResult


class SelfPlayTrainer:
    """Trains the model through self-play."""
    
    def __init__(self, model_path="case_closed_model.pth"):
        self.model = CaseClosedNet()
        self.encoder = StateEncoder()
        self.trainer = ModelTrainer(self.model, learning_rate=0.0005)
        self.experience_buffer = ExperienceBuffer(capacity=50000)
        self.model_path = model_path
        
        # Training stats
        self.games_played = 0
        self.wins_p1 = 0
        self.wins_p2 = 0
        self.draws = 0
    
    def direction_to_action_idx(self, direction: Direction) -> int:
        """Convert Direction enum to action index."""
        mapping = {
            Direction.UP: 0,
            Direction.DOWN: 1,
            Direction.LEFT: 2,
            Direction.RIGHT: 3,
        }
        return mapping.get(direction, 0)
    
    def action_idx_to_direction(self, idx: int) -> Direction:
        """Convert action index to Direction enum."""
        mapping = {
            0: Direction.UP,
            1: Direction.DOWN,
            2: Direction.LEFT,
            3: Direction.RIGHT,
        }
        return mapping.get(idx, Direction.RIGHT)
    
    def get_state_dict(self, game: Game, player_num: int) -> dict:
        """Create state dictionary for encoder."""
        return {
            "player_number": player_num,
            "agent1_trail": game.agent1.get_trail_positions(),
            "agent2_trail": game.agent2.get_trail_positions(),
            "agent1_boosts": game.agent1.boosts_remaining,
            "agent2_boosts": game.agent2.boosts_remaining,
            "agent1_length": game.agent1.length,
            "agent2_length": game.agent2.length,
            "turn_count": game.turns,
            "agent1_alive": game.agent1.alive,
            "agent2_alive": game.agent2.alive,
        }
    
    def get_valid_actions(self, game: Game, player_num: int) -> np.ndarray:
        """Get mask of valid actions for a player."""
        agent = game.agent1 if player_num == 1 else game.agent2
        other = game.agent2 if player_num == 1 else game.agent1
        
        mask = np.ones(5, dtype=np.float32)
        
        # Can't go opposite direction
        current_dir = agent.direction
        cur_dx, cur_dy = current_dir.value
        
        for idx, direction in enumerate([Direction.UP, Direction.DOWN, 
                                         Direction.LEFT, Direction.RIGHT]):
            req_dx, req_dy = direction.value
            if (req_dx, req_dy) == (-cur_dx, -cur_dy):
                mask[idx] = 0.0
        
        # Can't boost if no boosts
        if agent.boosts_remaining <= 0:
            mask[4] = 0.0
        
        return mask
    
    def select_action(self, state_dict: dict, valid_mask: np.ndarray, 
                     epsilon: float = 0.1) -> Tuple[int, bool]:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state_dict: Game state
            valid_mask: Valid actions mask
            epsilon: Exploration probability
        
        Returns:
            action_idx: Direction index (0-3)
            use_boost: Whether to use boost
        """
        # Epsilon-greedy: explore vs exploit
        if random.random() < epsilon:
            # Random valid action
            valid_actions = np.where(valid_mask[:4] > 0)[0]
            if len(valid_actions) > 0:
                action_idx = random.choice(valid_actions)
            else:
                action_idx = 0
            use_boost = random.random() < 0.3 and valid_mask[4] > 0
        else:
            # Use model
            board_state, features = self.encoder.encode_state(state_dict)
            action_idx, probs, value = self.model.predict_move(
                board_state, features, valid_mask
            )
            
            # Check if boost action selected
            use_boost = action_idx == 4
            if use_boost:
                # Pick best direction
                direction_probs = probs[:4] * valid_mask[:4]
                if direction_probs.sum() > 0:
                    action_idx = np.argmax(direction_probs)
                else:
                    action_idx = 0
        
        return action_idx, use_boost
    
    def play_game(self, epsilon: float = 0.1, save_experience: bool = True) -> GameResult:
        """
        Play one self-play game.
        
        Args:
            epsilon: Exploration rate
            save_experience: Whether to save experiences to buffer
        
        Returns:
            Game result
        """
        game = Game()
        game_history = []  # (state, action, player) tuples
        
        max_turns = 200
        for turn in range(max_turns):
            # Player 1's turn
            state1 = self.get_state_dict(game, 1)
            valid_mask1 = self.get_valid_actions(game, 1)
            action1_idx, boost1 = self.select_action(state1, valid_mask1, epsilon)
            dir1 = self.action_idx_to_direction(action1_idx)
            
            # Save state before action
            board1, features1 = self.encoder.encode_state(state1)
            state1_flat = np.concatenate([board1.flatten(), features1])
            
            # Player 2's turn
            state2 = self.get_state_dict(game, 2)
            valid_mask2 = self.get_valid_actions(game, 2)
            action2_idx, boost2 = self.select_action(state2, valid_mask2, epsilon)
            dir2 = self.action_idx_to_direction(action2_idx)
            
            # Save state before action
            board2, features2 = self.encoder.encode_state(state2)
            state2_flat = np.concatenate([board2.flatten(), features2])
            
            # Execute moves
            result = game.step(dir1, dir2, boost1, boost2)
            
            # Save to history
            game_history.append((state1_flat, action1_idx, 1, boost1))
            game_history.append((state2_flat, action2_idx, 2, boost2))
            
            # Check if game ended
            if result is not None:
                # Assign rewards based on result
                if save_experience:
                    self.process_game_history(game_history, result)
                
                # Update stats
                self.games_played += 1
                if result == GameResult.AGENT1_WIN:
                    self.wins_p1 += 1
                elif result == GameResult.AGENT2_WIN:
                    self.wins_p2 += 1
                else:
                    self.draws += 1
                
                return result
        
        # Max turns reached
        if save_experience:
            self.process_game_history(game_history, GameResult.DRAW)
        self.games_played += 1
        self.draws += 1
        return GameResult.DRAW
    
    def process_game_history(self, history: List, result: GameResult):
        """Process game history and add experiences to buffer."""
        # Assign rewards
        if result == GameResult.AGENT1_WIN:
            p1_reward = 1.0
            p2_reward = -1.0
        elif result == GameResult.AGENT2_WIN:
            p1_reward = -1.0
            p2_reward = 1.0
        else:  # Draw
            p1_reward = 0.0
            p2_reward = 0.0
        
        # Add experiences to buffer
        for i, (state, action, player, boost) in enumerate(history):
            # Determine reward for this player
            reward = p1_reward if player == 1 else p2_reward
            
            # Get next state (if exists)
            if i + 2 < len(history):
                # Next state is two steps ahead (after both players moved)
                next_state = history[i + 2][0]
                done = False
            else:
                # Game ended
                next_state = state  # Dummy
                done = True
            
            # Add to buffer
            self.experience_buffer.push(state, action, reward, next_state, done)
    
    def train_step(self, batch_size: int = 64) -> dict:
        """Perform one training step."""
        if len(self.experience_buffer) < batch_size:
            return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}
        
        # Sample batch
        states, actions, rewards, next_states, dones = \
            self.experience_buffer.sample(batch_size)
        
        # Train
        total_loss, policy_loss, value_loss = self.trainer.train_step(
            states, actions, rewards, next_states, dones
        )
        
        return {
            "loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss
        }
    
    def train(self, num_games: int = 1000, batch_size: int = 64, 
              train_every: int = 10, save_every: int = 100):
        """
        Main training loop.
        
        Args:
            num_games: Number of self-play games
            batch_size: Training batch size
            train_every: Train every N games
            save_every: Save model every N games
        """
        print(f"Starting training for {num_games} games...")
        print(f"Batch size: {batch_size}, Train every: {train_every}, Save every: {save_every}")
        
        for game_num in range(num_games):
            # Decay epsilon (exploration rate)
            epsilon = max(0.05, 0.5 - game_num / num_games * 0.45)
            
            # Play game
            result = self.play_game(epsilon=epsilon, save_experience=True)
            
            # Train periodically
            if game_num > 0 and game_num % train_every == 0:
                losses = []
                for _ in range(5):  # Multiple training steps per game batch
                    loss_dict = self.train_step(batch_size)
                    losses.append(loss_dict["loss"])
                avg_loss = np.mean(losses) if losses else 0.0
                
                # Print stats
                win_rate_p1 = self.wins_p1 / self.games_played if self.games_played > 0 else 0
                win_rate_p2 = self.wins_p2 / self.games_played if self.games_played > 0 else 0
                draw_rate = self.draws / self.games_played if self.games_played > 0 else 0
                
                print(f"Game {game_num}/{num_games} | Loss: {avg_loss:.4f} | "
                      f"P1 Win: {win_rate_p1:.2%} | P2 Win: {win_rate_p2:.2%} | "
                      f"Draw: {draw_rate:.2%} | Buffer: {len(self.experience_buffer)}")
            
            # Save model periodically
            if game_num > 0 and game_num % save_every == 0:
                save_model(self.model, self.model_path)
                print(f"Model saved at game {game_num}")
        
        # Final save
        save_model(self.model, self.model_path)
        print(f"\nTraining complete! Final stats:")
        print(f"  Games played: {self.games_played}")
        print(f"  P1 wins: {self.wins_p1} ({self.wins_p1/self.games_played:.2%})")
        print(f"  P2 wins: {self.wins_p2} ({self.wins_p2/self.games_played:.2%})")
        print(f"  Draws: {self.draws} ({self.draws/self.games_played:.2%})")
        print(f"  Buffer size: {len(self.experience_buffer)}")


def quick_train(num_games=500):
    """Quick training for testing."""
    trainer = SelfPlayTrainer()
    trainer.train(num_games=num_games, batch_size=32, train_every=5, save_every=100)


def full_train(num_games=5000):
    """Full training for competition."""
    trainer = SelfPlayTrainer()
    trainer.train(num_games=num_games, batch_size=64, train_every=10, save_every=500)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "quick":
            print("Running quick training (500 games)...")
            quick_train()
        elif mode == "full":
            print("Running full training (5000 games)...")
            full_train()
        else:
            print(f"Unknown mode: {mode}")
            print("Usage: python train_model.py [quick|full]")
    else:
        print("Running default training (1000 games)...")
        trainer = SelfPlayTrainer()
        trainer.train(num_games=1000)
