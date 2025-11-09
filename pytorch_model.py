"""
PyTorch Neural Network for Case Closed Agent
Uses a policy network to predict best moves based on game state.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List

class CaseClosedNet(nn.Module):
    """
    Neural network for evaluating board positions and predicting moves.
    
    Architecture:
    - Input: Board state + game features (flattened)
    - Hidden layers: 3 fully connected layers with ReLU
    - Output: Action probabilities (4 directions) + value estimation
    """
    
    def __init__(self, board_height=18, board_width=20):
        super(CaseClosedNet, self).__init__()
        
        self.board_height = board_height
        self.board_width = board_width
        
        # Input features:
        # - Board grid (18x20 = 360 cells, 3 channels: empty/my_trail/opponent_trail)
        # - Position features (4: my_x, my_y, opp_x, opp_y normalized)
        # - Game state (6: turn/200, my_boosts/3, opp_boosts/3, my_length, opp_length, space_ratio)
        input_size = (board_height * board_width * 3) + 4 + 6
        
        # Network layers
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        
        # Policy head (action probabilities)
        self.policy_head = nn.Linear(128, 5)  # 4 directions + boost decision
        
        # Value head (position evaluation)
        self.value_head = nn.Linear(128, 1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        """Forward pass through the network."""
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        
        # Policy output (action probabilities)
        policy = self.policy_head(x)
        policy = F.softmax(policy, dim=-1)
        
        # Value output (position evaluation)
        value = torch.tanh(self.value_head(x))
        
        return policy, value
    
    def predict_move(self, board_state, features, valid_actions_mask):
        """
        Predict best move given board state.
        
        Args:
            board_state: (board_height, board_width, 3) numpy array
            features: (10,) numpy array of game features
            valid_actions_mask: (5,) boolean array indicating valid actions
        
        Returns:
            action_idx: Index of chosen action (0-4)
            action_probs: Probability distribution over actions
            value: Estimated value of position
        """
        self.eval()
        with torch.no_grad():
            # Flatten board state
            board_flat = board_state.flatten()
            
            # Concatenate all features
            input_features = np.concatenate([board_flat, features])
            input_tensor = torch.FloatTensor(input_features).unsqueeze(0)
            
            # Forward pass
            policy, value = self.forward(input_tensor)
            
            # Apply valid actions mask
            policy = policy.squeeze(0).numpy()
            policy = policy * valid_actions_mask
            
            # Renormalize
            policy_sum = policy.sum()
            if policy_sum > 0:
                policy = policy / policy_sum
            else:
                # If all masked, use uniform over valid actions
                policy = valid_actions_mask.astype(float)
                policy = policy / policy.sum()
            
            # Choose action (greedy or stochastic)
            action_idx = np.argmax(policy)
            
            return action_idx, policy, value.item()


class ExperienceBuffer:
    """Stores experiences for training."""
    
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """Sample random batch of experiences."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class ModelTrainer:
    """Handles model training and optimization."""
    
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.policy_loss_fn = nn.CrossEntropyLoss()
        self.value_loss_fn = nn.MSELoss()
    
    def train_step(self, states, actions, rewards, next_states, dones):
        """Perform one training step."""
        self.model.train()
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.LongTensor(actions)
        rewards_tensor = torch.FloatTensor(rewards)
        next_states_tensor = torch.FloatTensor(next_states)
        dones_tensor = torch.FloatTensor(dones)
        
        # Forward pass
        policy_pred, value_pred = self.model(states_tensor)
        _, next_value = self.model(next_states_tensor)
        
        # Calculate losses
        # Policy loss: Cross-entropy (supervised by better moves)
        policy_loss = self.policy_loss_fn(policy_pred, actions_tensor)
        
        # Value loss: TD error
        target_value = rewards_tensor + 0.99 * next_value.squeeze() * (1 - dones_tensor)
        value_loss = self.value_loss_fn(value_pred.squeeze(), target_value.detach())
        
        # Combined loss
        total_loss = policy_loss + 0.5 * value_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return total_loss.item(), policy_loss.item(), value_loss.item()


def save_model(model, path):
    """Save model weights to disk."""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(model, path):
    """Load model weights from disk."""
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    print(f"Model loaded from {path}")
    return model


# Example usage
if __name__ == "__main__":
    # Create model
    model = CaseClosedNet()
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    test_input = torch.randn(1, (18 * 20 * 3) + 4 + 6)
    policy, value = model(test_input)
    print(f"Policy shape: {policy.shape}, Value shape: {value.shape}")
    print(f"Policy probs: {policy}")
    print(f"Value: {value.item()}")
