"""
Pathfinding utilities for Case Closed agent.
Provides collision detection, space evaluation, and strategic move selection.
"""

from collections import deque
from typing import List, Tuple, Set, Optional

class PathfindingUtils:
    def __init__(self, board_height=18, board_width=20):
        self.height = board_height
        self.width = board_width
    
    def torus_normalize(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        """Normalize position for torus wraparound."""
        x, y = pos
        return (x % self.width, y % self.height)
    
    def get_next_position(self, current: Tuple[int, int], direction: str) -> Tuple[int, int]:
        """Calculate next position given current position and direction."""
        x, y = current
        moves = {
            "UP": (x, y - 1),
            "DOWN": (x, y + 1),
            "LEFT": (x - 1, y),
            "RIGHT": (x + 1, y)
        }
        next_pos = moves.get(direction, current)
        return self.torus_normalize(next_pos)
    
    def is_opposite_direction(self, current_dir: str, new_dir: str) -> bool:
        """Check if new direction is opposite to current (invalid move)."""
        opposites = {
            "UP": "DOWN", "DOWN": "UP",
            "LEFT": "RIGHT", "RIGHT": "LEFT"
        }
        return opposites.get(current_dir) == new_dir
    
    def get_valid_directions(self, current_dir: str) -> List[str]:
        """Get all valid directions (excluding opposite)."""
        all_dirs = ["UP", "DOWN", "LEFT", "RIGHT"]
        opposites = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}
        return [d for d in all_dirs if d != opposites.get(current_dir)]
    
    def is_safe_position(self, pos: Tuple[int, int], my_trail: List, other_trail: List, 
                        allow_head: bool = False) -> bool:
        """Check if a position is safe (not occupied by any trail)."""
        pos = self.torus_normalize(pos)
        my_trail_set = set(tuple(p) for p in my_trail)
        other_trail_set = set(tuple(p) for p in other_trail)
        
        # Allow checking the head position (current position)
        if allow_head and my_trail and pos == tuple(my_trail[-1]):
            return True
        
        return pos not in my_trail_set and pos not in other_trail_set
    
    def count_reachable_spaces(self, start_pos: Tuple[int, int], my_trail: List, 
                               other_trail: List, max_depth: int = 15) -> int:
        """
        BFS to count reachable empty spaces from a starting position.
        This helps evaluate how much "room" we have to maneuver.
        """
        visited = set()
        queue = deque([(self.torus_normalize(start_pos), 0)])
        visited.add(self.torus_normalize(start_pos))
        
        my_trail_set = set(tuple(p) for p in my_trail)
        other_trail_set = set(tuple(p) for p in other_trail)
        occupied = my_trail_set | other_trail_set
        
        count = 0
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # UP, DOWN, LEFT, RIGHT
        
        while queue:
            pos, depth = queue.popleft()
            if depth >= max_depth:
                continue
            
            for dx, dy in directions:
                next_pos = self.torus_normalize((pos[0] + dx, pos[1] + dy))
                
                if next_pos not in visited and next_pos not in occupied:
                    visited.add(next_pos)
                    queue.append((next_pos, depth + 1))
                    count += 1
        
        return count
    
    def evaluate_move_safety(self, current_pos: Tuple[int, int], direction: str,
                            my_trail: List, other_trail: List) -> Tuple[bool, int]:
        """
        Evaluate safety of a move by checking:
        1. Immediate collision
        2. Reachable space after the move
        
        Returns: (is_immediately_safe, reachable_spaces)
        """
        next_pos = self.get_next_position(current_pos, direction)
        
        # Check immediate safety
        is_safe = self.is_safe_position(next_pos, my_trail, other_trail)
        
        if not is_safe:
            return (False, 0)
        
        # Check reachable space after this move
        # Simulate adding this position to our trail
        simulated_trail = my_trail + [next_pos]
        reachable = self.count_reachable_spaces(next_pos, simulated_trail, other_trail)
        
        return (True, reachable)
    
    def get_current_direction(self, trail: List) -> str:
        """Determine current direction from trail history."""
        if len(trail) < 2:
            return "RIGHT"  # Default
        
        prev = trail[-2]
        current = trail[-1]
        dx = current[0] - prev[0]
        dy = current[1] - prev[1]
        
        # Handle torus wrapping
        if abs(dx) > 1:
            dx = -1 if dx > 0 else 1
        if abs(dy) > 1:
            dy = -1 if dy > 0 else 1
        
        if dx == 1:
            return "RIGHT"
        elif dx == -1:
            return "LEFT"
        elif dy == 1:
            return "DOWN"
        elif dy == -1:
            return "UP"
        
        return "RIGHT"
    
    def predict_opponent_next_pos(self, opponent_trail: List) -> Optional[Tuple[int, int]]:
        """Predict where opponent will likely move next."""
        if len(opponent_trail) < 2:
            return None
        
        current_dir = self.get_current_direction(opponent_trail)
        current_pos = opponent_trail[-1]
        
        # Assume opponent continues straight (most common)
        return self.get_next_position(current_pos, current_dir)
    
    def should_use_boost(self, my_trail: List, other_trail: List, boosts_remaining: int,
                        turn_count: int, reachable_spaces: int) -> bool:
        """
        Decide whether to use a speed boost based on:
        - Available boosts
        - Current game phase
        - Space pressure
        """
        if boosts_remaining <= 0:
            return False
        
        # Early game: save boosts
        if turn_count < 40:
            return False
        
        # Late game: use remaining boosts
        if turn_count > 150 and boosts_remaining > 0:
            return True
        
        # Mid game: use if space is getting tight
        total_cells = self.height * self.width
        occupied_cells = len(my_trail) + len(other_trail)
        space_ratio = (total_cells - occupied_cells) / total_cells
        
        # If less than 40% space remaining and we have low reachable space, boost!
        if space_ratio < 0.4 and reachable_spaces < 30:
            return True
        
        # If we're in a very tight spot, use boost to escape
        if reachable_spaces < 15 and boosts_remaining >= 2:
            return True
        
        return False
