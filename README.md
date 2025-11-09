# Case Closed - Winning Agent Strategy 🏆

## Overview

This agent uses **advanced pathfinding and territory control** to dominate the Case Closed challenge. The strategy focuses on:

1. **Collision Avoidance**: BFS-based lookahead to avoid immediate crashes
2. **Space Control**: Maximizes reachable territory (Voronoi-style)
3. **Strategic Boosts**: Uses speed boosts when space gets tight
4. **Opponent Prediction**: Anticipates opponent moves to avoid head-on collisions

## Files Structure

```
├── agent.py                  # Main agent with winning strategy ⭐
├── pathfinding_utils.py      # Pathfinding & evaluation utilities
├── case_closed_game.py       # Game engine (provided)
├── judge_engine.py           # Match orchestrator (provided)
├── sample_agent.py           # Basic agent example (provided)
├── local-tester.py           # API compliance tester (provided)
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Test API Compliance

Make sure your agent responds correctly to judge requests:

```bash
# Terminal 1: Start your agent
python agent.py

# Terminal 2: Run compliance tests
python local-tester.py
```

Expected output:
```
✅ PASS: Initial Latency Check
✅ PASS: Receive Game State
✅ PASS: Send Move
✅ PASS: Send Move with Boost Format
✅ PASS: End Game Notification
🎉 CONGRATULATIONS! Your agent passed all API compliance checks!
```

### 3. Run Full Game Simulation

Test your agent against the sample agent:

```bash
# Terminal 1: Start your agent (Player 1)
python agent.py

# Terminal 2: Start opponent (Player 2)
PORT=5009 python sample_agent.py

# Terminal 3: Run the judge
python judge_engine.py
```

Watch the match unfold in Terminal 3! You'll see the board state, trail lengths, and boost usage.

### 4. Test Against Yourself

Run two instances of your agent:

```bash
# Terminal 1: Agent on port 5008
python agent.py

# Terminal 2: Same agent on different port
PORT=5009 python agent.py

# Terminal 3: Run judge with custom URLs
PLAYER1_URL=http://localhost:5008 PLAYER2_URL=http://localhost:5009 python judge_engine.py
```

## Strategy Breakdown

### Core Algorithm

The `send_move()` function implements this decision flow:

1. **Get Game State**
   - Extract my trail, opponent trail, turn count, boosts remaining

2. **Evaluate All Valid Moves**
   - For each direction (excluding opposite):
     - Check immediate safety (no collision)
     - Count reachable spaces using BFS (depth=15)
     - Add bonuses for continuing straight
     - Penalize head-on collision predictions

3. **Select Best Move**
   - Pick direction with highest score (most space)
   - Decide if boost should be used based on:
     - Turn count (save for mid/late game)
     - Space pressure (use when trapped)
     - Boost availability

4. **Return Move**
   - Format: `"DIRECTION"` or `"DIRECTION:BOOST"`

### Key Functions (pathfinding_utils.py)

#### `evaluate_move_safety()`
Checks if a move is safe and calculates reachable space:
```python
is_safe, reachable = pathfinder.evaluate_move_safety(
    current_pos, direction, my_trail, other_trail
)
```

#### `count_reachable_spaces()`
BFS to count accessible cells (territory control):
```python
spaces = pathfinder.count_reachable_spaces(
    start_pos, my_trail, other_trail, max_depth=15
)
```

#### `should_use_boost()`
Smart boost logic based on game phase:
- **Early game (0-40 turns)**: Save boosts
- **Mid game (40-150)**: Use if space < 40% or reachable < 30
- **Late game (150-200)**: Use remaining boosts

#### `predict_opponent_next_pos()`
Assumes opponent continues straight (most likely):
```python
predicted_pos = pathfinder.predict_opponent_next_pos(opponent_trail)
```

## Configuration

### Change Participant/Agent Name

Edit `agent.py`:
```python
PARTICIPANT = "YourTeamName"
AGENT_NAME = "YourAgentName"
```

### Adjust Strategy Parameters

In `pathfinding_utils.py`, tweak these values:

```python
# BFS depth for space counting
max_depth = 15  # Higher = more thorough, slower

# Boost thresholds
if turn_count < 40:  # Early game cutoff
if space_ratio < 0.4:  # Space pressure threshold
if reachable_spaces < 30:  # Tightness threshold
```

### Change Server Port

```bash
PORT=5010 python agent.py
```

Or in code:
```python
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5008"))
```

## Debugging Tips

### Print Debug Info

Add logging in `send_move()`:

```python
print(f"Turn {turn_count}: Evaluating moves...")
for direction in valid_directions:
    is_safe, reachable = pathfinder.evaluate_move_safety(...)
    print(f"  {direction}: safe={is_safe}, reachable={reachable}")
print(f"Chose: {move}")
```

### Visualize Board State

The judge displays the board each turn:
```
. . . . . . . . . . . . . . . . . . . .
. A A . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . A A .
...
```

### Check Game String

At game end, the judge prints the move sequence:
```
Game String: 1RB-2L-1R-2LB-1U-...
```
Format: `{player}{direction}{B=boost}{R=random}-`

## Common Issues

### "Could not connect to agent"
- Ensure agent is running: `python agent.py`
- Check port matches: default is 5008
- Verify no firewall blocking localhost

### "Invalid move format"
- Must return `{"move": "DIRECTION"}` or `{"move": "DIRECTION:BOOST"}`
- Valid directions: UP, DOWN, LEFT, RIGHT
- Case-insensitive but uppercase preferred

### "Agent forfeited"
- Both move attempts failed
- Ran out of random backup moves (5 total)
- Check your logic for edge cases

### Import Error for pathfinding_utils
- Ensure `pathfinding_utils.py` is in same directory as `agent.py`
- Or install as package: `pip install -e .`

## Performance Optimization

### Speed Up BFS

Reduce max_depth in `count_reachable_spaces()`:
```python
reachable = pathfinder.count_reachable_spaces(
    next_pos, simulated_trail, other_trail, max_depth=10  # Faster
)
```

### Cache Calculations

Store frequently accessed data:
```python
# In send_move()
if not hasattr(send_move, "move_cache"):
    send_move.move_cache = {}
```

### Simplify Scoring

Remove less critical scoring factors:
```python
# Remove opponent prediction if too slow
# Just focus on reachable space
score = reachable
```

## Submission Checklist

✅ **API Compliance**: All tests pass in `local-tester.py`  
✅ **Full Game Test**: Beats sample_agent.py consistently  
✅ **Dependencies**: All imports in `requirements.txt`  
✅ **Docker Build**: Test with provided Dockerfile  
✅ **Participant Name**: Updated in `agent.py`  
✅ **No Edits**: Only `agent.py` and new files modified  
✅ **GitHub Repo**: Pushed to GitHub, ready for Devpost  

## Advanced Strategies (Future Improvements)

### 1. Minimax Search
Implement 2-3 ply minimax to predict opponent moves more accurately.

### 2. Territory Partitioning
Use flood fill to calculate controlled vs opponent-controlled regions.

### 3. Endgame Optimization
When turns > 180, switch to pure trail-length maximization mode.

### 4. Opponent Modeling
Track opponent's boost usage patterns to predict their strategy.

### 5. Machine Learning
Train a neural network to evaluate board positions (requires pre-training).

## Resources

- **Game Rules**: See `Case_Closed_Challenge_Description.pdf`
- **Game Engine**: Read `case_closed_game.py` for exact mechanics
- **Judge Protocol**: Study `judge_engine.py` for API flow
- **Example Agent**: Check `sample_agent.py` for basic structure

## Support

If you encounter issues:
1. Re-run `local-tester.py` to verify API compliance
2. Check terminal output for error messages
3. Add debug prints to trace execution
4. Test with different opponent agents

## License

This agent implementation is provided for the Case Closed Challenge. Use and modify as needed for the competition.

---

**Good luck, detective! May your trails be long and your collisions be few.** 🕵️‍♂️🏁
