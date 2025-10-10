# Troubleshooting Guide

Common problems and their solutions for all AI modules.

## Installation Issues

### "pip: command not found"

**Problem:** pip is not installed or not in PATH

**Solutions:**

```bash
# Try pip3
pip3 install -r requirements.txt

# Or use python -m
python -m pip install -r requirements.txt

# Or python3 -m
python3 -m pip install -r requirements.txt
```

### "pygame not found" After Installation

**Problem:** Installed in different Python version than you're running

**Check which Python:**
```bash
which python
which python3
python --version
python3 --version
```

**Solution:**
```bash
# Use the same Python for both install and run
python3 -m pip install pygame
python3 scripts/run_search.py
```

### "No module named 'modules'"

**Problem:** Running script from wrong directory

**Solution:**
```bash
# Make sure you're in project root
cd intro-ai
ls  # Should see: modules/, scripts/, web/

# Run from project root
python scripts/run_search.py
```

**OR if scripts were moved:**
Check that scripts have correct path calculation:
```python
# Should be in run_*.py:
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
```

### Permission Errors

**macOS/Linux:**
```bash
# Install with --user flag
pip install --user -r requirements.txt

# Or use sudo (not recommended)
sudo pip install -r requirements.txt
```

**Windows:**
- Run command prompt as Administrator
- Or use `--user` flag

## Runtime Errors

### "pygame.error: No video device"

**Problem:** Running on server/headless system

**Solution:**
Use web version instead, or set up virtual display:
```bash
# Linux with Xvfb
xvfb-run python scripts/run_search.py
```

### Window Doesn't Open / Black Screen

**macOS specific:**

**Solution:**
```bash
# Give terminal accessibility permissions
# System Preferences → Security & Privacy → Privacy → Accessibility
# Add Terminal.app or your Python IDE
```

**General:**
```bash
# Check pygame works
python -c "import pygame; pygame.init(); print('OK')"
```

### Module Crashes Immediately

**Check for errors:**
```bash
python scripts/run_search.py 2>&1 | more
# Read the full error message
```

**Common causes:**
- Import error (missing dependency)
- File path issue
- Config error

## Search Module Issues

### Algorithm Not Starting

**Symptom:** Press 1-5, nothing happens

**Checks:**
1. Make sure window has focus (click on it)
2. Look at sidebar - does it show algorithm name?
3. Check console for error messages

**Solution:**
```bash
# Try with simpler maze
python scripts/run_search.py --preset simple
```

### Maze Looks Weird / Too Small

**Problem:** Cell size or maze size misconfigured

**Solution:**
```python
# In scripts/run_search.py, add before main():
from modules.search.config import config
config.CELL_SIZE = 20  # Adjust cell size
config.MAZE_WIDTH = 40  # Adjust dimensions
```

### Can't See Frontier/Explored Cells

**Check display toggles:**
```python
# In config.py
config.SHOW_EXPLORED = True
config.SHOW_FRONTIER = True
```

### Heuristic Weight Slider Not Working

**Symptom:** Adjust slider, click Apply, nothing changes

**Check:**
1. Is A* or Greedy running? (Only affects those)
2. Did you click Apply button?
3. Watch console for "Restarting..." message

**If still broken:** Check `modules/search/main.py:on_parameter_change()` function

## MDP Module Issues

### Value Iteration Doesn't Start

**Symptom:** Press SPACE, nothing happens

**Check:**
1. Window has focus
2. Console doesn't show errors
3. Try pressing S to step manually

### Values Don't Converge

**Symptom:** Values keep changing, never stabilize

**Check:**
```python
# In modules/mdp/core/solver.py
CONVERGENCE_THRESHOLD = 0.001  # Should be small
MAX_ITERATIONS = 1000  # Should be large enough
```

### Policy Arrows Not Showing

**Press P key** to toggle policy display

**If still not showing:**
- Check console for errors
- Verify value iteration completed
- Check config: `SHOW_POLICY = True`

### Grid Too Small / Too Large

```bash
# Adjust grid size
python scripts/run_mdp.py --grid-size 8  # Larger
python scripts/run_mdp.py --grid-size 4  # Smaller
```

## Reinforcement Learning Issues

### Snake Doesn't Improve

**Symptom:** Average score stuck at 0-1 after 500 episodes

**Diagnose:**

1. **Check epsilon is decaying:**
   - Watch ε in sidebar
   - Should go from 1.0 → 0.01
   - If stuck at 1.0, epsilon decay not working

2. **Check learning rate:**
   - Too low (0.0001)? Learning too slow
   - Try 0.01 or 0.05

3. **Check Q-table is growing:**
   - Should reach 1000+ states by episode 500
   - If stuck at small number, state representation issue

**Solutions:**
```bash
# Try fast learning preset
python scripts/run_rl.py --preset fast_learning

# Or increase learning rate manually
python scripts/run_rl.py --lr 0.05 --episodes 1000
```

### Performance Decreases After Improving

**Symptom:** Average score goes 5 → 8 → 3

**Causes:**
1. **Learning rate too high** - Q-values oscillating
   - Solution: Lower α to 0.001

2. **Too much exploration** - ε not decaying enough
   - Solution: Faster decay (0.98 instead of 0.995)

3. **Normal variation** - Randomness in training
   - Solution: Train longer, average will stabilize

### Training is Too Slow

**Symptom:** Want results faster

**Solutions:**
```bash
# Use turbo preset (200 steps/second)
python scripts/run_rl.py --preset turbo

# Or manually set speed
python scripts/run_rl.py --speed 200

# Reduce episodes if just testing
python scripts/run_rl.py --episodes 200
```

### Can't Load Saved Model

**Symptom:** Press L or use --load, file not found

**Check:**
```bash
ls models/
# Should see: q_table.json
```

**If missing:**
- Train first, then press S to save
- Or train with `--episodes 100` then auto-saves

**If exists but won't load:**
- Check file isn't corrupted (open in text editor)
- Check JSON format is valid
- Try training fresh model

### Q-Values All Zero or Strange

**Problem:** Learning not happening

**Checks:**
1. Training mode enabled? (not inference mode)
2. Learning rate > 0?
3. Rewards configured correctly?

**Debug:**
```python
# Add print in learn() method
print(f"TD error: {td_error}, reward: {reward}")
```

## Web Version Issues

### Blank Page in Browser

**Check:**
1. Wait 30 seconds (first load is slow)
2. Open browser console (F12) for errors
3. Try different browser (Chrome recommended)
4. Clear cache and reload

**Common causes:**
- NumPy imports (web version must be NumPy-free)
- Double pygame.init() calls
- Autorun not enabled in index.html

### Web Version Loads but Freezes

**Symptoms:** Loading progress bar completes, then nothing

**Checks:**
1. Console shows errors?
2. "Ready to start!" message appears?
3. Try clicking on page (user interaction needed)

**Solutions:**
- Check main.py has `await asyncio.sleep(0)` in loop
- Verify autorun = 1 in index.html
- Check no blocking time.sleep() calls

### Web Version Works Locally but Not on GitHub Pages

**Problem:** Deployment issue

**Check:**
1. Files at correct path on gh-pages branch?
2. index.html, .apk file both present?
3. GitHub Pages enabled in repo settings?

**Verify deployment:**
```bash
git checkout gh-pages
ls reinforcement_learning/
# Should show: index.html, reinforcement_learning.apk, favicon.png
```

## Performance Issues

### Low FPS / Laggy Visualization

**Solutions:**

```python
# In config.py
config.FPS = 30  # Instead of 60
config.CELL_SIZE = 15  # Smaller cells
config.MAZE_WIDTH = 30  # Smaller maze
```

### Training Crashes After Many Episodes

**Symptom:** Memory error after 5000+ episodes

**Cause:** Q-table too large (millions of states)

**Solutions:**
1. Reduce state space dimensionality
2. Limit Q-table size (max entries)
3. Use episodic training (save/restart)

### Computer Fan Loud / Hot

**Normal!** Training is computation-intensive

**Reduce load:**
- Lower FPS (30 instead of 60)
- Reduce game speed
- Train in batches with breaks

## Common Error Messages

### "IndexError: list index out of range"

**Common in:** Snake environment

**Cause:** Trying to access state outside boundaries

**Check:**
- Collision detection working?
- Boundary checks in place?

### "KeyError" in Q-table

**Cause:** State tuple not in Q-table (shouldn't happen with defaultdict)

**If happens:**
- Check state is tuple, not list
- Check all state values are hashable

### "AttributeError: 'NoneType' object has no attribute..."

**Common in:** Visualizer when algorithm not selected

**Solution:**
- Check current_algorithm is not None before using
- Press 1-5 to select algorithm first

## Platform-Specific Issues

### macOS: "Python quit unexpectedly"

**Cause:** Usually pygame/SDL issue

**Solution:**
```bash
# Reinstall pygame
pip uninstall pygame
pip install pygame

# Or use --no-cache-dir
pip install --no-cache-dir pygame
```

### Windows: "DLL load failed"

**Cause:** Missing Visual C++ redistributables

**Solution:**
1. Install [Microsoft Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)
2. Reinstall pygame
3. Or use pre-compiled wheels

### Linux: "No module named '_tkinter'"

**Not actually needed** (we use pygame, not tkinter)

**But if matplotlib needs it:**
```bash
# Ubuntu/Debian
sudo apt-get install python3-tk

# Fedora
sudo dnf install python3-tkinter
```

## Getting More Help

### 1. Check Console Output

**Always read error messages:**
```bash
python scripts/run_search.py 2>&1 | tee output.log
# Saves output to output.log for analysis
```

### 2. Enable Debug Mode

```python
# In main.py, add at top
import traceback
import sys

try:
    main()
except Exception as e:
    traceback.print_exc()
    sys.exit(1)
```

### 3. Minimal Reproduction

**Isolate the problem:**
```python
# Test just the import
python -c "from modules.search.core.environment import Maze; print('OK')"

# Test just maze creation
python -c "from modules.search.core.environment import Maze; m = Maze(10,10); print('OK')"
```

### 4. Check Dependencies

```bash
# List installed packages
pip list | grep -E 'pygame|numpy|matplotlib'

# Should see versions
# pygame 2.x
# numpy 1.x
# matplotlib 3.x
```

### 5. Start Fresh

**Last resort:**
```bash
# Create new virtual environment
python3 -m venv env
source env/bin/activate  # or: env\Scripts\activate on Windows

# Install fresh
pip install pygame numpy matplotlib

# Test
python scripts/run_search.py
```

## Still Stuck?

1. **Search this wiki** - Specific error might be documented
2. **Check module README** - Module-specific troubleshooting
3. **Ask in class** - Instructor and TAs can help
4. **GitHub Issues** - Report bugs with:
   - Full error message
   - Operating system
   - Python version
   - Steps to reproduce

---

**Most issues are:** Path problems, missing dependencies, or Python version mismatches. Start with the basics!
