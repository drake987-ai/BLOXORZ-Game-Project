# Bloxorz

A Panda3D implementation of Bloxorz (fragile tiles, bridges, soft/heavy/split switches) with four built-in solvers: BFS, DFS, UCS and A*.

## Requirements

- Python **3.10 or newer** (the code uses `dataclass(slots=True)` and `X | Y` type hints; it will not run on 3.9)
- `panda3d==1.10.16` (listed in `requirements.txt`)
- A machine with a display and OpenGL support (a desktop window opens)

## Setup

From the `Source/` folder:

```bash
# 1. Create and activate a virtual environment (recommended)
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install the dependency
python -m pip install -r requirements.txt
```

Any Python 3.10+ works in place of `python3.12`. Check yours with `python3 --version`.

## Run

Still in `Source/`, with the environment active:

```bash
python main.py
```

The game opens on the first level. Always start it from `Source/` (or by its full path), because it loads levels relative to `main.py`.

## Controls

| Input | Action |
|---|---|
| `W` `A` `S` `D` / arrow keys | Roll the block up / left / down / right |
| `Space` | Swap the active cube (only while the block is split) |
| `R` | Restart the level |
| `N` | Open or close the level selector (New Game) |
| `Esc` | Close the selector, otherwise quit |
| `Q` / `E` (hold) | Rotate the camera left / right (trackpad-friendly) |
| Mouse wheel | Zoom |
| Right mouse button + drag | Pan the camera |
| `C` / `F` | Reset the camera / focus on the block |

On-screen buttons: **New Game** and **Restart** (bottom-left), **BFS**, **DFS**, **UCS**, **A\*** (bottom-right).

Text is rendered in Poppins (bundled in `gui/fonts/`, SIL Open Font License).

## Using the solvers

1. Pick a level, and optionally play a few moves first.
2. Click **BFS**, **DFS**, **UCS** or **A\*** from the current state.
3. A results card (top-right) shows Search Time, Memory Usage, Expanded Nodes and Solution Length (Moves; `SWAP` is not counted).
4. The found path is checked, then played automatically. Playback controls (**Step Back**, **Play**, **Pause**, **Step Forward**) appear above the status line. Making a manual move leaves the solution path.

| Solver | Finds | Notes |
|---|---|---|
| BFS | Fewest actions | `SWAP` counts as an action |
| DFS | Any solution | Not optimal, so paths can be long |
| UCS | Lowest risk-weighted cost | Landing on a fragile tile costs +4; roll = 1, `SWAP` = 0 (see `ucs_costs.md`) |
| A* | Lowest risk-weighted cost | Same cost as UCS, with a centre-distance heuristic; fewer expansions |

Test All (`test_all.json`, always last in the menu) is the hardest level and is a good one to try first: it uses every mechanic.

## Levels

Levels live in `levels/` as JSON files, and `levels/campaign.json` lists them in order (`test1.json` … `test34.json`, then `test_all.json`). To add a level, create a new JSON file and add its filename to `campaign.json`. Every level is validated at startup, so a malformed file stops the program with an error message naming the problem.

Tile symbols:

| Symbol | Meaning |
|---|---|
| `#` | Void (empty) |
| `.` | Floor |
| `S` / `G` | Start / goal (exactly one of each) |
| `F` | Fragile (the block may lie on it, not stand on it) |
| `B` | Bridge (defined in `bridges`) |
| `o` | Soft switch (any part of the block, or a cube, triggers it) |
| `X` | Heavy switch (only a standing block triggers it) |
| `P` | Split switch (a standing block splits into two cubes) |

## Project layout

```
Source/
├── main.py            entry point
├── requirements.txt
├── models/            Level, GameState, level catalog
├── engine/            game rules (transition, successors, is_goal)
├── solvers/           bfs.py, dfs.py, ucs.py, astar.py, common.py
├── gui/               Panda3D app, animation, geometry, meshes
└── levels/            campaign.json, test1–test34.json and test_all.json
```

See `codebase_structure.md` for how the modules interact and `algorithm_logic.md` for the solver code and explanations (both in the repository root).

## Troubleshooting

| Error | Cause and fix |
|---|---|
| `TypeError: dataclass() got an unexpected keyword argument 'slots'` | Python is older than 3.10. Use a newer interpreter. |
| `ModuleNotFoundError: No module named 'direct'` (or `panda3d`) | Panda3D is not installed in the interpreter you ran. Install it with the same interpreter: `python -m pip install -r requirements.txt`. |
| A solver button does nothing and the status says "not implemented" | The solver file still contains the original stub. |
