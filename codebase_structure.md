# Codebase Structure

A Panda3D implementation of Bloxorz (1×1×2 block, fragile tiles, bridges, soft/heavy/split switches) plus four search solvers. The code needs Python 3.10+ (`dataclass(slots=True)`, `X | Y` type hints). The only dependency is `panda3d==1.10.16`, for the GUI.

## 1. Directory tree

```
CSAI-Lab-Bloxorz/
├── README.md                    one-line project description
├── bloxorz.pdf                  assignment specification
├── codebase_structure.md        (this file)
├── algorithm_logic.md           solver code + explanation
└── Source/
    ├── main.py                  entry point
    ├── requirements.txt         panda3d==1.10.16
    ├── models/                  immutable data: levels and search states
    │   ├── state.py             BlockState, GameState
    │   ├── level.py             Level + JSON loader/validator
    │   └── catalog.py           ordered list of levels (LevelEntry)
    ├── engine/
    │   └── transitions.py       the game rules: transition(), successors(), is_goal()
    ├── solvers/
    │   ├── common.py            SearchResult + shared move_cost (UCS/A*)
    │   ├── bfs.py               breadth-first search
    │   ├── dfs.py               depth-first search
    │   ├── ucs.py               uniform-cost search
    │   ├── astar.py             A* search
    │   └── README_Solvers.md    team conventions (cost model, API, rules)
    ├── gui/                     Panda3D front end
    │   ├── app.py               BloxorzApp: window, input, game loop, solver buttons
    │   ├── animation.py         Animator: roll / split / slide / merge animations
    │   ├── geometry.py          grid coordinates -> 3D world poses
    │   └── meshes.py            procedural box, switch and bridge models
    └── levels/
        ├── campaign.json        ordered list of level files
        └── test1.json … test34.json, test_all.json   levels (Test 9 and Test All use every mechanic)
```

## 2. Layers and dependencies

```
        main.py
           │
           ▼
   gui/app.py ───────────────► solvers/{bfs,dfs,ucs,astar}.py
   (BloxorzApp)  │                      │            │
        │        │                      ▼            ▼
        │        │              solvers/common.py   engine/transitions.py
        ▼        ▼                                  │
 gui/animation  gui/geometry, gui/meshes            ▼
        │                                     models/{level,state}.py
        └────────────► models/state.py ◄───────────┘
```

The dependency direction is strictly downward, and the solvers never import anything from `gui/`:

- `models/` imports nothing from the rest of the project, except that `catalog.py` calls `level.load_level`.
- `engine/` imports `models/` only.
- `solvers/` import `engine/` and `models/` only.
- `gui/` imports everything above it.

This is what allows a solver to be written against the engine alone and still drive the GUI.

## 3. Module reference

### 3.1 `models/state.py` — search states

| Type | Meaning |
|---|---|
| `BlockState(orientation, row, col)` | Whole block. `S` = standing on `(row,col)`; `H` = lying on `(row,col)` and `(row,col+1)`; `V` = lying on `(row,col)` and `(row+1,col)`. `occupied_cells()` returns its cells. |
| `GameState(block, bridge_mask, cubes, active_cube)` | A complete search state. `bridge_mask` is an int bitmask with one bit per bridge group. `cubes` is `None` for a whole block, or `((r0,c0),(r1,c1))` when split. `active_cube` (0/1) says which cube the arrow keys move. |

Both are `frozen=True, slots=True` dataclasses, so they are hashable. Solvers can use whole `GameState`s as dict keys and set members, which is how repeated states are detected. The same block position with a different `bridge_mask` or `active_cube` is a different state.

### 3.2 `models/level.py` — static level data

- `Level` holds `name`, `rows` (the map), `start`, `goal`, `bridges`, `switches`, `split_switches`. It does not change during play.
- Map tiles: `#` void, `.` floor, `S` start, `G` goal, `F` fragile, `B` bridge, `o` soft switch, `X` heavy switch, `P` split switch.
- Helpers:
  - `tile_at(r,c)` returns `#` for any out-of-range position.
  - `is_supported(r,c,mask)` is true unless the tile is void, or a bridge whose bit in `mask` is 0.
  - `bridge_index(id)` / `bridge_index_at(r,c)` map bridge ids and cells to mask bits.
  - `split_at(r,c)` finds the split-switch spec at a cell.
  - `initial_bridge_mask` is the mask built from `initially_open` flags.
- `BridgeGroup`, `SwitchSpec` and `SplitSwitchSpec` describe the dynamic mechanics (see section 5).
- `load_level(path)` parses a JSON level and validates it heavily:
  - exactly one `S` and one `G`;
  - every `B` belongs to exactly one bridge group;
  - every `o`/`X` has a switch definition;
  - every `P` has a split definition.

### 3.3 `models/catalog.py`

`load_catalog(campaign.json)` returns a tuple of `LevelEntry(path, name)`. It calls `load_level` on every file, which also validates the whole campaign at startup. It rejects duplicate files and unsafe paths.

### 3.4 `engine/transitions.py` — the rules (single source of truth)

| Function | Role |
|---|---|
| `roll_block(block, action)` | The 3 orientations × 4 directions table. Pure geometry, no terrain. |
| `is_legal(level, block, mask)` | Every occupied cell is supported, and a standing block is not on a fragile (`F`) tile. |
| `initial_state(level)` | Standing block on `S`, with the level's initial bridge mask. |
| `transition(level, state, action)` | Applies one action and returns the next `GameState`, or `None` if illegal. |
| `successors(level, state)` | All legal `(action, next_state)` pairs. Adds `SWAP` only when the state is split. |
| `is_goal(level, state)` | True when the block is whole, standing, and on `G`. |
| `step_cost(before, after, action)` | Engine's example cost: 2 if the bridge mask changed, else 1. **Not used** by the solvers; see `algorithm_logic.md`. |

How `transition()` works:

1. `SWAP`: returns the same state with `active_cube` flipped. It is illegal when the block is whole.
2. **Split mode**:
   - The active cube moves one cell, and the move is rejected if it hits the other cube or an unsupported cell.
   - Switches pressed by the move are applied. Cubes can press soft switches only.
   - If the cubes end up adjacent, they merge into an `H` or `V` block.
3. **Whole block**:
   - Roll with `roll_block` and check legality.
   - Apply switches that were freshly pressed. Soft switches trigger on any cell; heavy switches trigger only when standing on them.
   - If the block lands standing on a `P` tile, apply its optional bridge effect and teleport it into two cubes at the configured positions. If those cubes are adjacent, the block merges again.

Switch effects are either `toggle` (flip the bridge bit) or `permanent` (force the bit open or closed).

### 3.5 `solvers/` — search algorithms

- Each solver is a function `solver(level: Level, start: GameState) -> SearchResult | None`.
- `common.SearchResult` carries `path`, `search_time`, `peak_memory`, `expanded_nodes`. The property `solution_length` counts Moves, not `SWAP`s.
- Solvers use only `successors()` and `is_goal()` from the engine, so they inherit all game rules. They start from the state passed in, which may be mid-level, split, or have bridges already toggled.
- `README_Solvers.md` documents the conventions (cost model, return values, mandatory rules).
- Algorithm details are in `algorithm_logic.md`.

### 3.6 `gui/` — presentation

- `app.py` — `BloxorzApp(ShowBase)`:
  - Builds the scene from a `Level`: floor boxes, bridge models, switch markers.
  - Maintains `self.state` and `self.level`.
  - `try_move(action)` calls `engine.transition`, starts an `Animator`, updates status, and handles win.
  - Has New Game / Restart / level selector and four solver buttons.
  - `_run_solver(name, solver)` calls the solver with `(self.level, self.state)`, shows the stats line, re-validates the returned path step by step with `transition`, and then replays it through `try_move(..., from_solver=True)`.
- `animation.py` — `Animator`: chooses the animation phase (roll, split, cube slide, merge) from the `before`/`after` states.
- `geometry.py` — `tile_to_world`, `block_to_world(BlockState) -> BoxPose`.
- `meshes.py` — procedural geometry for the unit box, switch markers and bridge pieces.

## 4. Runtime flow

**Startup** — `main.py`:
1. `load_catalog(levels/campaign.json)` validates and lists levels.
2. `load_level(entries[0].path)` loads the first level.
3. `BloxorzApp(level, entries).run()` starts the Panda3D loop.

**Manual play** — key press → `try_move(action)` → `transition(level, state, action)`. If the result is `None` the move is illegal and shown as such. Otherwise `self.state` is replaced and the animation runs. `is_goal` is checked for the win.

**Solver run** — button → `_run_solver` → `solver(level, state)`.
1. The solver repeatedly calls `successors()` to explore.
2. It returns a `SearchResult` with the action list and the metrics.
3. The GUI shows `Time / Peak / Expanded / Length` and animates the path.

## 5. Level JSON format

```json
{
  "name": "Test 9",
  "map": ["####", "#S.G", "..."],
  "bridges":  [{"id": "b1", "cells": [[r,c]], "initially_open": false}],
  "switches": [{"position": [r,c], "type": "soft|heavy",
                "effect": "toggle|permanent", "bridges": ["b1"], "set_open": true}],
  "split_switches": [{"position": [r,c], "cubes": [[r0,c0],[r1,c1]],
                      "effect": "toggle|permanent", "bridges": ["b1"], "set_open": true}]
}
```

`set_open` is required for `permanent` effects and forbidden for `toggle`. `effect`/`bridges` on a split switch are optional but must be given together. A simple level (like `test1.json`) has only `name` and `map`.

## 6. Notes

- `__pycache__/*.pyc` files are tracked in git; consider adding a `.gitignore`.
