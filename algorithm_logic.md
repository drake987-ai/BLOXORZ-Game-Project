# Search Algorithms for Bloxorz

Implementations: `Source/solvers/bfs.py`, `dfs.py`, `ucs.py`, `astar.py`. Each keeps the stub signature `solver(level, start) -> SearchResult` and plugs into the GUI buttons unchanged. The engine, GUI and levels are not modified.

## 1. Problem formalisation

### 1.1 State representation

A search state is a `GameState` (`models/state.py`):

| Field | Meaning |
|---|---|
| `block = (orientation, row, col)` | `S` standing at `(r,c)`; `H` lying on `(r,c),(r,c+1)`; `V` lying on `(r,c),(r+1,c)` |
| `bridge_mask` | Integer bitmask, bit *i* = bridge group *i* is open |
| `cubes` | `None` for a whole block, else the two cube positions `((r0,c0),(r1,c1))` |
| `active_cube` | Which cube the next move applies to (0/1) |

- Orientation is part of the state, because it decides which cells are occupied and which moves are legal.
- `bridge_mask` is part of the state. The same block position with a different mask is a different state.
- In split mode, `(cubes, active_cube)` replaces the block position. Which cube is active matters because it decides which cube the next move moves.
- The state is a frozen dataclass, so it is hashable. Every solver compares **whole** `GameState`s, never just coordinates.
- Repeated-state handling is therefore uniform: a dict or set keyed by `GameState`.

The static level (map, goal, switch definitions) lives in `Level` and is not part of the state.

### 1.2 Initial state and goal test

- Initial state: whatever state the GUI passes in as `start`. It may be mid-level, split, or have bridges already toggled. `initial_state(level)` is the level's opening state, not a replacement for `start`.
- Goal test: `is_goal(level, s)`. The block must be whole, standing (`S`), on the goal cell `G`. A lying block on `G` does not win, and a single cube cannot win.

### 1.3 Actions and transition model

Actions are `UP, DOWN, LEFT, RIGHT` and, only when split, `SWAP`.

The solvers never re-implement rules. They call `successors(level, state)` from `engine/transitions.py`, which returns the legal `(action, next_state)` pairs. The transition is deterministic and covers:

1. **Orientation table** (`roll_block`): 12 cases, e.g. `S` + `RIGHT` → `H` at `(r, c+1)`, and `H` + `RIGHT` → `S` at `(r, c+2)`.
2. **Legality** (`is_legal`): every occupied cell is supported (not void/out of range, not a closed bridge), and a standing block is not on a fragile tile.
3. **Switches**: soft switches fire if any occupied cell is on them, heavy switches only when standing on them. A switch fires on a fresh press only, and the effect is `toggle` (flip the bridge bit) or `permanent` (set it open or closed).
4. **Split switch**: a block that arrives standing on `P` is teleported into two cubes at the configured cells, and it may also change bridges. Cubes then move one at a time and can press soft switches only. `SWAP` flips `active_cube`.
5. **Merge**: when the two cubes become adjacent they fuse into an `H`/`V` block (if that position is legal).

An illegal move simply does not appear in `successors`.

### 1.4 Cost function

UCS and A* share one risk-aware cost, `move_cost(level, before, after, action)` in `solvers/common.py`. Full justification: `ucs_costs.md`.

| Transition | Cost |
|---|---|
| `SWAP` | 0 |
| Any roll or cube step | 1 |
| Any occupied cell ends on a fragile tile | +4 |

The cost is deliberately **not** the GUI's Moves counter, so UCS/A* may return a path with more Moves than BFS if it avoids fragile tiles. `SearchResult.solution_length` still reports Moves. The engine's `step_cost()` is still unused.

Consequences:

- Costs are non-uniform and BFS no longer optimises the same objective: BFS minimises *actions*, UCS/A* minimise *risk-weighted cost*.
- All costs are >= 0, so closing states on pop is safe. `SWAP = 0` cannot loop: the closed set expands each `GameState` once.
- Every non-SWAP move costs >= 1, which is what the A* heuristic relies on (section 3).

## 2. Algorithms

All four share a few conventions:

- `parents[state] = (previous_state, action)` allows the path to be rebuilt in reverse.
- `time.perf_counter()` measures search time. `tracemalloc` records peak memory. A trace that is already running is reused (peak reset), otherwise one is started and stopped.
- `expanded_nodes` counts states whose successors were generated.
- A start that is already a goal returns `path=[]`. An exhausted search returns `path=None` with real metrics.

### 2.1 BFS — `bfs.py`

- Queue: FIFO `deque`. `parents` doubles as the visited set.
- A state is marked when generated, so each state is enqueued once.
- The goal test is done at generation, which is safe for BFS because the first generated goal is at minimum depth.
- **Optimal in the number of actions** (SWAP counts as a step), complete, O(b^d) time and memory.
- It is **not** guaranteed to minimise Moves, as `README_Solvers.md` explains: 10 Moves + 3 SWAP (13 actions) loses to 11 Moves (11 actions).

```python
import time
import tracemalloc
from collections import deque

from engine.transitions import SearchAction, successors, is_goal
from models.level import Level
from models.state import GameState
from solvers.common import SearchResult


def _rebuild(parents: dict, state: GameState) -> list[SearchAction]:
    """Walk parent links from `state` back to the start and reverse them."""
    actions: list[SearchAction] = []
    while parents[state] is not None:
        state, action = parents[state]
        actions.append(action)
    actions.reverse()
    return actions


def bfs(level: Level, start: GameState) -> SearchResult | None:
    """Breadth-first search; optimal in the number of actions (SWAP included)."""
    own_trace = not tracemalloc.is_tracing()
    if own_trace:
        tracemalloc.start()
    else:
        tracemalloc.reset_peak()
    t0 = time.perf_counter()
    expanded = 0
    path: list[SearchAction] | None = None
    try:
        if is_goal(level, start):
            path = []
        else:
            # `parents` doubles as the visited set; states are marked when generated.
            parents: dict[GameState, tuple[GameState, SearchAction] | None] = {start: None}
            frontier = deque([start])
            while frontier and path is None:
                state = frontier.popleft()
                expanded += 1
                for action, nxt in successors(level, state):
                    if nxt in parents:
                        continue
                    parents[nxt] = (state, action)
                    # FIFO order means the first time the goal is generated is optimal.
                    if is_goal(level, nxt):
                        path = _rebuild(parents, nxt)
                        break
                    frontier.append(nxt)
        elapsed = time.perf_counter() - t0
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        if own_trace:
            tracemalloc.stop()
    return SearchResult(path=path, search_time=elapsed, peak_memory=peak, expanded_nodes=expanded)
```

### 2.2 DFS — `dfs.py`

- Explicit stack, iterative, so no recursion limit on long paths.
- A state is closed when **popped**, and the parent link recorded at that moment is the route DFS actually took.
- Successors are pushed in reverse so the engine's order (`UP, DOWN, LEFT, RIGHT, SWAP`) is explored first.
- The visited set stops cycles. Because the state space is finite, DFS is complete here.
- It finds some solution quickly with low frontier memory, but the path is far from optimal (Test All: 1363 Moves vs 84 optimal).

```python
import time
import tracemalloc

from engine.transitions import SearchAction, is_goal, successors
from models.level import Level
from models.state import GameState
from solvers.common import SearchResult


def _rebuild(parents: dict, state: GameState) -> list[SearchAction]:
    """Walk parent links from `state` back to the start and reverse them."""
    actions: list[SearchAction] = []
    while parents[state] is not None:
        state, action = parents[state]
        actions.append(action)
    actions.reverse()
    return actions


def dfs(level: Level, start: GameState) -> SearchResult | None:
    """Iterative depth-first search with a visited set; finds a path, not the shortest."""
    own_trace = not tracemalloc.is_tracing()
    if own_trace:
        tracemalloc.start()
    else:
        tracemalloc.reset_peak()
    t0 = time.perf_counter()
    expanded = 0
    path: list[SearchAction] | None = None
    try:
        # Stack entries carry their parent link; a state is closed when popped, so the
        # parent that is recorded is the one DFS actually reached it through first.
        parents: dict[GameState, tuple[GameState, SearchAction] | None] = {}
        stack: list[tuple[GameState, tuple[GameState, SearchAction] | None]] = [(start, None)]
        while stack:
            state, link = stack.pop()
            if state in parents:
                continue
            parents[state] = link
            if is_goal(level, state):
                path = _rebuild(parents, state)
                break
            expanded += 1
            # Reversed so the first action in successors() order is explored first.
            for action, nxt in reversed(successors(level, state)):
                if nxt not in parents:
                    stack.append((nxt, (state, action)))
        elapsed = time.perf_counter() - t0
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        if own_trace:
            tracemalloc.stop()
    return SearchResult(path=path, search_time=elapsed, peak_memory=peak, expanded_nodes=expanded)
```

### 2.3 UCS — `ucs.py`

- Priority queue (`heapq`) keyed on `(g, number_of_actions, counter)`, where `g` is the accumulated `move_cost`. The action count gives a deterministic preference for fewer actions among equal-cost paths, and the counter avoids comparing `GameState` objects.
- `best[state]` stores the cheapest known `g`. A new route is pushed only if it is strictly cheaper, which covers "a later path reaches the same state at lower cost".
- Stale heap entries are skipped through the `closed` set. This is lazy deletion: the heap has no decrease-key operation.
- The goal test is done on **pop**, which is required for UCS optimality when costs are non-uniform.
- Complete and optimal in `move_cost`, because all costs are ≥ 0. Not necessarily optimal in Moves.

```python
import heapq
import itertools
import time
import tracemalloc

from engine.transitions import SearchAction, is_goal, successors
from models.level import Level
from models.state import GameState
from solvers.common import SearchResult, move_cost


def ucs(level: Level, start: GameState) -> SearchResult | None:
    """Uniform-cost search minimising the risk-aware cost of solvers.common.move_cost."""
    own_trace = not tracemalloc.is_tracing()
    if own_trace:
        tracemalloc.start()
    else:
        tracemalloc.reset_peak()
    t0 = time.perf_counter()
    expanded = 0
    path: list[SearchAction] | None = None
    try:
        best: dict[GameState, int] = {start: 0}
        parents: dict[GameState, tuple[GameState, SearchAction] | None] = {start: None}
        closed: set[GameState] = set()
        tie = itertools.count()
        # Heap key: (cost, number of actions, insertion order). The action count breaks
        # ties between equal-cost paths in favour of fewer actions.
        heap: list[tuple[int, int, int, GameState]] = [(0, 0, next(tie), start)]
        while heap:
            g, depth, _, state = heapq.heappop(heap)
            if state in closed:
                continue          # stale entry: a cheaper route was already expanded
            if is_goal(level, state):
                path = []
                node = state
                while parents[node] is not None:
                    node, action = parents[node]
                    path.append(action)
                path.reverse()
                break
            closed.add(state)
            expanded += 1
            for action, nxt in successors(level, state):
                if nxt in closed:
                    continue
                new_g = g + move_cost(level, state, nxt, action)
                # Record a strictly cheaper route (closed states are final: costs are never negative).
                if nxt not in best or new_g < best[nxt]:
                    best[nxt] = new_g
                    parents[nxt] = (state, action)
                    heapq.heappush(heap, (new_g, depth + 1, next(tie), nxt))
        elapsed = time.perf_counter() - t0
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        if own_trace:
            tracemalloc.stop()
    return SearchResult(path=path, search_time=elapsed, peak_memory=peak, expanded_nodes=expanded)
```

### 2.4 A* — `astar.py`

Same skeleton as UCS with priority `f = g + h`, tie-broken on smaller `h`. The cost `g` is the same `move_cost`. `closed`-set pruning (no re-opening) is valid because the heuristic is consistent (see 3.3).

```python
import heapq
import itertools
import time
import tracemalloc

from engine.transitions import SearchAction, is_goal, successors
from models.level import Level
from models.state import GameState
from solvers.common import SearchResult, move_cost

# Positions are doubled ("half units") so block centres stay integers:
#   S at (r, c) -> (2r, 2c)    H at (r, c) -> (2r, 2c + 1)    V at (r, c) -> (2r + 1, 2c)
#   two cubes   -> (r0 + r1, c0 + c1)   (twice the midpoint; equals the block centre once merged)
Point = tuple[int, int]


def _centre(state: GameState) -> Point:
    if state.cubes is not None:
        (r0, c0), (r1, c1) = state.cubes
        return r0 + r1, c0 + c1
    b = state.block
    if b.orientation == "S":
        return 2 * b.row, 2 * b.col
    if b.orientation == "H":
        return 2 * b.row, 2 * b.col + 1
    return 2 * b.row + 1, 2 * b.col


def _lower_bound(a: Point, b: Point) -> int:
    """Fewest rolls to shift the centre from a to b, ignoring all terrain.

    A roll changes one coordinate only, by at most 3 half units (1.5 cells: standing
    <-> lying), and a cube roll by 1 half unit. Every non-SWAP move costs at least 1 under
    solvers.common.move_cost (penalties only add), so ceil(manhattan / 3) never overestimates.
    """
    return -(-(abs(a[0] - b[0]) + abs(a[1] - b[1])) // 3)


def _make_heuristic(level: Level):
    """Build h(state) for this level.

    h(p) = min( direct(p -> goal),  min over split switches q of direct(p -> q) + T[q] ),
    where T[q] is the same quantity measured from the cubes' midpoint after q fires.
    A split switch teleports the block, so ignoring it would break admissibility.
    T is solved as a small fixed point (Bellman-Ford over the split switches).
    """
    goal: Point = (2 * level.goal[0], 2 * level.goal[1])
    portals = [((2 * s.position[0], 2 * s.position[1]),
                (s.cubes[0][0] + s.cubes[1][0], s.cubes[0][1] + s.cubes[1][1]))
               for s in level.split_switches]

    tail = [_lower_bound(exit_, goal) for _, exit_ in portals]
    for _ in range(len(portals)):
        changed = False
        for i, (_, exit_) in enumerate(portals):
            for j, (entry, _) in enumerate(portals):
                via = _lower_bound(exit_, entry) + tail[j]
                if via < tail[i]:
                    tail[i] = via
                    changed = True
        if not changed:
            break

    def h(state: GameState) -> int:
        p = _centre(state)
        best = _lower_bound(p, goal)
        for (entry, _), t in zip(portals, tail):
            best = min(best, _lower_bound(p, entry) + t)
        return best

    return h


def astar(level: Level, start: GameState) -> SearchResult | None:
    """A* minimising the risk-aware cost of solvers.common.move_cost (same as UCS)."""
    own_trace = not tracemalloc.is_tracing()
    if own_trace:
        tracemalloc.start()
    else:
        tracemalloc.reset_peak()
    t0 = time.perf_counter()
    expanded = 0
    path: list[SearchAction] | None = None
    try:
        h = _make_heuristic(level)
        best: dict[GameState, int] = {start: 0}
        parents: dict[GameState, tuple[GameState, SearchAction] | None] = {start: None}
        closed: set[GameState] = set()
        tie = itertools.count()
        # Heap key: (f, h, insertion order); preferring small h dives toward the goal on ties.
        h0 = h(start)
        heap: list[tuple[int, int, int, GameState]] = [(h0, h0, next(tie), start)]
        while heap:
            _, _, _, state = heapq.heappop(heap)
            if state in closed:
                continue
            if is_goal(level, state):
                path = []
                node = state
                while parents[node] is not None:
                    node, action = parents[node]
                    path.append(action)
                path.reverse()
                break
            closed.add(state)
            expanded += 1
            g = best[state]
            for action, nxt in successors(level, state):
                if nxt in closed:
                    continue
                new_g = g + move_cost(level, state, nxt, action)
                if nxt not in best or new_g < best[nxt]:
                    best[nxt] = new_g
                    parents[nxt] = (state, action)
                    hn = h(nxt)
                    heapq.heappush(heap, (new_g + hn, hn, next(tie), nxt))
        elapsed = time.perf_counter() - t0
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        if own_trace:
            tracemalloc.stop()
    return SearchResult(path=path, search_time=elapsed, peak_memory=peak, expanded_nodes=expanded)
```

## 3. A* heuristic design

### 3.1 Idea: bound on the block's centre

Ignore terrain, fragile tiles, bridges and orientation legality, and ask only how many rolls it takes to move the block's centre to the goal's centre.

Use **doubled coordinates** so centres stay integers:

| Shape | Centre (doubled) |
|---|---|
| `S` at `(r,c)` | `(2r, 2c)` |
| `H` at `(r,c)` | `(2r, 2c+1)` |
| `V` at `(r,c)` | `(2r+1, 2c)` |
| two cubes | `(r0+r1, c0+c1)` (= doubled midpoint) |

The goal is a standing block on `G`, so its centre is `(2·gr, 2·gc)`.

Facts about one move, in doubled units (the fragile penalty only adds to the base cost of 1 per roll, so the per-move shift bound still holds):

- A roll changes only one coordinate.
- Standing ↔ lying shifts the centre by 3 (1.5 cells). Lying → lying shifts by 2.
- A cube roll shifts the cubes' midpoint by 1 (0.5 cells).
- `SWAP` changes nothing.

So every move (cost >= 1) shifts the centre by at most 3 in Manhattan distance, and a free `SWAP` shifts it by 0. This gives

```
direct(p, q) = ceil( (|Δrow| + |Δcol|) / 3 )        (doubled-coordinate distance)
```

This matches the course hint "Manhattan distance divided by 2", but it is tighter (1.5 cells per roll, not 2) and it is orientation-aware without a separate penalty: a lying block's centre is at a half-integer offset, so the bound is at least 1 for it.

### 3.2 Split switches (why a naive distance is wrong)

A split switch teleports the block, so `direct(block, goal)` could overestimate. Using `direct(p, goal)` alone would be inadmissible.

The code handles this exactly:

- For each split switch *q* with entry point `E_q` (the `P` tile) and exit point `X_q` (midpoint of its destination cubes), compute `T[q]`, the best bound from the exit to the goal. It may use other switches too, so `T` is solved by Bellman-Ford style relaxation over the (few) switches.
- The heuristic is

```
h(p) = min( direct(p, goal),  min over q of [ direct(p, E_q) + T[q] ] )
```

The teleport is treated as free in the relaxation, which is why it can only be a lower bound.

Cubes are handled by the same formula: moving the midpoint by 1 per move and then, once merged, by up to 3 per move gives total ≥ `ceil(D/3)` (proof: `k + (D−k)/3 ≥ D/3` for any number k ≥ 0 of pre-merge moves).

### 3.3 Admissibility and consistency (with respect to `move_cost`)

- **Admissible.** Any real path to the goal either never fires a split switch or fires one at some point. In the first case, each move changes the centre by ≤ 3, so at least `direct(p, goal)` moves are needed. In the second case, the moves up to the teleport number at least `direct(p, E_q)`, and the rest at least `T[q]`, by the same argument applied to the exit. So `h(p)` is a lower bound on the number of non-SWAP moves, hence on the true `move_cost` (each costs ≥ 1), and `SWAP` (cost 0) never invalidates it because it doesn't move the centre.
- **Consistent.** For a move of cost *c* from *s* to *s′*:
  - Plain rolls: `direct` obeys the triangle inequality, since `ceil((a+b)/3) ≤ ceil(a/3) + ceil(b/3)`, and a roll shifts the centre by ≤ 3, so `h(s) ≤ 1 + h(s′) ≤ c + h(s′)` for any real cost `c ≥ 1`, including moves that land on fragile tiles.
  - `SWAP`: the centre is unchanged, so `h(s) = h(s′)` and the cost is 0.
  - Teleport: `h(s)` is at most `direct(p, E_q) + T[q] ≤ 1 + h(s′)`, because `T[q]` is exactly the heuristic value at the exit.
- Consistency lets A* close each state on first expansion and stay optimal, and it makes A* return the same cost as UCS.

Verification (ad-hoc scripts, not part of the repo): it was checked that `h(s) ≤ c + h(s′)` on every edge sampled, and that with the risk-aware cost: 0 inconsistent edges out of 54,668 explored across all 35 levels, h never exceeded the true optimal cost on 210 random mid-level starts, and A* and UCS returned equal cost on every level and sampled start.

### 3.4 Limits

- The bound ignores terrain, so it is loose in mazes. A* expands fewer nodes than UCS but is far from perfect on Test 5–8.
- A tighter, still admissible alternative is a relaxed "point on the grid" BFS distance computed once per level. This is not implemented.

## 4. Results (default levels, start state)

Measured on this machine with the code above. `Moves` is the GUI counter (SWAP excluded); `cost` is `move_cost` (section 1.4), which is what UCS and A* minimise. All paths were replayed through `transition()` and ended in `is_goal`.

| Level | BFS Moves / cost (exp.) | DFS Moves / cost (exp.) | UCS Moves / cost (exp.) | A* Moves / cost (exp.) |
|---|---|---|---|---|
| Test 1 | 14 / 14 (105) | 78 / 78 (87) | 14 / 14 (122) | 14 / 14 (19) |
| Test 2 | 18 / 22 (36) | 33 / 37 (38) | 18 / 22 (111) | 18 / 22 (42) |
| Test 3 | 23 / 23 (112) | 72 / 72 (97) | 23 / 23 (122) | 23 / 23 (40) |
| Test 4 | 22 / 22 (308) | 156 / 156 (191) | 22 / 22 (312) | 22 / 22 (112) |
| Test 5 | 22 / 22 (219) | 94 / 94 (157) | 22 / 22 (228) | 22 / 22 (130) |
| Test 6 | 25 / 29 (233) | 82 / 86 (103) | 25 / 29 (241) | 25 / 29 (162) |
| Test 7 | 27 / 27 (217) | 121 / 121 (172) | 27 / 27 (234) | 27 / 27 (178) |
| Test 8 | 25 / 25 (338) | 95 / 95 (190) | 25 / 25 (360) | 25 / 25 (138) |
| Test 9 | 21 / 25 (834) | 75 / 79 (90) | 21 / 25 (1068) | 21 / 25 (333) |
| Test All | 84 / 88 (5838) | 1363 / 1367 (2519) | 84 / 88 (6398) | 84 / 88 (2610) |

Observations:

- UCS and A* always return the same cost; BFS (fewest actions) and DFS do not minimise it. Across all 35 campaign levels the UCS path differs from BFS on only 6 and its cost is strictly lower on 5 (Tests 15, 16, 18, 26, 27), all of which have fragile tiles on BFS's shortest route. On Tests 15, 18 and 26 UCS takes more Moves (19 vs 18, 22 vs 20, 24 vs 21) to avoid them. On the other 29 levels the shortest path is already the cheapest, so UCS and BFS tie.
- Because the heuristic prices every move at 1 and only fragile moves cost more, it stays close to the true cost on most levels, and A* expands the fewest nodes of the optimal algorithms.
- DFS is cheap but returns long, wasteful paths.
