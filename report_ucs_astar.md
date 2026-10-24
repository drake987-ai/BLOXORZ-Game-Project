# Bloxorz Solver — Algorithm Description: UCS and A*

*Simulated report section (covers §3.4, §3.5 and the "Algorithm Description" part of §3.6 of the assignment, for UCS and A\* only). All numbers were measured on the submitted code, from each level's start state.*

---

## 1. Problem formalisation

Both algorithms solve the same search problem. Neither re-implements game rules: they call `successors()` and `is_goal()` from `engine/transitions.py`.

### 1.1 State representation

A state is an immutable, hashable `GameState`:

| Field | Meaning |
|---|---|
| `block = (orientation, row, col)` | `S` standing on `(r,c)`; `H` lying on `(r,c),(r,c+1)`; `V` lying on `(r,c),(r+1,c)` |
| `bridge_mask` | Integer bitmask, bit *i* = bridge group *i* is open |
| `cubes` | `None` for a whole block, else the two cube positions after a split |
| `active_cube` | Which cube the next move applies to (0 or 1) |

- Orientation is part of the state, because it decides which cells are occupied and which moves are legal.
- `bridge_mask` is part of the state: the same block placement with a different bridge configuration is a **different** state, as §3.1 requires.
- Split mode replaces the block position with `(cubes, active_cube)`.
- The static level (map, goal, switch definitions) is not part of the state.

### 1.2 Initial state and goal test

- **Initial state:** whatever state the GUI passes as `start`. It may be mid-level, split, or have bridges already toggled.
- **Goal test:** `is_goal(level, s)` is true only when the block is whole, **standing**, on the goal cell. A lying block on the goal does not win and a single cube cannot win (§2.4, §2.5).

### 1.3 Actions and transition model

Actions are `UP`, `DOWN`, `LEFT`, `RIGHT` and, only while split, `SWAP`. The transition model is deterministic (the game is fully observable, §1):

1. **Orientation table:** 12 cases, e.g. `S` + `RIGHT` → `H` at `(r, c+1)`; `H` + `RIGHT` → `S` at `(r, c+2)`.
2. **Legality:** every occupied cell must be supported (not void, not off the board, not a closed bridge) and a standing block must not be on a fragile tile.
3. **Switches:** soft switches fire if any occupied cell is on them, heavy switches only when standing, split switches only when the block arrives standing. An effect is `toggle` or `permanent`.
4. **Split and merge:** a block landing on a split switch becomes two cubes. Cubes move one at a time, can press soft switches only, and merge into an `H`/`V` block when adjacent. `SWAP` flips `active_cube`.

Illegal moves never appear in `successors()`.

### 1.4 Avoiding revisited states

Both solvers keep a dictionary `best[state]` (cheapest known cost) and a `closed` set, both keyed by the **whole** `GameState`. A state is expanded at most once. This is also why the zero-cost `SWAP` cannot cause an infinite loop: `SWAP` followed by `SWAP` returns to a state that is already closed.

---

## 2. Cost function (UCS and A\*)

`move_cost(level, before, after, action)` in `solvers/common.py`:

| Transition | Cost |
|---|---|
| `SWAP` | 0 |
| Any roll or cube step | 1 |
| Any part of the block ends on a fragile tile | **+4** |

### 2.1 Justification

- **Why not uniform:** with every move at cost 1, UCS reduces to BFS (§3.4). We therefore make a risky move expensive.
- **Fragile penalty:** §3.4 itself suggests that "rolling onto a fragile tile is expensive because it is risky". A fragile tile breaks if the block ends up standing on it, so a path that lies on fragile tiles is one wrong orientation from failure. The penalty of 4 makes the solver accept a detour of up to three extra rolls to avoid one fragile landing. The value is a modelling choice, not derived from the game: it is large enough to change the optimal path when a detour exists, and small enough that unavoidable fragile tiles are still crossed.
- **`SWAP` is free:** it changes who is controlled, not where anything is. This also matches the GUI, which does not count `SWAP` as a Move.
- **Not the GUI Moves counter:** UCS and A\* minimise `move_cost`, not Moves. `SearchResult.solution_length` still reports Moves.
- **Not the engine's `step_cost()`:** that function (2 if the bridge mask changes, else 1) is only used for reporting in the experiments.

### 2.2 Consequences

- Costs are non-negative, so closing a state on pop is safe and the first goal popped is optimal.
- Every non-`SWAP` move costs at least 1. The A\* heuristic relies on this (section 4).
- BFS minimises the **number of actions**, UCS minimises `move_cost`. They coincide when the shortest path crosses no fragile tile.

---

## 3. Uniform-Cost Search

### 3.1 Algorithm

UCS expands states in order of increasing path cost `g`.

1. Push `(0, 0, tie, start)` on a `heapq` priority queue. The key is `(g, number_of_actions, counter)`: the action count prefers fewer actions among equal-cost paths, and the counter avoids comparing `GameState` objects.
2. Pop the entry with the smallest key. If its state is already in `closed`, discard it (a stale entry: the heap has no decrease-key operation, so cheaper routes are pushed as new entries).
3. **Goal test on pop**, not on generation. With non-uniform costs a goal first *generated* by an expensive route could be beaten by a cheaper route found later, so only the goal on pop is guaranteed optimal.
4. Mark the state closed, count it as expanded, and for each `(action, next)` from `successors()` compute `new_g = g + move_cost(...)`. If `next` is not closed and either unseen or `new_g < best[next]`, record `best[next]`, `parents[next] = (state, action)` and push it.
5. When the goal is popped, rebuild the action list from `parents`.
6. If the heap empties, return `path=None` with the real measurements.

### 3.2 Properties

- **Complete:** the state space is finite and every state is expanded at most once.
- **Optimal in `move_cost`:** all step costs are ≥ 0.
- **Time and memory:** worst case O(N log N) over the N reachable states; memory holds `best`, `parents`, `closed` and the heap.
- **Not optimal in Moves.** See section 5.

### 3.3 Why UCS differs from BFS here

| | BFS | UCS |
|---|---|---|
| Minimises | number of actions | `move_cost` |
| Frontier | FIFO queue | priority queue |
| Goal test | on generation | on pop |
| Fragile tiles | ignored | avoided if a detour is cheaper |

---

## 4. A* Search

### 4.1 Algorithm

A\* has the same skeleton and the same cost `g` as UCS, but orders the queue by `f = g + h`. The heap key is `(f, h, counter)`, so among equal `f` the state closer to the goal (smaller `h`) is preferred. Closed-set pruning with no re-opening is valid because the heuristic is consistent (section 4.3).

### 4.2 Heuristic design

**Idea.** Ignore terrain, fragile tiles, bridges and orientation legality. Ask only how many rolls it takes to move the block's centre to the goal's centre.

Positions are doubled ("half-units") so that centres are integers:

| Shape | Centre (doubled) |
|---|---|
| `S` at `(r,c)` | `(2r, 2c)` |
| `H` at `(r,c)` | `(2r, 2c+1)` |
| `V` at `(r,c)` | `(2r+1, 2c)` |
| two cubes | `(r0+r1, c0+c1)` |

The goal is a standing block, so its centre is `(2·gr, 2·gc)`.

**Facts about one move**, in doubled units:

- A roll changes only one coordinate.
- Standing ↔ lying shifts the centre by **3** (1.5 cells); lying → lying shifts it by 2.
- A cube step shifts the cubes' midpoint by 1.
- `SWAP` shifts nothing.

Hence

```
direct(p, q) = ceil( (|Δrow| + |Δcol|) / 3 )
```

This is the assignment's "Manhattan distance divided by 2" hint, tightened to 1.5 cells per roll, and it is orientation-aware without a separate penalty: a lying block's centre sits at a half-integer offset, so its bound is at least 1.

**Split switches.** A split switch teleports the block, so `direct(p, goal)` alone could overestimate. For each split switch *q* with entry point `E_q` (its `P` tile) and exit point `X_q` (the midpoint of its destination cubes) the code computes `T[q]`, the best bound from the exit to the goal (possibly through other switches, solved by Bellman-Ford style relaxation). The heuristic is

```
h(p) = min( direct(p, goal),  min over q of [ direct(p, E_q) + T[q] ] )
```

The teleport is treated as free, which can only make `h` smaller.

### 4.3 Admissibility and consistency

The cost function matters here, because §3.5 requires admissibility **with respect to the chosen cost**.

**Admissible.** Every non-`SWAP` move costs at least 1 (`move_cost` only adds the fragile penalty on top of 1) and shifts the centre by at most 3. A path that never fires a split switch therefore needs at least `direct(p, goal)` moves, each costing ≥ 1. A path that fires switch *q* needs at least `direct(p, E_q)` moves before the teleport and at least `T[q]` after it, by the same argument applied from the exit. `SWAP` costs 0 but does not move the centre, so it cannot reduce the number of moves needed. Therefore `h(p)` never exceeds the true remaining `move_cost`.

**Consistent.** For an edge `s → s'` with cost `c`:

- *Plain roll or cube step:* `direct` obeys the triangle inequality, since `ceil((a+b)/3) ≤ ceil(a/3) + ceil(b/3)`, and a move shifts the centre by ≤ 3, so `h(s) ≤ 1 + h(s')`. Because `c ≥ 1`, this gives `h(s) ≤ c + h(s')`; the fragile penalty only enlarges `c`.
- *`SWAP`:* the centre is unchanged, so `h(s) = h(s')` and `c = 0`.
- *Teleport:* `h(s)` is at most `direct(p, E_q) + T[q] ≤ 1 + h(s')`, because `T[q]` is exactly the heuristic value at the exit.

**Empirical check.** On the submitted code, over all 35 campaign levels: 0 inconsistent edges out of 54,668 explored; `h` never exceeded the true optimal cost on 210 random mid-level start states; A\* and UCS returned the same cost on every level and every sampled start.

### 4.4 Limits

- The bound ignores terrain, so it is loose in maze-like levels, and it prices every move at 1, so it underestimates whenever a fragile tile is involved.
- A tighter admissible alternative is a relaxed point-on-grid BFS distance computed once per level. It was not implemented.

---

## 5. Results

Start state of each level, Moves / cost (expanded nodes). `cost` is `move_cost`; `Moves` is the GUI counter.

| Level | BFS | UCS | A\* |
|---|---|---|---|
| Test 1 | 14 / 14 (105) | 14 / 14 (122) | 14 / 14 (19) |
| Test 2 | 18 / 22 (36) | 18 / 22 (111) | 18 / 22 (42) |
| Test 3 | 23 / 23 (112) | 23 / 23 (122) | 23 / 23 (40) |
| Test 4 | 22 / 22 (308) | 22 / 22 (312) | 22 / 22 (112) |
| Test 5 | 22 / 22 (219) | 22 / 22 (228) | 22 / 22 (130) |
| Test 6 | 25 / 29 (233) | 25 / 29 (241) | 25 / 29 (162) |
| Test 7 | 27 / 27 (217) | 27 / 27 (234) | 27 / 27 (178) |
| Test 8 | 25 / 25 (338) | 25 / 25 (360) | 25 / 25 (138) |
| Test 9 | 21 / 25 (834) | 21 / 25 (1068) | 21 / 25 (333) |
| Test All | 84 / 88 (5838) | 84 / 88 (6398) | 84 / 88 (2610) |

Levels where the fragile penalty changes the answer (BFS vs UCS, Moves / cost):

| Level | BFS | UCS |
|---|---|---|
| Test 15 | 18 / 22 | 19 / 19 |
| Test 16 | 22 / 34 | 22 / 30 |
| Test 18 | 20 / 28 | 22 / 26 |
| Test 26 | 21 / 25 | 24 / 24 |
| Test 27 | 26 / 38 | 26 / 26 |

### 5.1 Discussion

- **UCS and A\* always agree on cost**, as both are optimal for `move_cost`.
- **UCS vs BFS:** over all 35 levels the UCS path differs from BFS on 6, and its cost is strictly lower on 5 (Tests 15, 16, 18, 26, 27). On Tests 15, 18 and 26 it takes 1 to 3 more Moves to avoid fragile tiles. On the other 29 levels the shortest path is already the cheapest, so the two tie. This is inherent: where no hazard lies on the route, any hazard-based cost agrees with BFS.
- **UCS vs BFS expansions:** UCS expands slightly more than BFS (for example 1068 against 834 on Test 9). BFS stops when a goal is generated, UCS only when it is popped, and UCS maintains a priority queue.
- **A\* vs UCS:** A\* expands far fewer nodes, with the largest savings where the straight-line bound is close to the real cost (Test 1: 19 against 122; Test 9: 333 against 1068; Test All: 2610 against 6398). The saving is smallest in mazes (Test 6: 162 against 241).
- **Trade-off:** UCS and A\* are not Moves-optimal. In exchange they return the least risky path. Where Moves matter more than risk, BFS remains the reference.

### 5.2 Strengths and weaknesses

| | Strengths | Weaknesses |
|---|---|---|
| UCS | Optimal for any non-negative cost; needs no heuristic | Expands many states; indistinguishable from BFS on hazard-free routes |
| A\* | Optimal and consistent; many fewer expansions than UCS | Heuristic ignores terrain and prices every move at 1, so it weakens in mazes and around fragile tiles |
