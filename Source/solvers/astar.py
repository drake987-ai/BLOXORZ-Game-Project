"""
1. CÁC THÀNH PHẦN CÓ SẴN
- Hàm cần hoàn thiện: astar(level: Level, start: GameState) -> SearchResult | None.
- Level (models.level): bản đồ hiện tại; level.goal là tọa độ đích (row, col).
- level.rows chứa bản đồ; bridges/switches/split_switches chứa cơ chế đặc biệt.
- GameState (models.state): trạng thái bất biến, dùng được trong set và dict.
- state.block.orientation = 'S' (đứng), 'H' (ngang), 'V' (dọc).
- state.block.row, state.block.col là tọa độ; bridge_mask là trạng thái cầu.
- state.cubes = None hoặc ((r0,c0), (r1,c1)); active_cube = 0/1.
- state.is_split kiểm tra chế độ tách; occupied_cells() trả các ô đang chiếm.
- SearchAction (engine.transitions): 'UP', 'DOWN', 'LEFT', 'RIGHT', 'SWAP'.
- successors(level, state) -> list[tuple[SearchAction, GameState]]:
  ví dụ: [('LEFT', next_state), ('SWAP', swapped_state)].
  Đã xử lý mọi luật di chuyển, cầu, Fragile, công tắc, split và ghép.
- is_goal(level, state) -> bool: True khi khối nguyên đứng tại ô G.
- transition(level, state, action) -> GameState | None: thử một hành động.
- step_cost(before, after, action) -> int: đổi cầu = 2, còn lại = 1.
- heapq (Python chuẩn) có thể dùng cho hàng đợi ưu tiên.
- Heuristic do người viết chọn; phải tương thích với hàm chi phí đã chọn.
- Ví dụ API: for action, next_state in successors(level, start): ...

2. YÊU CẦU TRẢ VỀ
- SearchResult (solvers.common) có bốn trường:
  path: list[SearchAction] | None; search_time: float (giây);
  peak_memory: int (byte); expanded_nodes: int (số trạng thái mở rộng).
- Có lời giải: SearchResult(path=path, search_time=t, peak_memory=m, expanded_nodes=n).
- path đi từ start đến goal, không được bỏ 'SWAP' nếu cần đổi cube.
- Tìm xong mà không có đường: SearchResult(path=None, ...các số đo thật...).
- start đã ở goal: path=[]; chưa viết thuật toán: return None.
- result.solution_length tự tính Moves, không tính SWAP; không truyền vào hàm tạo.
- Có thể đo thời gian bằng time.perf_counter(), bộ nhớ bằng tracemalloc.

3. QUY ƯỚC BẮT BUỘC
- start là trạng thái lúc người chơi bấm A*, không nhất thiết đầu màn.
- So sánh đủ GameState; vị trí giống nhưng bridge_mask hoặc active_cube khác vẫn khác.
- SWAP không cộng Moves trong GUI; nhóm dự kiến cost SWAP=0, di chuyển=1.
- step_cost() của engine đổi cầu=2; không nhầm với cách tính Moves.
- Phải mô tả heuristic, đánh giá admissible/consistent theo cost đã chọn.
- Split Switch có thể dịch chuyển cube nên khoảng cách thường dễ ước lượng sai.
- Không sửa GUI, engine, level hoặc tên hàm; chỉ trả SearchResult đúng chuẩn.
"""

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
