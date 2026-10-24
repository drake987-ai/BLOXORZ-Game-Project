"""
1. CÁC THÀNH PHẦN CÓ SẴN
- Hàm cần hoàn thiện: ucs(level: Level, start: GameState) -> SearchResult | None.
- Level (models.level): bản đồ hiện tại; level.goal là tọa độ đích (row, col).
- level.rows chứa các hàng bản đồ; bridges/switches/split_switches mô tả ô đặc biệt.
- GameState (models.state): trạng thái bất biến, dùng được làm key dict / phần tử set.
- state.block.orientation = 'S' (đứng), 'H' (ngang), 'V' (dọc).
- state.block.row, state.block.col là tọa độ; state.bridge_mask chứa bit trạng thái cầu.
- state.cubes = None khi chưa split; nếu split là ((r0,c0), (r1,c1)).
- state.active_cube = 0/1; state.is_split cho biết đang tách khối hay không.
- state.occupied_cells(): các ô khối/cube đang chiếm.
- SearchAction (engine.transitions): 'UP', 'DOWN', 'LEFT', 'RIGHT', 'SWAP'.
- successors(level, state) -> list[tuple[SearchAction, GameState]]:
  ví dụ: [('UP', next_state), ('SWAP', swapped_state)].
  Đã loại nước đi sai, xử lý cầu, switch, fragile, split và merge.
- is_goal(level, state) -> bool: True khi khối nguyên đứng tại ô G.
- transition(level, state, action) -> GameState | None: thử một hành động.
- step_cost(before, after, action) -> int: HÀM CÓ SẴN; đổi cầu = 2, còn lại = 1.
- heapq là thư viện Python chuẩn cho hàng đợi ưu tiên, có thể import khi viết.
- Ví dụ API: for action, next_state in successors(level, start): ...

2. YÊU CẦU TRẢ VỀ
- SearchResult (solvers.common) có bốn trường:
  path: list[SearchAction] | None; search_time: float (giây);
  peak_memory: int (byte); expanded_nodes: int (số trạng thái mở rộng).
- Có lời giải: SearchResult(path=path, search_time=t, peak_memory=m, expanded_nodes=n).
- path đi từ start đến goal, gồm cả 'SWAP' nếu lời giải dùng đổi cube.
- Tìm xong mà không có đường: SearchResult(path=None, ...các số đo thật...).
- start đã ở goal: path=[]; chưa viết thuật toán: return None.
- result.solution_length tự tính Moves (bỏ SWAP); không truyền vào constructor.
- Có thể đo thời gian bằng time.perf_counter(), bộ nhớ bằng tracemalloc.

3. QUY ƯỚC BẮT BUỘC
- start có thể ở giữa màn, đang split hoặc cầu đã đổi trạng thái.
- Phân biệt toàn bộ GameState; cùng vị trí nhưng cầu khác là trạng thái khác.
- GUI không tính SWAP là Move; nhóm dự kiến chi phí di chuyển=1, SWAP=0.
- step_cost() của engine đang dùng quy tắc KHÁC, không tự động là Moves.
- Chốt và ghi rõ hàm chi phí trước khi làm; đề khuyến khích chi phí không đều.
- UCS phải xét khả năng đường mới đến cùng trạng thái có chi phí thấp hơn.
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
