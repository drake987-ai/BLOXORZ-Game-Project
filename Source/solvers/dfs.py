"""
1. CÁC THÀNH PHẦN CÓ SẴN
- Hàm cần hoàn thiện: dfs(level: Level, start: GameState) -> SearchResult | None.
- Level (models.level): bản đồ hiện tại; level.goal là tọa độ đích (row, col).
- level.rows chứa các hàng bản đồ; level.bridges/switches/split_switches mô tả ô đặc biệt.
- GameState (models.state): một trạng thái bất biến, dùng được làm key dict / phần tử set.
- state.block.orientation = 'S' (đứng), 'H' (nằm ngang), 'V' (nằm dọc).
- state.block.row, state.block.col là tọa độ khối; state.bridge_mask là trạng thái các cầu.
- state.cubes = None khi chưa tách; nếu tách là ((r0,c0), (r1,c1)).
- state.active_cube = 0 hoặc 1: cube đang điều khiển; state.is_split là bool.
- state.occupied_cells(): các ô hiện khối hoặc hai cube đang chiếm.
- SearchAction (engine.transitions): 'UP', 'DOWN', 'LEFT', 'RIGHT', 'SWAP'.
- successors(level, state) -> list[tuple[SearchAction, GameState]]:
  ví dụ: [('RIGHT', next_state), ('SWAP', swapped_state)].
  Chỉ gồm nước hợp lệ; tự xử lý ô trống, Fragile, cầu, switch, split/merge.
- is_goal(level, state) -> bool: True khi khối nguyên đứng trên đích.
- transition(level, state, action) -> GameState | None: thử một hành động.
- initial_state(level) tạo trạng thái đầu màn; không dùng để thay thế start.
- Ví dụ API: for action, next_state in successors(level, start): ...

2. YÊU CẦU TRẢ VỀ
- SearchResult nằm trong solvers.common và gồm bốn trường:
  path: list[SearchAction] | None; search_time: float (giây);
  peak_memory: int (byte); expanded_nodes: int (số trạng thái mở rộng).
- Có lời giải: SearchResult(path=path, search_time=t, peak_memory=m, expanded_nodes=n).
- path phải theo đúng thứ tự đi từ start đến goal; giữ cả SWAP nếu cần.
- Tìm xong nhưng không có đường: SearchResult(path=None, ...các số đo thật...).
- start đã ở goal: path=[]; hàm chưa được viết: return None.
- result.solution_length tự đếm số Move khác SWAP, không truyền vào constructor.
- Có thể đo thời gian bằng time.perf_counter(), bộ nhớ bằng tracemalloc.

3. QUY ƯỚC BẮT BUỘC
- start là trạng thái đang chơi, có thể đã di chuyển, mở cầu hoặc đang split.
- So sánh toàn bộ GameState, không chỉ (row, col); tránh duyệt lặp vô hạn.
- Không tự viết lại luật chơi hay gọi GUI; chỉ lấy nước từ successors().
- Không sửa tên hàm/chữ ký, models, engine, GUI, animation hoặc level JSON.
- Không print thay cho return; GUI sẽ xác minh path rồi phát animation.
- DFS không bắt buộc tìm đường ngắn nhất; kiểm tra cả màn có split (Test 9/10).
"""

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
