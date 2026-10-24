"""
1. CÁC THÀNH PHẦN CÓ SẴN
- Hàm cần viết: bfs(level: Level, start: GameState) -> SearchResult | None.
- Level (models.level): dữ liệu màn chơi; level.goal là ô đích (row, col).
- level.rows là bản đồ; level.bridges, switches, split_switches là ô đặc biệt.
- GameState (models.state): trạng thái bất biến, có thể làm key dict/phần tử set.
- state.block.orientation: 'S' (đứng), 'H' (ngang), 'V' (dọc).
- state.block.row, state.block.col: vị trí khối nguyên.
- state.bridge_mask: trạng thái đóng/mở của tất cả nhóm cầu.
- state.cubes: None nếu khối nguyên, hoặc ((r0, c0), (r1, c1)) nếu tách.
- state.active_cube: 0 hoặc 1; state.is_split: có đang tách khối không.
- state.occupied_cells(): những ô mà khối hoặc hai cube đang chiếm.
- SearchAction (engine.transitions): 'UP', 'DOWN', 'LEFT', 'RIGHT', 'SWAP'.
- successors(level, state) -> list[tuple[SearchAction, GameState]]:
  ví dụ [('RIGHT', next_state), ('SWAP', swapped_state)].
  Chỉ trả bước hợp lệ, đã xử lý sàn, fragile, bridge, switch, split/merge.
- is_goal(level, state) -> bool: True nếu khối nguyên đứng trên đích.
- transition(level, state, action) -> GameState | None: áp dụng một hành động.
- initial_state(level): trạng thái đầu màn; KHÔNG thay thế tham số start.
- deque (collections): hàng đợi hai đầu; hỗ trợ append() và popleft().
- Ví dụ API: for action, next_state in successors(level, start): ...

2. YÊU CẦU TRẢ VỀ
- SearchResult ở solvers.common, có 4 trường:
  path: list[SearchAction] | None; search_time: float (giây);
  peak_memory: int (byte); expanded_nodes: int (số trạng thái mở rộng).
- Tìm được đường: SearchResult(path=path, search_time=t,
  peak_memory=m, expanded_nodes=n).
- path là danh sách hành động theo thứ tự từ start tới goal; giữ SWAP nếu cần.
- Không tìm được đường: SearchResult(path=None, ...các chỉ số đo thực tế...).
- start đã thắng: path=[]; thuật toán chưa triển khai: return None.
- result.solution_length tự tính số Moves, không tính SWAP.
- Có thể dùng time.perf_counter() đo thời gian và tracemalloc đo bộ nhớ.
- GUI tự kiểm tra path, hiện thông số và phát animation; không cần sửa app.py.

3. QUY ƯỚC BẮT BUỘC
- start là trạng thái lúc nhấn BFS, không nhất thiết là đầu màn.
- Phải hỗ trợ trạng thái đang split và cầu đã mở/đóng.
- So sánh toàn bộ GameState, không chỉ tọa độ; tránh trạng thái lặp.
- Dùng successors() thay vì viết lại luật di chuyển/công tắc/cầu.
- BFS thông thường tối ưu số hành động, bao gồm cả SWAP.
- Solution Length theo GUI là số Moves, không tính SWAP.
- Không đổi chữ ký hàm, không sửa Engine, GUI, Animation hoặc level.
- Không chỉ print lời giải; phải trả SearchResult đúng định dạng.
"""

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
