# Hướng dẫn viết Solver

## 1. Quy ước tính số bước (Moves)

Trong game Bloxorz gốc, thao tác **SWAP** (nhấn `Space` để chuyển quyền điều khiển giữa hai khối sau khi tách) **không được tính là một Move**.

- `UP`, `DOWN`, `LEFT`, `RIGHT`: mỗi thao tác hợp lệ tính **1 Move**.
- `SWAP`: tính **0 Move**, nhưng vẫn là **một Action** trong đường đi.

Vì vậy, các thuật toán có mục tiêu tối ưu khác nhau:

- **BFS:** tối ưu **tổng số Action** (mỗi Action, kể cả `SWAP`, được tính là một bước tìm kiếm). Do đó, BFS **không đảm bảo tối ưu số Moves hiển thị**.
- **UCS và A\*:** tối ưu **chi phí rủi ro** `move_cost` trong `solvers/common.py` (xem `ucs_costs.md`): `SWAP` = 0, mỗi bước lật = 1, +4 nếu kết thúc trên ô fragile. Số Moves hiển thị không còn là mục tiêu tối ưu.
- **DFS:** tìm một lời giải hợp lệ, không bắt buộc tối ưu.

Ví dụ: đường A có 10 Moves + 3 SWAP (13 Actions), đường B có 11 Moves + 0 SWAP (11 Actions). BFS có thể chọn B vì ít Actions hơn. UCS/A\* chọn đường có `move_cost` thấp nhất, có thể là đường dài hơn nhưng tránh ô fragile.

**Lưu ý:** `SearchResult.solution_length` đếm **Moves**, không đếm `SWAP`. Khi viết UCS/A\*, không mặc định dùng `step_cost()` nếu cách tính chi phí của hàm đó khác quy ước `SWAP = 0`, di chuyển = `1` nêu trên.

## 2. Cách tích hợp thuật toán

Chỉ cần hoàn thiện hàm tương ứng trong file của mình:

- `bfs.py` → `bfs(level, start)`
- `dfs.py` → `dfs(level, start)`
- `ucs.py` → `ucs(level, start)`
- `astar.py` → `astar(level, start)`

Giữ nguyên tên hàm, tham số và kiểu trả về theo khung có sẵn. **Không cần sửa GUI hoặc Engine**: chỉ cần viết thuật toán và trả về `SearchResult` đúng yêu cầu thì có thể mở game, nhấn nút Solver tương ứng để chạy và xem kết quả/animation.

`SearchResult` cần có:

- `path`: danh sách các Action dẫn tới đích; `None` nếu không tìm được lời giải.
- `search_time`: thời gian tìm kiếm (giây).
- `peak_memory`: bộ nhớ đỉnh (byte).
- `expanded_nodes`: số trạng thái đã mở rộng.

`return None` chỉ là trạng thái **chưa triển khai thuật toán**, không phải kết quả “không tìm thấy lời giải”. Xem thêm note trong từng file Solver để dùng đúng các hàm có sẵn.

## 3. Kiểm tra sau khi viết

**Nên test trước ở màn cuối (`Test All`)**, vì màn này có hết cơ chế
