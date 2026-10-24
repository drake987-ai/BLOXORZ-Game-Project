import csv
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
if (EXP_DIR.parent / "models").exists():
    SOURCE_DIR = EXP_DIR.parent
elif (EXP_DIR.parent / "Source" / "models").exists():
    SOURCE_DIR = EXP_DIR.parent / "Source"
else:
    raise RuntimeError("Hãy đặt thư mục 'experiments' bên trong 'Source' hoặc ngang hàng 'Source'!")

if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from models.catalog import load_catalog
from models.level import load_level
from engine.transitions import initial_state, transition, step_cost
from solvers.bfs import bfs
from solvers.dfs import dfs
from solvers.ucs import ucs
from solvers.astar import astar


def analyze_path_details(level, start_state, path):
    """Phân tích sâu đường đi: SWAP, đổi cầu, giẫm gạch mỏng."""
    if path is None:
        return 0, 0, 0, 0
    state = start_state
    bridge_toggles = 0
    fragile_touches = 0
    engine_cost = 0
    for action in path:
        next_state = transition(level, state, action)
        if next_state is None:
            break
        if next_state.bridge_mask != state.bridge_mask:
            bridge_toggles += 1
        engine_cost += step_cost(state, next_state, action)
        if any(level.tile_at(r, c) == "F" for r, c in next_state.occupied_cells()):
            fragile_touches += 1
        state = next_state
    swap_count = sum(1 for a in path if a == "SWAP")
    return swap_count, bridge_toggles, fragile_touches, engine_cost


def classify_level(level) -> str:
    """Phân loại độ khó đồng bộ tuyệt đối với generator của nhóm."""
    has_fragile = any("F" in row for row in level.rows)
    has_switch = len(level.switches) > 0
    has_split = len(level.split_switches) > 0
    if not has_fragile and not has_switch and not has_split:
        return "Group 1: Basic"
    if has_fragile and not has_switch and not has_split:
        return "Group 2: Fragile"
    if not has_fragile and (has_switch or has_split):
        return "Group 3: Bridges & Switches"
    return "Group 4: Combined"


def main():
    entries = load_catalog(SOURCE_DIR / "levels" / "campaign.json")
    algorithms = [("BFS", bfs), ("DFS", dfs), ("UCS", ucs), ("A*", astar)]

    summary_rows = []
    paradox_logs = []

    print("=" * 95)
    print(f"{'LEVEL':<10} | {'GROUP':<10} | {'ALGO':<4} | {'TIME(ms)':>9} | {'EXPANDED':>9} | {'MOVES':>6} | {'SWAP':>4} | {'PRUNING':>8}")
    print("=" * 95)

    for entry in entries:
        # Bỏ qua map test_all tổng hợp nếu quá nặng khi chạy tự động hàng loạt
        if entry.path.name == "test_all.json":
            print(f"[*] Bỏ qua {entry.name} để tránh nghẽn bộ nhớ.")
            continue

        level = load_level(entry.path)
        start = initial_state(level)
        group_name = classify_level(level)
        level_results = {}

        for algo_name, solver_fn in algorithms:
            res = solver_fn(level, start)
            if res is None or res.path is None:
                continue

            moves = res.solution_length
            total_actions = len(res.path)
            swaps, toggles, fragiles, eng_cost = analyze_path_details(level, start, res.path)
            time_ms = res.search_time * 1000.0
            mem_kb = res.peak_memory / 1024.0

            level_results[algo_name] = {
                "res": res,
                "moves": moves,
                "total": total_actions,
                "swaps": swaps,
                "toggles": toggles,
                "fragiles": fragiles,
                "eng_cost": eng_cost,
                "time_ms": time_ms,
                "mem_kb": mem_kb,
                "path": res.path
            }

        # Tính toán các chỉ số AI chuyên sâu so sánh trực tiếp với Baseline (BFS)
        bfs_data = level_results.get("BFS")
        bfs_exp = bfs_data["res"].expanded_nodes if bfs_data else 0
        bfs_moves = bfs_data["moves"] if bfs_data else 1

        for algo_name, data in level_results.items():
            res = data["res"]
            # 1. Tỷ lệ cắt tỉa của Heuristic so với BFS
            pruning_pct = ((bfs_exp - res.expanded_nodes) / bfs_exp * 100.0) if bfs_exp > 0 else 0.0
            # 2. Độ lãng phí đường đi (Suboptimality Ratio)
            subopt_ratio = (data["moves"] / bfs_moves) if bfs_moves > 0 else 1.0
            # 3. Hao tổn mở rộng trên mỗi bước đi tìm được
            work_per_move = (res.expanded_nodes / data["moves"]) if data["moves"] > 0 else 0.0

            pruning_str = f"{pruning_pct:+.1f}%" if algo_name == "A*" else "-"
            print(
                f"{entry.name:<10} | {group_name[:10]:<10} | {algo_name:<4} | {data['time_ms']:9.2f} | "
                f"{res.expanded_nodes:9d} | {data['moves']:6d} | {data['swaps']:4d} | {pruning_str:>8}"
            )

            summary_rows.append({
                "Level": entry.name,
                "File": entry.path.name,
                "Group": group_name,
                "Algorithm": algo_name,
                "Search_Time_ms": round(data["time_ms"], 3),
                "Peak_Memory_KB": round(data["mem_kb"], 2),
                "Expanded_Nodes": res.expanded_nodes,
                "Solution_Moves": data["moves"],
                "SWAP_Count": data["swaps"],
                "Total_Actions": data["total"],
                "Suboptimality_Ratio": round(subopt_ratio, 2),
                "Pruning_Efficiency_Pct": round(pruning_pct, 2) if algo_name == "A*" else 0.0,
                "Work_Per_Move": round(work_per_move, 2),
                "Bridge_Toggles": data["toggles"],
                "Fragile_Touches": data["fragiles"],
                "Engine_Step_Cost": data["eng_cost"]
            })
        print("-" * 95)

        # Lưu vết các màn có nghịch lý đường đi giữa BFS, UCS và A*
        if "BFS" in level_results and "UCS" in level_results and "A*" in level_results:
            b, u, a = level_results["BFS"], level_results["UCS"], level_results["A*"]
            if b["path"] != u["path"] or u["path"] != a["path"] or b["swaps"] > 0:
                paradox_logs.append(
                    f"=== {entry.name} ({group_name}) ===\n"
                    f"  * BFS : Moves={b['moves']}, SWAP={b['swaps']}, Expanded={b['res'].expanded_nodes}\n"
                    f"  * UCS : Moves={u['moves']}, SWAP={u['swaps']}, Expanded={u['res'].expanded_nodes}\n"
                    f"  * A*  : Moves={a['moves']}, SWAP={a['swaps']}, Expanded={a['res'].expanded_nodes}\n"
                    f"  * Lộ trình BFS vs UCS: {'GIỐNG NHAU' if b['path'] == u['path'] else 'RẼ NHÁNH KHÁC BIỆT'}\n"
                    f"  * Lộ trình UCS vs A* : {'GIỐNG NHAU' if u['path'] == a['path'] else 'RẼ NHÁNH DO HEURISTIC TIE-BREAKING'}\n"
                )

    # Xuất ra file CSV hoàn chỉnh
    csv_file = EXP_DIR / "summary_4_metrics.csv"
    if summary_rows:
        with csv_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\n[XONG] Bảng dữ liệu chuẩn đã lưu tại: experiments/{csv_file.name}")

    txt_file = EXP_DIR / "cost_and_path_analysis.txt"
    with txt_file.open("w", encoding="utf-8") as f:
        f.write("PHÂN TÍCH CHUYÊN SÂU: NGHỊCH LÝ ĐƯỜNG ĐI VÀ ĐÁNH GIÁ HEURISTIC TRÊN TẬP TEST MỞ RỘNG\n")
        f.write("=" * 80 + "\n\n")
        f.write("\n".join(paradox_logs))
    print(f"[XONG] Phân tích nghịch lý đã lưu tại: experiments/{txt_file.name}")


if __name__ == "__main__":
    main()