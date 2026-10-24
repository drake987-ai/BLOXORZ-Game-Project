"""Vẽ biểu đồ và xuất bảng tổng hợp từ experiments/summary_4_metrics.csv.

Dùng cho mục "Present results with: Tables / Charts" trong phần Experiments của report.

Cài thư viện (một lần):
    python -m pip install pandas matplotlib seaborn

Chạy (từ thư mục gốc của repo):
    python experiments/plot_results.py
    python experiments/plot_results.py --csv experiments/summary_4_metrics.csv --out experiments/figures --dpi 300

Kết quả:
    experiments/figures/fig1a_time_by_group.png     thời gian trung bình theo nhóm màn chơi
    experiments/figures/fig1b_memory_by_group.png   bộ nhớ đỉnh trung bình theo nhóm
    experiments/figures/fig1c_expanded_by_group.png số node mở rộng trung bình theo nhóm
    experiments/figures/fig1d_moves_by_group.png    độ dài lời giải trung bình theo nhóm
    experiments/figures/fig2_expanded_per_level.png số node mở rộng ở từng màn (thang log)
    experiments/figures/fig3_moves_distribution.png phân bố độ dài lời giải (boxplot)
    experiments/figures/fig4_time_vs_expanded.png   quan hệ node mở rộng ↔ thời gian (small multiples)
    experiments/figures/fig5_expanded_ratio_heatmap.png  node mở rộng so với BFS (heatmap)
    experiments/figures/fig6_risk_tradeoff.png      BFS vs UCS/A*: số bước và chi phí rủi ro
    experiments/tables/*.csv, *.md                  bảng tổng hợp để dán vào report
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend không cần màn hình: chỉ ghi file ảnh
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

EXP_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# 1. Quy ước chung: thứ tự, màu, tên cột
# ---------------------------------------------------------------------------
ALGOS = ["BFS", "DFS", "UCS", "A*"]
# Mỗi thuật toán luôn giữ MỘT màu cố định ở mọi biểu đồ (màu theo thực thể, không theo thứ hạng).
# Bộ 4 màu này đã kiểm tra phân biệt được với người mù màu (protan/deutan/tritan).
COLORS = {"BFS": "#2a78d6", "DFS": "#eb6834", "UCS": "#1baf7a", "A*": "#eda100"}
MARKERS = {"BFS": "o", "DFS": "s", "UCS": "^", "A*": "D"}  # mã hóa phụ, không chỉ dựa vào màu
GROUPS = ["Group 1: Basic", "Group 2: Fragile", "Group 3: Bridges & Switches", "Group 4: Combined"]
GROUP_SHORT = {
    "Group 1: Basic": "G1 Basic",
    "Group 2: Fragile": "G2 Fragile",
    "Group 3: Bridges & Switches": "G3 Bridges & Switches",
    "Group 4: Combined": "G4 Combined",
}
# 4 chỉ số đề bài yêu cầu: (tên cột trong CSV, nhãn trục)
METRICS = [
    ("Search_Time_ms", "Search time (ms)"),
    ("Peak_Memory_KB", "Peak memory (KB)"),
    ("Expanded_Nodes", "Expanded nodes"),
    ("Solution_Moves", "Solution length (moves)"),
]
FRAGILE_PENALTY = 4  # khớp solvers/common.py: mỗi bước kết thúc trên ô F bị cộng thêm 4

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"


def setup_style() -> None:
    """Cấu hình giao diện chung cho mọi biểu đồ (gọi một lần)."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "legend.frameon": False,
        "font.size": 10,
    })


# ---------------------------------------------------------------------------
# 2. Đọc và chuẩn bị dữ liệu
# ---------------------------------------------------------------------------
def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Số thứ tự màn ("Test 12" -> 12) để sắp xếp đúng thứ tự số, không theo chữ.
    df["Level_No"] = df["Level"].str.extract(r"(\d+)").astype(int)
    # Biến Algorithm/Group thành kiểu "category" có thứ tự -> seaborn vẽ đúng thứ tự mong muốn.
    df["Algorithm"] = pd.Categorical(df["Algorithm"], categories=ALGOS, ordered=True)
    df["Group"] = pd.Categorical(df["Group"], categories=GROUPS, ordered=True)
    # Chi phí rủi ro mà UCS/A* tối thiểu hóa (move_cost): 1/bước + 4 nếu kết thúc trên ô fragile.
    # SWAP có chi phí 0 nên không cộng.
    df["Risk_Cost"] = df["Solution_Moves"] + FRAGILE_PENALTY * df["Fragile_Touches"]
    return df.sort_values(["Group", "Level_No", "Algorithm"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Bảng tổng hợp (Tables)
# ---------------------------------------------------------------------------
def to_markdown(table: pd.DataFrame, floatfmt: str = "{:,.2f}") -> str:
    """Đổi DataFrame thành bảng Markdown (không cần cài thêm thư viện tabulate)."""
    table = table.reset_index()
    header = "| " + " | ".join(str(c) for c in table.columns) + " |"
    sep = "|" + "|".join("---" for _ in table.columns) + "|"
    rows = []
    for _, row in table.iterrows():
        cells = []
        for v in row:
            if isinstance(v, (float, np.floating)):
                cells.append(floatfmt.format(v))
            elif isinstance(v, (int, np.integer)):
                cells.append(f"{v:,}")
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *rows]) + "\n"


def save_table(table: pd.DataFrame, out_dir: Path, name: str) -> None:
    table.to_csv(out_dir / f"{name}.csv")
    (out_dir / f"{name}.md").write_text(to_markdown(table), encoding="utf-8")


def make_tables(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = [m for m, _ in METRICS]

    # Bảng 1: trung bình mỗi thuật toán trên toàn bộ 34 màn.
    overall = df.groupby("Algorithm", observed=True).agg(
        Levels=("Level", "nunique"),
        Time_ms_mean=("Search_Time_ms", "mean"),
        Memory_KB_mean=("Peak_Memory_KB", "mean"),
        Expanded_mean=("Expanded_Nodes", "mean"),
        Moves_mean=("Solution_Moves", "mean"),
        Risk_Cost_mean=("Risk_Cost", "mean"),
        Moves_vs_BFS=("Suboptimality_Ratio", "mean"),
    )
    save_table(overall, out_dir, "table1_overall")

    # Bảng 2: trung bình theo (nhóm, thuật toán) cho 4 chỉ số.
    by_group = df.groupby(["Group", "Algorithm"], observed=True)[cols].mean()
    save_table(by_group, out_dir, "table2_by_group")

    # Bảng 3: chi tiết từng màn, mỗi chỉ số một bảng dạng "màn x thuật toán" (pivot).
    for col, _ in METRICS:
        wide = df.pivot_table(index=["Level_No", "Level", "Group"], columns="Algorithm",
                              values=col, aggfunc="first", observed=True).sort_index(level="Level_No")
        wide = wide.droplevel("Level_No")
        save_table(wide, out_dir, f"table3_per_level_{col}")


# ---------------------------------------------------------------------------
# 4. Biểu đồ (Charts)
# ---------------------------------------------------------------------------
def finish(fig: plt.Figure, path: Path, dpi: int) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# Tên file riêng cho từng chỉ số của Hình 1 (a, b, c, d theo thứ tự METRICS).
FIG1_FILES = {
    "Search_Time_ms": "fig1a_time_by_group.png",
    "Peak_Memory_KB": "fig1b_memory_by_group.png",
    "Expanded_Nodes": "fig1c_expanded_by_group.png",
    "Solution_Moves": "fig1d_moves_by_group.png",
}


def fig1_group_means(df: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    """Hình 1a-1d: mỗi chỉ số một hình riêng, cột = trung bình theo nhóm màn, màu = thuật toán."""
    for col, label in METRICS:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        sns.barplot(data=df, x="Group", y=col, hue="Algorithm", hue_order=ALGOS,
                    palette=COLORS, errorbar=None, ax=ax,
                    edgecolor=SURFACE, linewidth=1.5)  # viền màu nền = khe hở giữa các cột
        for container in ax.containers:  # ghi giá trị lên đầu mỗi cột
            ax.bar_label(container, fmt="%.0f", fontsize=8, color=INK_2, padding=2)
        ax.set_title(f"Mean {label} by level group")
        ax.set_xlabel("")
        ax.set_ylabel(label)
        ax.set_xticks(range(len(GROUPS)), [GROUP_SHORT[g] for g in GROUPS])
        ax.margins(y=0.12)  # chừa chỗ cho nhãn số trên cột cao nhất
        ax.grid(axis="x", visible=False)
        ax.legend(ncol=4, loc="upper left", bbox_to_anchor=(0, -0.1))  # chú thích dưới trục
        sns.despine(ax=ax, left=True)
        fig.tight_layout()
        finish(fig, out_dir / FIG1_FILES[col], dpi)


def _level_axis(df: pd.DataFrame) -> tuple[list[str], list[tuple[str, int, int]]]:
    """Thứ tự màn trên trục x (theo nhóm rồi theo số) và vị trí bắt đầu/kết thúc của mỗi nhóm."""
    order = df.drop_duplicates("Level")[["Level", "Group"]]
    levels = order["Level"].tolist()
    spans = []
    for g in GROUPS:
        idx = [i for i, grp in enumerate(order["Group"]) if grp == g]
        if idx:
            spans.append((g, idx[0], idx[-1]))
    return levels, spans


def fig2_expanded_per_level(df: pd.DataFrame, path: Path, dpi: int) -> None:
    """Số node mở rộng ở từng màn, trục y thang log (giá trị chênh nhau nhiều lần)."""
    levels, spans = _level_axis(df)
    x_of = {lv: i for i, lv in enumerate(levels)}
    fig, ax = plt.subplots(figsize=(14, 5))
    for k, (g, a, b) in enumerate(spans):  # tô nền xen kẽ để tách các nhóm
        if k % 2 == 0:
            ax.axvspan(a - 0.5, b + 0.5, color="#f0efec", zorder=0)
        ax.text((a + b) / 2, 1.01, GROUP_SHORT[g], transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", color=INK_2, fontsize=9)
    for algo in ALGOS:
        sub = df[df["Algorithm"] == algo]
        ax.plot(sub["Level"].map(x_of), sub["Expanded_Nodes"], color=COLORS[algo],
                marker=MARKERS[algo], markersize=5, linewidth=2, label=algo, zorder=3)
    ax.set_yscale("log")
    ax.set_xticks(range(len(levels)), [lv.replace("Test ", "T") for lv in levels], fontsize=8)
    ax.set_xlim(-0.5, len(levels) - 0.5)
    ax.set_xlabel("Level (sorted by group)")
    ax.set_ylabel("Expanded nodes (log scale)")
    ax.set_title("Expanded nodes per level", pad=22)
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=4, loc="upper left", bbox_to_anchor=(0, -0.13))  # đặt dưới trục, không che dữ liệu
    sns.despine(ax=ax, left=True)
    fig.tight_layout()
    finish(fig, path, dpi)


def fig3_moves_distribution(df: pd.DataFrame, path: Path, dpi: int) -> None:
    """Phân bố độ dài lời giải: boxplot (tứ phân vị) + từng điểm (stripplot)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="Algorithm", y="Solution_Moves", hue="Algorithm", order=ALGOS,
                palette=COLORS, width=0.5, fliersize=0, legend=False, dodge=False, ax=ax,
                boxprops={"alpha": 0.35}, medianprops={"color": INK, "linewidth": 2})
    sns.stripplot(data=df, x="Algorithm", y="Solution_Moves", hue="Algorithm", order=ALGOS,
                  palette=COLORS, size=5, jitter=0.18, legend=False, dodge=False, ax=ax,
                  edgecolor=SURFACE, linewidth=0.6)
    medians = df.groupby("Algorithm", observed=True)["Solution_Moves"].median()
    for i, algo in enumerate(ALGOS):
        ax.text(i + 0.28, medians[algo], f"median {medians[algo]:.0f}", va="center",
                fontsize=8, color=INK_2)
    ax.set_xlabel("")
    ax.set_ylabel("Solution length (moves)")
    ax.set_title("Solution length across 34 levels")
    sns.despine(ax=ax, left=True)
    fig.tight_layout()
    finish(fig, path, dpi)


def fig4_time_vs_expanded(df: pd.DataFrame, path: Path, dpi: int) -> None:
    """Small multiples: mỗi ô tô màu một thuật toán, các thuật toán khác làm nền xám."""
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), sharex=True, sharey=True)
    for ax, algo in zip(axes, ALGOS):
        ax.scatter(df["Expanded_Nodes"], df["Search_Time_ms"], s=14, color="#c9c8c2", zorder=1)
        sub = df[df["Algorithm"] == algo]
        ax.scatter(sub["Expanded_Nodes"], sub["Search_Time_ms"], s=36, color=COLORS[algo],
                   marker=MARKERS[algo], edgecolor=SURFACE, linewidth=0.8, zorder=2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(algo)
        ax.set_xlabel("Expanded nodes")
        sns.despine(ax=ax)
    axes[0].set_ylabel("Search time (ms)")
    fig.suptitle("Search time grows with expanded nodes (log-log)", y=1.02,
                 fontweight="bold", color=INK)
    fig.tight_layout()
    finish(fig, path, dpi)


def fig5_expanded_ratio_heatmap(df: pd.DataFrame, path: Path, dpi: int) -> None:
    """Heatmap: Expanded / Expanded của BFS ở cùng màn. <1 (xanh) = ít node hơn BFS, >1 (đỏ) = nhiều hơn."""
    wide = df.pivot_table(index=["Level_No", "Level"], columns="Algorithm",
                          values="Expanded_Nodes", observed=True).sort_index()
    wide = wide.droplevel("Level_No")
    ratio = wide.div(wide["BFS"], axis=0)
    log_ratio = np.log2(ratio)  # dùng log2 để 0.5x và 2x cách đều điểm giữa 1x
    cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
    lim = float(np.abs(log_ratio).max().max())
    fig, ax = plt.subplots(figsize=(6.5, 11))
    sns.heatmap(log_ratio, cmap=cmap, norm=TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim),
                annot=ratio, fmt=".2f", annot_kws={"fontsize": 7, "color": INK},
                linewidths=1, linecolor=SURFACE, cbar_kws={"label": "log2(expanded / BFS expanded)"},
                ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Expanded nodes relative to BFS\n(cell value = ratio; blue = fewer, red = more)")
    fig.tight_layout()
    finish(fig, path, dpi)


def fig6_risk_tradeoff(df: pd.DataFrame, path: Path, dpi: int) -> None:
    """Những màn mà BFS và UCS chọn đường khác nhau: so số bước (Moves) và chi phí rủi ro."""
    wide = df.pivot_table(index=["Level_No", "Level"], columns="Algorithm",
                          values="Risk_Cost", observed=True)
    changed = wide.index[wide["UCS"] < wide["BFS"]].get_level_values("Level")
    sub = df[df["Level"].isin(changed) & df["Algorithm"].isin(["BFS", "UCS", "A*"])].copy()
    sub["Algorithm"] = sub["Algorithm"].cat.remove_unused_categories()
    order = sub.drop_duplicates("Level").sort_values("Level_No")["Level"].tolist()
    algos = ["BFS", "UCS", "A*"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=False)
    for ax, (col, label) in zip(axes, [("Solution_Moves", "Solution length (moves)"),
                                      ("Risk_Cost", "Risk-weighted cost (move_cost)")]):
        sns.barplot(data=sub, x="Level", y=col, hue="Algorithm", order=order, hue_order=algos,
                    palette=COLORS, errorbar=None, edgecolor=SURFACE, linewidth=1.5, ax=ax)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.0f", fontsize=8, color=INK_2, padding=2)
        ax.set_title(label)
        ax.set_xlabel("")
        ax.set_ylabel(label)
        ax.grid(axis="x", visible=False)
        ax.get_legend().remove()
        sns.despine(ax=ax, left=True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("UCS/A* trade a few extra moves for a lower-risk path", y=1.10,
                 fontweight="bold", color=INK)
    fig.tight_layout()
    finish(fig, path, dpi)


# ---------------------------------------------------------------------------
# 5. Chương trình chính
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path, default=EXP_DIR / "summary_4_metrics.csv")
    parser.add_argument("--out", type=Path, default=EXP_DIR / "figures")
    parser.add_argument("--tables", type=Path, default=EXP_DIR / "tables")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    setup_style()
    df = load_data(args.csv)
    print(f"Loaded {len(df)} rows: {df['Level'].nunique()} levels x {df['Algorithm'].nunique()} algorithms")

    make_tables(df, args.tables)
    print(f"  tables written to {args.tables}")

    args.out.mkdir(parents=True, exist_ok=True)
    fig1_group_means(df, args.out, args.dpi)
    fig2_expanded_per_level(df, args.out / "fig2_expanded_per_level.png", args.dpi)
    fig3_moves_distribution(df, args.out / "fig3_moves_distribution.png", args.dpi)
    fig4_time_vs_expanded(df, args.out / "fig4_time_vs_expanded.png", args.dpi)
    fig5_expanded_ratio_heatmap(df, args.out / "fig5_expanded_ratio_heatmap.png", args.dpi)
    fig6_risk_tradeoff(df, args.out / "fig6_risk_tradeoff.png", args.dpi)


if __name__ == "__main__":
    main()
