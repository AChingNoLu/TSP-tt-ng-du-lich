"""
utils.py
========
Tiện ích vẽ đồ thị và xuất bảng kết quả cho dự án TSP.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # backend không cần GUI, an toàn khi chạy script
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

# Palette màu nhất quán
PALETTE = {
    "GA-OX":                 "#4C9BE8",
    "GA-PMX":                "#F4A261",
    "GA-CX":                 "#2EC4B6",
    "Greedy (Nearest Neighbor)": "#E63946",
    "Greedy Best-Start":     "#FF6B6B",
    "Simulated Annealing":   "#A8DADC",
    "default":               "#888888",
}


def _get_color(algo_name: str) -> str:
    for key in PALETTE:
        if key in algo_name:
            return PALETTE[key]
    return PALETTE["default"]


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Đường hội tụ GA
# ---------------------------------------------------------------------------

def plot_convergence(ga_results: list, title: str = "Convergence",
                     save_path: str = None):
    """
    Vẽ đồ thị best tour length theo từng thế hệ của một hoặc nhiều lần
    chạy GA (để so sánh crossover).

    Parameters
    ----------
    ga_results : list[dict]
        Danh sách kết quả từ run_ga().
    title : str
    save_path : str, optional
        Nếu cung cấp, lưu file PNG tại đường dẫn này.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    for res in ga_results:
        label = res.get("algorithm", "GA")
        color = _get_color(label)
        ax.plot(res["history"], label=label, color=color, linewidth=2, alpha=0.9)

    ax.set_xlabel("Thế hệ (Generation)", color="white", fontsize=12)
    ax.set_ylabel("Độ dài tour tốt nhất", color="white", fontsize=12)
    ax.set_title(title, color="white", fontsize=14, fontweight="bold")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444")
    ax.legend(facecolor="#0f3460", labelcolor="white", fontsize=10)
    ax.grid(True, color="#2a2a4a", linestyle="--", linewidth=0.7)

    plt.tight_layout()
    if save_path:
        _ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  [saved] {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Vẽ tour trên mặt phẳng 2D
# ---------------------------------------------------------------------------

def plot_tours(instance, results: list, title: str = "Tour Comparison",
               save_path: str = None):
    """
    Vẽ đường đi của nhiều thuật toán trên cùng bản đồ thành phố.

    Parameters
    ----------
    instance : TSPInstance
    results : list[dict]
        Kết quả từ các thuật toán.
    title : str
    save_path : str, optional
    """
    n_algo = len(results)
    cols = min(n_algo, 3)
    rows = (n_algo + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    fig.patch.set_facecolor("#1a1a2e")

    if n_algo == 1:
        axes = [axes]
    elif rows == 1:
        axes = list(axes)
    else:
        axes = [ax for row in axes for ax in row]

    coords = instance.coords

    for ax, res in zip(axes, results):
        ax.set_facecolor("#16213e")
        tour = res["tour"] + [res["tour"][0]]  # khép kín
        xs = coords[tour, 0]
        ys = coords[tour, 1]
        algo = res.get("algorithm", "?")
        color = _get_color(algo)

        ax.plot(xs, ys, "-o", color=color, markersize=5,
                linewidth=1.5, markerfacecolor="white", markeredgewidth=0.5)
        ax.scatter(coords[0, 0], coords[0, 1], color="#FFD700",
                   s=120, zorder=5, label="Start")
        ax.set_title(f"{algo}\nLength: {res['length']:.2f}",
                     color="white", fontsize=10)
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#444")
        ax.legend(facecolor="#0f3460", labelcolor="white", fontsize=8)

    # Ẩn axes thừa
    for ax in axes[n_algo:]:
        ax.set_visible(False)

    fig.suptitle(title, color="white", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    if save_path:
        _ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  [saved] {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Biểu đồ cột so sánh thuật toán
# ---------------------------------------------------------------------------

def plot_comparison_bar(data: dict, metric: str = "length",
                        title: str = "Algorithm Comparison",
                        ylabel: str = "Tour Length",
                        save_path: str = None):
    """
    Biểu đồ cột so sánh các thuật toán.

    Parameters
    ----------
    data : dict[str, float]
        {tên_thuật_toán: giá_trị}
    metric, title, ylabel, save_path : str
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    names = list(data.keys())
    values = list(data.values())
    colors = [_get_color(n) for n in names]

    bars = ax.bar(names, values, color=colors, width=0.5, edgecolor="#444",
                  linewidth=0.8)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.01,
                f"{val:.2f}", ha="center", va="bottom",
                color="white", fontsize=9, fontweight="bold")

    ax.set_title(title, color="white", fontsize=13, fontweight="bold")
    ax.set_ylabel(ylabel, color="white", fontsize=11)
    ax.tick_params(colors="white", axis="both")
    ax.spines[:].set_color("#444")
    ax.grid(True, axis="y", color="#2a2a4a", linestyle="--", linewidth=0.7)
    plt.xticks(rotation=15, ha="right", color="white")
    plt.tight_layout()

    if save_path:
        _ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  [saved] {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Scalability plot
# ---------------------------------------------------------------------------

def plot_scalability(sizes: list, results_by_algo: dict,
                     metric: str = "best_length",
                     title: str = "Scalability Analysis",
                     ylabel: str = "Tour Length",
                     save_path: str = None):
    """
    Vẽ đường biểu diễn metric theo kích thước bài toán cho nhiều thuật toán.

    Parameters
    ----------
    sizes : list[int]
        Danh sách kích thước (N thành phố).
    results_by_algo : dict[str, list[float]]
        {tên_algo: [giá_trị_tại_N10, giá_trị_tại_N30, giá_trị_tại_N50]}
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    for algo, values in results_by_algo.items():
        color = _get_color(algo)
        ax.plot(sizes, values, "-o", label=algo, color=color,
                linewidth=2, markersize=8, markerfacecolor="white")

    ax.set_xlabel("Số thành phố (N)", color="white", fontsize=12)
    ax.set_ylabel(ylabel, color="white", fontsize=12)
    ax.set_title(title, color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444")
    ax.legend(facecolor="#0f3460", labelcolor="white", fontsize=10)
    ax.grid(True, color="#2a2a4a", linestyle="--", linewidth=0.7)
    ax.set_xticks(sizes)

    plt.tight_layout()
    if save_path:
        _ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  [saved] {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Bảng tổng hợp kết quả
# ---------------------------------------------------------------------------

def make_summary_table(records: list) -> pd.DataFrame:
    """
    Tạo DataFrame tổng hợp từ list kết quả.

    Parameters
    ----------
    records : list[dict]
        Mỗi dict cần có: 'algorithm', 'best_length', 'time_sec'.
        Tùy chọn: 'n_cities', 'crossover_type', 'avg_length', 'std_length'.

    Returns
    -------
    pd.DataFrame
    """
    rows = []
    for r in records:
        rows.append({
            "Thuật toán": r.get("algorithm", "?"),
            "N thành phố": r.get("n_cities", "-"),
            "Tour tốt nhất": f"{r.get('best_length', 0):.2f}",
            "Tour TB": f"{r.get('avg_length', r.get('best_length', 0)):.2f}",
            "Std": f"{r.get('std_length', 0):.2f}",
            "Thời gian (s)": f"{r.get('time_sec', 0):.4f}",
        })
    return pd.DataFrame(rows)


def print_table(df: pd.DataFrame, title: str = ""):
    """In bảng ra console với định dạng đẹp."""
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
    print(df.to_string(index=False))
    print()


def save_table_csv(df: pd.DataFrame, path: str):
    """Lưu bảng ra file CSV."""
    _ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  [saved] {path}")
