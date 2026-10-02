"""
greedy.py
=========
Thuật toán Greedy (Nearest Neighbor Heuristic) cho bài toán TSP.
Luôn di chuyển đến thành phố chưa thăm gần nhất.
"""

import time
import numpy as np
from .tsp_core import TSPInstance


def greedy_tsp(instance: TSPInstance, start: int = 0) -> dict:
    """
    Giải TSP bằng thuật toán Nearest Neighbor Greedy.

    Bắt đầu từ thành phố `start`, mỗi bước chọn thành phố chưa thăm
    gần nhất cho đến khi thăm hết tất cả thành phố.

    Parameters
    ----------
    instance : TSPInstance
        Bài toán TSP cần giải.
    start : int
        Chỉ số thành phố xuất phát (mặc định 0).

    Returns
    -------
    dict với các khóa:
        - 'tour'        : list[int] — tour tìm được
        - 'length'      : float    — tổng độ dài tour
        - 'time_sec'    : float    — thời gian chạy (giây)
        - 'algorithm'   : str
    """
    t0 = time.perf_counter()

    n = instance.n
    dm = instance.dist_matrix
    visited = [False] * n
    tour = [start]
    visited[start] = True

    for _ in range(n - 1):
        current = tour[-1]
        # Tìm thành phố chưa thăm gần nhất
        best_next = -1
        best_dist = float("inf")
        for j in range(n):
            if not visited[j] and dm[current, j] < best_dist:
                best_dist = dm[current, j]
                best_next = j
        tour.append(best_next)
        visited[best_next] = True

    elapsed = time.perf_counter() - t0
    length = instance.tour_length(tour)

    return {
        "tour": tour,
        "length": length,
        "time_sec": elapsed,
        "algorithm": "Greedy (Nearest Neighbor)",
    }


def greedy_best_start(instance: TSPInstance) -> dict:
    """
    Chạy Greedy với mọi thành phố làm điểm xuất phát, trả về tour tốt nhất.
    Tốt hơn greedy_tsp đơn lẻ nhưng O(n²) lần chạy.
    """
    t0 = time.perf_counter()
    best = None

    for start in range(instance.n):
        result = greedy_tsp(instance, start=start)
        if best is None or result["length"] < best["length"]:
            best = result

    best["time_sec"] = time.perf_counter() - t0
    best["algorithm"] = "Greedy Best-Start"
    return best
