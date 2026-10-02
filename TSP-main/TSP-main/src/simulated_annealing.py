"""
simulated_annealing.py
======================
Thuật toán Simulated Annealing (SA) cho bài toán TSP.

Sử dụng Metropolis criterion để chấp nhận nghiệm tệ hơn với xác suất
phụ thuộc nhiệt độ. Nhiệt độ giảm dần theo lịch làm nguội hình học.
"""

import time
import math
import numpy as np
from .tsp_core import TSPInstance, random_tour


def run_sa(
    instance: TSPInstance,
    initial_temp: float = 10000.0,
    cooling_rate: float = 0.995,
    min_temp: float = 1e-8,
    max_iter_per_temp: int = 100,
    seed: int = None,
    verbose: bool = False,
) -> dict:
    """
    Chạy Simulated Annealing giải bài toán TSP.

    Neighbor generation: hoán đổi 2 thành phố ngẫu nhiên (swap move).
    Acceptance criterion: Metropolis — chấp nhận nghiệm tệ hơn với
    xác suất  exp(−Δ/T).

    Parameters
    ----------
    instance : TSPInstance
    initial_temp : float
        Nhiệt độ ban đầu T₀.
    cooling_rate : float
        Hệ số làm nguội α (0 < α < 1). T_{k+1} = α × T_k.
    min_temp : float
        Nhiệt độ tối thiểu để dừng.
    max_iter_per_temp : int
        Số bước thử nghiệm tại mỗi mức nhiệt độ.
    seed : int, optional
    verbose : bool

    Returns
    -------
    dict với các khóa:
        - 'best_tour'   : list[int]
        - 'best_length' : float
        - 'history'     : list[float] — best length sau mỗi chu kỳ nhiệt độ
        - 'time_sec'    : float
        - 'algorithm'   : str
    """
    rng = np.random.default_rng(seed)
    n = instance.n
    t0 = time.perf_counter()

    # Khởi tạo trạng thái ban đầu (ngẫu nhiên)
    current_tour = random_tour(n, rng)
    current_length = instance.tour_length(current_tour)

    best_tour = current_tour[:]
    best_length = current_length

    T = initial_temp
    history = []
    cycle = 0

    while T > min_temp:
        for _ in range(max_iter_per_temp):
            # Sinh nghiệm lân cận: hoán đổi 2 thành phố ngẫu nhiên
            i, j = rng.choice(n, 2, replace=False)
            neighbor = current_tour[:]
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            neighbor_length = instance.tour_length(neighbor)

            delta = neighbor_length - current_length

            # Metropolis criterion
            if delta < 0 or rng.random() < math.exp(-delta / T):
                current_tour = neighbor
                current_length = neighbor_length

            # Cập nhật best
            if current_length < best_length:
                best_length = current_length
                best_tour = current_tour[:]

        history.append(best_length)

        # Làm nguội
        T *= cooling_rate
        cycle += 1

        if verbose and cycle % 500 == 0:
            print(f"  Cycle {cycle:5d} | T={T:.4f} | Best: {best_length:.2f}")

    elapsed = time.perf_counter() - t0

    return {
        "best_tour": best_tour,
        "best_length": best_length,
        "history": history,
        "time_sec": elapsed,
        "algorithm": "Simulated Annealing",
    }
