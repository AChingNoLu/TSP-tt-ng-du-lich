"""
genetic_algorithm.py
====================
Thuật toán Di truyền (Genetic Algorithm) cho bài toán TSP.

Hỗ trợ 3 phương pháp lai ghép:
  - OX  : Order Crossover
  - PMX : Partially Mapped Crossover
  - CX  : Cycle Crossover

Chọn lọc: Tournament Selection
Đột biến : Swap Mutation
"""

import time
import numpy as np
from .tsp_core import TSPInstance, random_tour


# ---------------------------------------------------------------------------
# Crossover operators
# ---------------------------------------------------------------------------

def ox_crossover(p1: list, p2: list, rng: np.random.Generator) -> list:
    """
    Order Crossover (OX).

    1. Chọn đoạn [i, j) ngẫu nhiên từ p1, giữ nguyên vào offspring.
    2. Điền phần còn lại theo thứ tự xuất hiện trong p2
       (bỏ qua các phần tử đã có trong đoạn giữ).

    Parameters
    ----------
    p1, p2 : list[int]
        Hai cá thể cha mẹ (hoán vị).
    rng : np.random.Generator

    Returns
    -------
    list[int]
        Cá thể con.
    """
    n = len(p1)
    i, j = sorted(rng.choice(n, 2, replace=False))

    offspring = [None] * n
    offspring[i:j] = p1[i:j]
    segment_set = set(p1[i:j])

    # Điền các phần tử từ p2 theo thứ tự, bỏ qua phần tử đã có
    remaining = [x for x in p2[j:] + p2[:j] if x not in segment_set]
    pos = list(range(j, n)) + list(range(0, i))
    for idx, val in zip(pos, remaining):
        offspring[idx] = val

    return offspring


def pmx_crossover(p1: list, p2: list, rng: np.random.Generator) -> list:
    """
    Partially Mapped Crossover (PMX).

    1. Chọn đoạn [i, j) ngẫu nhiên.
    2. Copy đoạn từ p1 vào offspring.
    3. Ánh xạ (mapping) 2 chiều giữa đoạn p1 và p2 để điền phần còn lại.

    Parameters
    ----------
    p1, p2 : list[int]
    rng : np.random.Generator

    Returns
    -------
    list[int]
    """
    n = len(p1)
    i, j = sorted(rng.choice(n, 2, replace=False))

    offspring = [None] * n
    offspring[i:j] = p1[i:j]

    # Xây dựng mapping: p2[k] → p1[k] với k trong [i, j)
    mapping = {}
    for k in range(i, j):
        mapping[p2[k]] = p1[k]

    for k in range(n):
        if i <= k < j:
            continue
        val = p2[k]
        # Theo dõi chuỗi ánh xạ để tránh xung đột
        while val in mapping:
            val = mapping[val]
        offspring[k] = val

    return offspring


def cx_crossover(p1: list, p2: list, rng: np.random.Generator) -> list:
    """
    Cycle Crossover (CX).

    Xác định các chu trình (cycles) giữa p1 và p2.
    Chu trình lẻ lấy từ p1, chu trình chẵn lấy từ p2.

    Parameters
    ----------
    p1, p2 : list[int]
    rng : np.random.Generator  (không dùng trong CX nhưng giữ chữ ký nhất quán)

    Returns
    -------
    list[int]
    """
    n = len(p1)
    offspring = [None] * n
    visited = [False] * n

    # Lập index: vị trí của từng gene trong p2
    p2_index = {val: idx for idx, val in enumerate(p2)}

    cycle_num = 0
    for start in range(n):
        if visited[start]:
            continue
        # Truy vết chu trình
        cycle = []
        idx = start
        while not visited[idx]:
            visited[idx] = True
            cycle.append(idx)
            idx = p2_index[p1[idx]]

        # Chu trình chẵn → lấy từ p1; chu trình lẻ → lấy từ p2
        if cycle_num % 2 == 0:
            for pos in cycle:
                offspring[pos] = p1[pos]
        else:
            for pos in cycle:
                offspring[pos] = p2[pos]
        cycle_num += 1

    return offspring


CROSSOVER_FUNCTIONS = {
    "OX": ox_crossover,
    "PMX": pmx_crossover,
    "CX": cx_crossover,
}


# ---------------------------------------------------------------------------
# Mutation
# ---------------------------------------------------------------------------

def swap_mutation(tour: list, rng: np.random.Generator) -> list:
    """
    Swap Mutation: đổi chỗ ngẫu nhiên 2 thành phố trong tour.

    Parameters
    ----------
    tour : list[int]
    rng  : np.random.Generator

    Returns
    -------
    list[int]  (bản sao đã đột biến)
    """
    mutant = tour[:]
    i, j = rng.choice(len(tour), 2, replace=False)
    mutant[i], mutant[j] = mutant[j], mutant[i]
    return mutant


def inversion_mutation(tour: list, rng: np.random.Generator) -> list:
    """
    Inversion Mutation (2-opt style): đảo ngược một đoạn con ngẫu nhiên.
    """
    mutant = tour[:]
    i, j = sorted(rng.choice(len(tour), 2, replace=False))
    mutant[i:j+1] = mutant[i:j+1][::-1]
    return mutant


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def tournament_selection(population: list, fitnesses: np.ndarray,
                         k: int, rng: np.random.Generator) -> list:
    """
    Tournament Selection: chọn k cá thể ngẫu nhiên, trả về cá thể tốt nhất.

    Parameters
    ----------
    population : list[list[int]]
        Quần thể hiện tại.
    fitnesses : np.ndarray
        Giá trị fitness tương ứng (cao hơn = tốt hơn).
    k : int
        Kích thước tournament.
    rng : np.random.Generator

    Returns
    -------
    list[int]  — cá thể thắng tournament
    """
    indices = rng.choice(len(population), k, replace=False)
    best_idx = indices[np.argmax(fitnesses[indices])]
    return population[best_idx]


# ---------------------------------------------------------------------------
# Main GA
# ---------------------------------------------------------------------------

def run_ga(
    instance: TSPInstance,
    pop_size: int = 150,
    n_generations: int = 500,
    crossover_type: str = "OX",
    crossover_rate: float = 0.85,
    mutation_rate: float = 0.02,
    tournament_k: int = 5,
    elitism_size: int = 5,
    seed: int = None,
    verbose: bool = False,
) -> dict:
    """
    Chạy Genetic Algorithm giải bài toán TSP.

    Parameters
    ----------
    instance : TSPInstance
    pop_size : int
        Kích thước quần thể.
    n_generations : int
        Số thế hệ tiến hóa.
    crossover_type : str
        'OX', 'PMX', hoặc 'CX'.
    crossover_rate : float
        Xác suất áp dụng lai ghép (ngược lại sao chép nguyên).
    mutation_rate : float
        Xác suất đột biến mỗi cá thể.
    tournament_k : int
        Kích thước tournament trong chọn lọc.
    elitism_size : int
        Số cá thể tốt nhất giữ nguyên qua mỗi thế hệ.
    seed : int, optional
        Random seed.
    verbose : bool
        In tiến trình ra console.

    Returns
    -------
    dict với các khóa:
        - 'best_tour'       : list[int]
        - 'best_length'     : float
        - 'history'         : list[float] — best length mỗi thế hệ
        - 'time_sec'        : float
        - 'algorithm'       : str
        - 'crossover_type'  : str
    """
    if crossover_type not in CROSSOVER_FUNCTIONS:
        raise ValueError(f"crossover_type phải là một trong {list(CROSSOVER_FUNCTIONS)}")

    rng = np.random.default_rng(seed)
    crossover_fn = CROSSOVER_FUNCTIONS[crossover_type]
    n = instance.n

    t0 = time.perf_counter()

    # --- Khởi tạo quần thể ---
    population = [random_tour(n, rng) for _ in range(pop_size)]

    # --- Hàm fitness ---
    def compute_fitness(pop):
        lengths = np.array([instance.tour_length(t) for t in pop])
        # Tránh chia cho 0 (không xảy ra trong thực tế)
        return 1.0 / (lengths + 1e-10), lengths

    history = []  # best length theo từng thế hệ
    best_tour = None
    best_length = float("inf")

    for gen in range(n_generations):
        fitnesses, lengths = compute_fitness(population)

        # Cập nhật best toàn cục
        gen_best_idx = int(np.argmin(lengths))
        if lengths[gen_best_idx] < best_length:
            best_length = lengths[gen_best_idx]
            best_tour = population[gen_best_idx][:]

        history.append(best_length)

        if verbose and (gen % 50 == 0 or gen == n_generations - 1):
            print(f"  Gen {gen:4d}/{n_generations} | Best length: {best_length:.2f}")

        # --- Elitism: giữ nguyên top cá thể ---
        elite_indices = np.argsort(lengths)[:elitism_size]
        elites = [population[i][:] for i in elite_indices]

        # --- Tạo thế hệ mới ---
        new_population = elites[:]
        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, fitnesses, tournament_k, rng)
            parent2 = tournament_selection(population, fitnesses, tournament_k, rng)

            # Lai ghép
            if rng.random() < crossover_rate:
                child = crossover_fn(parent1, parent2, rng)
            else:
                child = parent1[:]

            # Đột biến
            if rng.random() < mutation_rate:
                child = swap_mutation(child, rng)

            new_population.append(child)

        population = new_population[:pop_size]

    elapsed = time.perf_counter() - t0

    return {
        "best_tour": best_tour,
        "best_length": best_length,
        "history": history,
        "time_sec": elapsed,
        "algorithm": f"GA-{crossover_type}",
        "crossover_type": crossover_type,
        "generations": n_generations,
        "pop_size": pop_size,
    }
