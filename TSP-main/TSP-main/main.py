# -*- coding: utf-8 -*-
"""
main.py
=======
Runner chinh: thuc hien toan bo thi nghiem va xuat ket qua.

Thi nghiem:
  1. So sanh GA (OX/PMX/CX) vs Greedy vs SA tren random N=30
  2. So sanh 3 crossover OX/PMX/CX (5 lan chay -> lay TB)
  3. Scalability: N = 10, 30, 50
  4. (Neu co data/berlin52.tsp) Kiem tra tren TSPLIB berlin52
"""
import sys
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import os
import numpy as np
import pandas as pd

# Đảm bảo import được package src khi chạy từ thư mục gốc
sys.path.insert(0, os.path.dirname(__file__))

from src.tsp_core import generate_random_instance, load_tsplib
from src.greedy import greedy_tsp, greedy_best_start
from src.genetic_algorithm import run_ga
from src.simulated_annealing import run_sa
from src.utils import (
    plot_convergence, plot_tours, plot_comparison_bar,
    plot_scalability, make_summary_table, print_table, save_table_csv
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
DATA_DIR    = os.path.join(os.path.dirname(__file__), "data")
BERLIN52    = os.path.join(DATA_DIR, "berlin52.tsp")

# ============================================================
# Thí nghiệm 1: So sánh GA vs Greedy vs SA (N=30)
# ============================================================

def experiment_main_comparison(seed: int = 42):
    print("\n" + "="*60)
    print("THÍ NGHIỆM 1: GA vs Greedy vs SA  (N=30)")
    print("="*60)

    instance = generate_random_instance(n=30, seed=seed)
    print(f"Bài toán: {instance}")

    # --- Greedy ---
    greedy_res = greedy_tsp(instance, start=0)
    print(f"\nGreedy: length={greedy_res['length']:.2f}  time={greedy_res['time_sec']:.4f}s")

    # --- SA ---
    sa_res = run_sa(instance, seed=seed, verbose=False)
    print(f"SA    : length={sa_res['best_length']:.2f}  time={sa_res['time_sec']:.4f}s")

    # --- GA (OX) ---
    ga_res = run_ga(instance, crossover_type="OX", seed=seed,
                    n_generations=200, verbose=False)
    print(f"GA-OX : length={ga_res['best_length']:.2f}  time={ga_res['time_sec']:.4f}s")

    # Vẽ convergence GA
    plot_convergence(
        [ga_res],
        title=f"Convergence GA-OX — {instance.name}",
        save_path=os.path.join(RESULTS_DIR, "exp1_convergence_ga.png"),
    )

    # Vẽ tour so sánh
    greedy_res_fmt = {**greedy_res, "tour": greedy_res["tour"]}
    sa_res_fmt     = {"tour": sa_res["best_tour"], "length": sa_res["best_length"],
                      "algorithm": sa_res["algorithm"]}
    ga_res_fmt     = {"tour": ga_res["best_tour"], "length": ga_res["best_length"],
                      "algorithm": ga_res["algorithm"]}

    plot_tours(
        instance,
        [greedy_res_fmt, sa_res_fmt, ga_res_fmt],
        title=f"Tour Comparison — {instance.name}",
        save_path=os.path.join(RESULTS_DIR, "exp1_tour_comparison.png"),
    )

    # Bảng tổng hợp
    records = [
        {"algorithm": greedy_res["algorithm"], "best_length": greedy_res["length"],
         "time_sec": greedy_res["time_sec"], "n_cities": instance.n},
        {"algorithm": sa_res["algorithm"],     "best_length": sa_res["best_length"],
         "time_sec": sa_res["time_sec"],       "n_cities": instance.n},
        {"algorithm": ga_res["algorithm"],     "best_length": ga_res["best_length"],
         "time_sec": ga_res["time_sec"],       "n_cities": instance.n},
    ]
    df = make_summary_table(records)
    print_table(df, "Kết quả Thí nghiệm 1")
    save_table_csv(df, os.path.join(RESULTS_DIR, "exp1_main_comparison.csv"))

    # Bar chart
    bar_data = {
        greedy_res["algorithm"]: greedy_res["length"],
        sa_res["algorithm"]:     sa_res["best_length"],
        ga_res["algorithm"]:     ga_res["best_length"],
    }
    plot_comparison_bar(
        bar_data,
        title=f"Tour Length — GA vs Greedy vs SA (N={instance.n})",
        save_path=os.path.join(RESULTS_DIR, "exp1_bar_comparison.png"),
    )

    return instance, greedy_res, sa_res, ga_res


# ============================================================
# Thí nghiệm 2: So sánh 3 crossover (OX / PMX / CX)
# ============================================================

def experiment_crossover_comparison(n_cities: int = 30, n_runs: int = 3, seed: int = 42):
    print("\n" + "="*60)
    print(f"THÍ NGHIỆM 2: OX vs PMX vs CX  (N={n_cities}, {n_runs} lần chạy)")
    print("="*60)

    instance = generate_random_instance(n=n_cities, seed=seed)
    crossovers = ["OX", "PMX", "CX"]
    all_results = {cx: [] for cx in crossovers}
    conv_results = []  # 1 lần chạy để vẽ convergence

    for cx in crossovers:
        lengths = []
        times   = []
        best_run = None
        for run in range(n_runs):
            r = run_ga(instance, crossover_type=cx,
                       seed=seed + run, n_generations=200, verbose=False)
            lengths.append(r["best_length"])
            times.append(r["time_sec"])
            if best_run is None or r["best_length"] < best_run["best_length"]:
                best_run = r
        all_results[cx] = {"lengths": lengths, "times": times, "best_run": best_run}
        conv_results.append(best_run)
        print(f"  {cx}: avg={np.mean(lengths):.2f}  std={np.std(lengths):.2f}  "
              f"best={np.min(lengths):.2f}  time_avg={np.mean(times):.4f}s")

    # Convergence plot (3 crossover cùng 1 đồ thị)
    plot_convergence(
        conv_results,
        title=f"Convergence — OX vs PMX vs CX (N={n_cities})",
        save_path=os.path.join(RESULTS_DIR, "exp2_crossover_convergence.png"),
    )

    # Bảng so sánh
    records = []
    for cx, data in all_results.items():
        records.append({
            "algorithm": f"GA-{cx}",
            "n_cities": n_cities,
            "best_length": np.min(data["lengths"]),
            "avg_length": np.mean(data["lengths"]),
            "std_length": np.std(data["lengths"]),
            "time_sec": np.mean(data["times"]),
        })
    df = make_summary_table(records)
    print_table(df, "Kết quả Thí nghiệm 2 — So sánh Crossover")
    save_table_csv(df, os.path.join(RESULTS_DIR, "exp2_crossover_comparison.csv"))

    return all_results


# ============================================================
# Thí nghiệm 3: Scalability — N = 10, 30, 50
# ============================================================

def experiment_scalability(sizes: list = None, seed: int = 42):
    if sizes is None:
        sizes = [10, 30, 50]

    print("\n" + "="*60)
    print(f"THÍ NGHIỆM 3: Scalability — N = {sizes}")
    print("="*60)

    # Theo dõi kết quả theo algo
    length_by_algo = {"GA-OX": [], "Greedy (Nearest Neighbor)": [], "Simulated Annealing": []}
    time_by_algo   = {"GA-OX": [], "Greedy (Nearest Neighbor)": [], "Simulated Annealing": []}
    all_records = []

    for n in sizes:
        print(f"\n  --- N = {n} ---")
        instance = generate_random_instance(n=n, seed=seed)

        greedy_r = greedy_tsp(instance)
        ga_r     = run_ga(instance, crossover_type="OX", seed=seed,
                          n_generations=200, verbose=False)
        sa_r     = run_sa(instance, seed=seed, verbose=False)

        print(f"  Greedy: {greedy_r['length']:.2f} ({greedy_r['time_sec']:.4f}s)")
        print(f"  GA-OX : {ga_r['best_length']:.2f} ({ga_r['time_sec']:.4f}s)")
        print(f"  SA    : {sa_r['best_length']:.2f} ({sa_r['time_sec']:.4f}s)")

        length_by_algo["GA-OX"].append(ga_r["best_length"])
        length_by_algo["Greedy (Nearest Neighbor)"].append(greedy_r["length"])
        length_by_algo["Simulated Annealing"].append(sa_r["best_length"])

        time_by_algo["GA-OX"].append(ga_r["time_sec"])
        time_by_algo["Greedy (Nearest Neighbor)"].append(greedy_r["time_sec"])
        time_by_algo["Simulated Annealing"].append(sa_r["time_sec"])

        for r, length, t in [
            (greedy_r["algorithm"], greedy_r["length"], greedy_r["time_sec"]),
            (ga_r["algorithm"],     ga_r["best_length"], ga_r["time_sec"]),
            (sa_r["algorithm"],     sa_r["best_length"], sa_r["time_sec"]),
        ]:
            all_records.append({"algorithm": r, "n_cities": n,
                                 "best_length": length, "time_sec": t})

    # Scalability plots
    plot_scalability(
        sizes, length_by_algo,
        title="Scalability — Tour Length theo N",
        ylabel="Tour Length",
        save_path=os.path.join(RESULTS_DIR, "exp3_scalability_length.png"),
    )
    plot_scalability(
        sizes, time_by_algo,
        title="Scalability — Thời gian chạy theo N",
        ylabel="Time (seconds)",
        save_path=os.path.join(RESULTS_DIR, "exp3_scalability_time.png"),
    )

    df = make_summary_table(all_records)
    print_table(df, "Kết quả Thí nghiệm 3 — Scalability")
    save_table_csv(df, os.path.join(RESULTS_DIR, "exp3_scalability.csv"))

    return length_by_algo, time_by_algo


# ============================================================
# Thí nghiệm 4 (Bonus): TSPLIB berlin52
# ============================================================

def experiment_berlin52(seed: int = 42):
    if not os.path.isfile(BERLIN52):
        print(f"\n[SKIP] Không tìm thấy {BERLIN52}. Bỏ qua thí nghiệm berlin52.")
        return None

    print("\n" + "="*60)
    print("THÍ NGHIỆM 4: TSPLIB berlin52 (optimal ≈ 7542)")
    print("="*60)

    instance = load_tsplib(BERLIN52)
    print(f"Bài toán: {instance}")

    greedy_r = greedy_best_start(instance)
    ga_r     = run_ga(instance, crossover_type="OX", seed=seed,
                      n_generations=300, pop_size=150, verbose=True)
    sa_r     = run_sa(instance, initial_temp=50000, seed=seed, verbose=False)

    OPTIMAL  = 7542.0
    print(f"\nGreedy : {greedy_r['length']:.2f}  (gap={100*(greedy_r['length']-OPTIMAL)/OPTIMAL:.1f}%)")
    print(f"GA-OX  : {ga_r['best_length']:.2f}  (gap={100*(ga_r['best_length']-OPTIMAL)/OPTIMAL:.1f}%)")
    print(f"SA     : {sa_r['best_length']:.2f}  (gap={100*(sa_r['best_length']-OPTIMAL)/OPTIMAL:.1f}%)")
    print(f"Optimal: {OPTIMAL:.2f}")

    plot_convergence(
        [ga_r],
        title="Convergence GA-OX — berlin52",
        save_path=os.path.join(RESULTS_DIR, "exp4_berlin52_convergence.png"),
    )

    greedy_fmt = {"tour": greedy_r["tour"], "length": greedy_r["length"],
                  "algorithm": greedy_r["algorithm"]}
    sa_fmt     = {"tour": sa_r["best_tour"], "length": sa_r["best_length"],
                  "algorithm": sa_r["algorithm"]}
    ga_fmt     = {"tour": ga_r["best_tour"], "length": ga_r["best_length"],
                  "algorithm": ga_r["algorithm"]}
    plot_tours(
        instance, [greedy_fmt, sa_fmt, ga_fmt],
        title="berlin52 — Tour Comparison",
        save_path=os.path.join(RESULTS_DIR, "exp4_berlin52_tours.png"),
    )

    records = [
        {"algorithm": "Optimal (known)",   "best_length": OPTIMAL, "time_sec": 0, "n_cities": 52},
        {"algorithm": greedy_r["algorithm"], "best_length": greedy_r["length"],
         "time_sec": greedy_r["time_sec"], "n_cities": 52},
        {"algorithm": sa_r["algorithm"],   "best_length": sa_r["best_length"],
         "time_sec": sa_r["time_sec"],     "n_cities": 52},
        {"algorithm": ga_r["algorithm"],   "best_length": ga_r["best_length"],
         "time_sec": ga_r["time_sec"],     "n_cities": 52},
    ]
    df = make_summary_table(records)
    print_table(df, "Kết quả Thí nghiệm 4 — berlin52")
    save_table_csv(df, os.path.join(RESULTS_DIR, "exp4_berlin52.csv"))

    return ga_r, greedy_r, sa_r


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n" + "*"*60)
    print("  TSP -- Genetic Algorithm Project")
    print("*"*60)

    experiment_main_comparison(seed=42)
    experiment_crossover_comparison(n_cities=30, n_runs=3, seed=42)
    experiment_scalability(sizes=[10, 30, 50], seed=42)
    experiment_berlin52(seed=42)

    print("\n" + "*"*60)
    print("  Hoan thanh! Ket qua luu tai thu muc: results/")
    print("*"*60)
