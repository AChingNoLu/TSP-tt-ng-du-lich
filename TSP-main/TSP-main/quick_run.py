"""
Ultra-fast experiment script.
SA: T0=1000, alpha=0.98, max_iter=50 -> ~225 cycles = 11,250 steps (fast)
GA: 100 generations, pop=80
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from src.tsp_core import generate_random_instance, load_tsplib
from src.greedy import greedy_tsp, greedy_best_start
from src.genetic_algorithm import run_ga
from src.simulated_annealing import run_sa
import numpy as np

SEED = 42

# SA fast params
SA_FAST  = dict(initial_temp=2000, cooling_rate=0.98, max_iter_per_temp=50)
SA_MED   = dict(initial_temp=5000, cooling_rate=0.99, max_iter_per_temp=50)
# GA fast params
GA_FAST  = dict(n_generations=100, pop_size=80)
GA_MED   = dict(n_generations=150, pop_size=100)

# ── EXP 1: N=30 ───────────────────────────────────────────────
print("=== EXP1: N=30 ===")
inst30 = generate_random_instance(30, seed=SEED)
g1 = greedy_tsp(inst30)
s1 = run_sa(inst30, seed=SEED, **SA_FAST)
a1 = run_ga(inst30, crossover_type="OX", seed=SEED, **GA_MED)
print(f"Greedy  : {g1['length']:.2f}  t={g1['time_sec']:.4f}s")
print(f"SA      : {s1['best_length']:.2f}  t={s1['time_sec']:.4f}s")
print(f"GA-OX   : {a1['best_length']:.2f}  t={a1['time_sec']:.4f}s")

# ── EXP 2: Crossover OX/PMX/CX, N=30, 3 runs ─────────────────
print("\n=== EXP2: Crossover (3 runs each) ===")
cx_results = {}
for cx in ["OX", "PMX", "CX"]:
    lengths, times = [], []
    for r in range(3):
        res = run_ga(inst30, crossover_type=cx, seed=SEED+r, **GA_MED)
        lengths.append(res["best_length"])
        times.append(res["time_sec"])
    cx_results[cx] = dict(best=min(lengths), mean=np.mean(lengths),
                          std=np.std(lengths), t=np.mean(times))
    print(f"  {cx}: best={min(lengths):.2f}  avg={np.mean(lengths):.2f}  "
          f"std={np.std(lengths):.2f}  t={np.mean(times):.4f}s")

# ── EXP 3: Scalability N=10,30,50 ────────────────────────────
print("\n=== EXP3: Scalability ===")
scale = {}
for n in [10, 30, 50]:
    inst = generate_random_instance(n, seed=SEED)
    gr = greedy_tsp(inst)
    sa = run_sa(inst, seed=SEED, **SA_FAST)
    ga = run_ga(inst, crossover_type="OX", seed=SEED, **GA_MED)
    scale[n] = dict(gl=gr["length"], gt=gr["time_sec"],
                    sl=sa["best_length"], st=sa["time_sec"],
                    al=ga["best_length"], at=ga["time_sec"])
    print(f"  N={n:2d}: Greedy={gr['length']:.2f}({gr['time_sec']:.5f}s) "
          f"SA={sa['best_length']:.2f}({sa['time_sec']:.4f}s) "
          f"GA={ga['best_length']:.2f}({ga['time_sec']:.4f}s)")

# ── EXP 4: berlin52 ──────────────────────────────────────────
print("\n=== EXP4: berlin52 (optimal=7542) ===")
BERLIN = os.path.join("data", "berlin52.tsp")
if os.path.isfile(BERLIN):
    inst52 = load_tsplib(BERLIN)
    gb = greedy_best_start(inst52)
    sb = run_sa(inst52, seed=SEED, **SA_MED)
    ab = run_ga(inst52, crossover_type="OX", seed=SEED, n_generations=150, pop_size=100)
    OPT = 7542.0
    print(f"  Greedy: {gb['length']:.2f}  gap={100*(gb['length']-OPT)/OPT:.1f}%  t={gb['time_sec']:.4f}s")
    print(f"  SA    : {sb['best_length']:.2f}  gap={100*(sb['best_length']-OPT)/OPT:.1f}%  t={sb['time_sec']:.4f}s")
    print(f"  GA-OX : {ab['best_length']:.2f}  gap={100*(ab['best_length']-OPT)/OPT:.1f}%  t={ab['time_sec']:.4f}s")
else:
    print("  berlin52.tsp not found")

print("\n=== DONE ===")
