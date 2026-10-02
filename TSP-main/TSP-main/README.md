# TSP — Genetic Algorithm Project

Giải bài toán Người du lịch (TSP) bằng Thuật toán Di truyền (GA), so sánh với Greedy và Simulated Annealing.

**Đề số 9 — Môn: Thuật toán & Ứng dụng**

---

## Cấu trúc dự án

```
tsp_ga/
├── src/
│   ├── tsp_core.py           # Dữ liệu thành phố, TSPLIB parser, tính khoảng cách
│   ├── genetic_algorithm.py  # GA hoàn chỉnh (OX, PMX, CX crossover)
│   ├── greedy.py             # Thuật toán Greedy Nearest Neighbor
│   ├── simulated_annealing.py# Thuật toán Simulated Annealing
│   └── utils.py              # Vẽ đồ thị, xuất bảng CSV
├── data/
│   └── berlin52.tsp          # Bộ dữ liệu TSPLIB chuẩn (optimal = 7542)
├── results/                  # Đồ thị PNG + bảng CSV (tự sinh khi chạy)
├── main.py                   # Runner chính — chạy toàn bộ thí nghiệm
├── requirements.txt
└── README.md
```

---

## Cài đặt

### Yêu cầu
- Python 3.9+
- Các thư viện: `numpy`, `matplotlib`, `pandas`, `tqdm`

### Bước 1 — Tạo môi trường ảo (khuyến nghị)

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### Bước 2 — Cài thư viện

```bash
pip install -r requirements.txt
```

---

## Cách chạy

### Chạy toàn bộ thí nghiệm

```bash
python main.py
```

Kết quả sẽ được lưu vào thư mục `results/`:
- `exp1_convergence_ga.png` — đường hội tụ GA
- `exp1_tour_comparison.png` — so sánh tour 3 thuật toán
- `exp1_bar_comparison.png` — biểu đồ cột độ dài tour
- `exp2_crossover_convergence.png` — hội tụ OX vs PMX vs CX
- `exp2_crossover_comparison.csv` — bảng so sánh crossover
- `exp3_scalability_length.png` — scalability theo N
- `exp3_scalability_time.png` — thời gian chạy theo N
- `exp4_berlin52_convergence.png` — convergence trên berlin52
- `exp4_berlin52_tours.png` — tour trên berlin52

### Chạy riêng từng module

```python
from src.tsp_core import generate_random_instance
from src.genetic_algorithm import run_ga
from src.greedy import greedy_tsp
from src.simulated_annealing import run_sa

# Sinh bài toán 30 thành phố
instance = generate_random_instance(n=30, seed=42)

# Chạy GA với OX crossover
result = run_ga(instance, crossover_type="OX", n_generations=300, verbose=True)
print(f"Best tour length: {result['best_length']:.2f}")
print(f"Time: {result['time_sec']:.4f}s")
```

---

## Thuật toán

### 1. Genetic Algorithm (GA)

| Thành phần | Lựa chọn |
|---|---|
| Biểu diễn | Hoán vị (permutation encoding) |
| Chọn lọc | Tournament Selection (k=5) |
| Lai ghép | OX / PMX / CX |
| Đột biến | Swap Mutation |
| Chiến lược | Elitism (giữ top-5 mỗi thế hệ) |

**Tham số mặc định:**

| Tham số | Giá trị |
|---|---|
| Population size | 150 |
| Số thế hệ | 500 |
| Crossover rate | 0.85 |
| Mutation rate | 0.02 |
| Tournament k | 5 |
| Elitism size | 5 |

**Crossover operators:**

- **OX (Order Crossover)**: Giữ đoạn con từ parent 1, điền phần còn lại theo thứ tự parent 2.
- **PMX (Partially Mapped Crossover)**: Ánh xạ 2 chiều giữa 2 đoạn con.
- **CX (Cycle Crossover)**: Xác định chu trình, gán xen kẽ từ 2 cha mẹ.

### 2. Greedy (Nearest Neighbor)

Bắt đầu từ thành phố xuất phát, luôn di chuyển đến thành phố chưa thăm gần nhất. Độ phức tạp O(n²).

### 3. Simulated Annealing (SA)

| Tham số | Giá trị |
|---|---|
| Nhiệt độ ban đầu T₀ | 10000 |
| Hệ số làm nguội α | 0.995 |
| Nhiệt độ tối thiểu | 1e-8 |
| Neighbor | Swap 2 thành phố |

---

## Kết quả mẫu (N=30, seed=42)

| Thuật toán | Tour Length | Thời gian (s) |
|---|---|---|
| Greedy | ~5800 | < 0.001 |
| Simulated Annealing | ~4200 | ~2.0 |
| GA-OX | ~4100 | ~8.0 |

*(Kết quả thực tế có thể khác nhau tùy máy)*

---

## Dữ liệu

- **Ngẫu nhiên**: Tọa độ sinh trong không gian [0, 1000] × [0, 1000]
- **TSPLIB berlin52**: 52 địa điểm tại Berlin, lời giải tối ưu đã biết = **7542**

---

## Tham khảo

1. Goldberg, D.E. (1989). *Genetic Algorithms in Search, Optimization and Machine Learning*.
2. Davis, L. (1985). Applying Adaptive Algorithms to Epistatic Domains. IJCAI.
3. Kirkpatrick, S., Gelatt, C.D., Vecchi, M.P. (1983). Optimization by Simulated Annealing. *Science*.
4. TSPLIB: http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/
