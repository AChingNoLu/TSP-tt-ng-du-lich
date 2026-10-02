# TSP Solver - Genetic Algorithm, Greedy, Simulated Annealing

Dự án này giải quyết bài toán Người du lịch (Traveling Salesman Problem - TSP) bằng cách so sánh ba phương pháp:

- Genetic Algorithm (GA)
- Greedy / Nearest Neighbor
- Simulated Annealing (SA)

Mục tiêu là tìm đường đi có chi phí thấp nhất, đồng thời đánh giá hiệu quả của từng thuật toán trên dữ liệu ngẫu nhiên và dữ liệu chuẩn `berlin52.tsp`.

## Mục tiêu của dự án

- Giải bài toán TSP trên tập dữ liệu khác nhau
- So sánh chất lượng lời giải giữa các thuật toán
- Phân tích sự hội tụ và độ ổn định của Genetic Algorithm
- Vẽ các biểu đồ kết quả và lưu ra thư mục `results/`

## Cấu trúc thư mục

```text
TSP-main/
├── data/
│   └── berlin52.tsp
├── results/
│   └── (nhiều file hình ảnh và CSV sau khi chạy)
├── src/
│   ├── __init__.py
│   ├── genetic_algorithm.py
│   ├── greedy.py
│   ├── simulated_annealing.py
│   ├── tsp_core.py
│   └── utils.py
├── main.py
├── quick_run.py
├── .gitignore
├── README.md
└── .
```

## Yêu cầu hệ thống

- Python 3.9+
- Pip

## Cài đặt

### 1. Tạo môi trường ảo (khuyến nghị)

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

Linux / macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Cài đặt thư viện cần thiết

```bash
pip install numpy pandas matplotlib tqdm
```

Nếu dự án có thêm file `requirements.txt` sau này, bạn có thể dùng:

```bash
pip install -r requirements.txt
```

## Chạy dự án

### Chạy toàn bộ thí nghiệm

```bash
python main.py
```

File `main.py` sẽ:

- tạo bài toán TSP ngẫu nhiên
- chạy Greedy, Simulated Annealing và GA
- so sánh hiệu năng giữa các thuật toán
- vẽ các đồ thị và lưu kết quả vào `results/`

### Chạy nhanh / demo nhanh

```bash
python quick_run.py
```

File này là phiên bản rút gọn, phù hợp để kiểm tra nhanh từng thí nghiệm mà không cần chạy toàn bộ pipeline phức tạp.

## Các thuật toán trong dự án

### 1. Genetic Algorithm (GA)

Thuật toán di truyền dựa trên khái niệm tiến hóa quần thể:

- mã hóa nghiệm dưới dạng hoán vị
- chọn lọc theo độ thích nghi
- lai ghép với các kiểu crossover như OX, PMX, CX
- đột biến bằng cách hoán đổi vị trí
- giữ elitism để không mất nghiệm tốt

Các kiểu crossover có sẵn:

- OX (Order Crossover)
- PMX (Partially Mapped Crossover)
- CX (Cycle Crossover)

### 2. Greedy Algorithm

Phương pháp đơn giản, chọn thành phố gần nhất ở mỗi bước:

- nhanh
- dễ triển khai
- thường cho lời giải tốt nhưng không tối ưu tuyệt đối

### 3. Simulated Annealing

Thuật toán mô phỏng quá trình làm nguội vật liệu:

- bắt đầu với một lời giải bất kỳ
- cho phép “đổi xấu” theo xác suất để tránh rơi vào cực tiểu cục bộ
- giảm nhiệt độ theo thời gian để hội tụ về nghiệm tốt hơn

## Dữ liệu đầu vào

Dự án hỗ trợ hai loại dữ liệu:

1. Dữ liệu ngẫu nhiên
   - sinh ngẫu nhiên số lượng thành phố theo yêu cầu
   - hữu ích để kiểm tra tốc độ và hiệu năng

2. Dữ liệu TSPLIB
   - file `data/berlin52.tsp`
   - là tập dữ liệu chuẩn cho bài toán TSP
   - giá trị tối ưu tham chiếu là khoảng 7542

## Kết quả đầu ra

Khi chạy xong, các file dưới đây sẽ được tạo trong thư mục `results/`:

- `exp1_convergence_ga.png`
- `exp1_tour_comparison.png`
- `exp1_bar_comparison.png`
- `exp2_crossover_convergence.png`
- `exp2_crossover_comparison.csv`
- `exp3_scalability_length.png`
- `exp3_scalability_time.png`
- `exp4_berlin52...` (nếu có thực thi mô phỏng trên file dữ liệu chuẩn)

## Ví dụ dùng API

```python
from src.tsp_core import generate_random_instance
from src.genetic_algorithm import run_ga

instance = generate_random_instance(n=30, seed=42)
result = run_ga(instance, crossover_type="OX", seed=42, n_generations=200, verbose=True)

print(f"Best length: {result['best_length']:.2f}")
print(f"Time: {result['time_sec']:.4f}s")
```

## Ghi chú

- Kết quả số liệu thực tế có thể thay đổi theo máy và seed
- Đối với dữ liệu lớn, Genetic Algorithm và Simulated Annealing sẽ tốn thời gian hơn so với Greedy
- Dự án được thiết kế để phục vụ mục đích nghiên cứu, demo và học tập

## Tài liệu tham khảo

- Goldberg, David E. - Genetic Algorithms in Search, Optimization, and Machine Learning
- Kirkpatrick et al. - Optimization by Simulated Annealing
- TSPLIB benchmark dataset

## License

Dự án này được cung cấp miễn phí cho mục đích học tập và nghiên cứu.
