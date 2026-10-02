"""
tsp_core.py
===========
Core data structures và utilities cho bài toán TSP.
- Sinh tọa độ thành phố ngẫu nhiên
- Parse file TSPLIB (.tsp)
- Tính ma trận khoảng cách
- Tính tổng độ dài tour
"""

import numpy as np
import os


class TSPInstance:
    """Đại diện cho một bài toán TSP cụ thể."""

    def __init__(self, coords: np.ndarray, name: str = "TSP"):
        """
        Parameters
        ----------
        coords : np.ndarray, shape (n, 2)
            Tọa độ (x, y) của n thành phố.
        name : str
            Tên bài toán để hiển thị/ghi log.
        """
        self.coords = np.array(coords, dtype=float)
        self.n = len(coords)
        self.name = name
        self._dist_matrix = None  # lazy computed

    @property
    def dist_matrix(self) -> np.ndarray:
        """Ma trận khoảng cách Euclidean n×n (tính một lần, cache lại)."""
        if self._dist_matrix is None:
            self._dist_matrix = self._compute_dist_matrix()
        return self._dist_matrix

    def _compute_dist_matrix(self) -> np.ndarray:
        """Tính ma trận khoảng cách Euclidean."""
        n = self.n
        c = self.coords
        # Broadcast trick: nhanh hơn vòng lặp kép
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]  # (n, n, 2)
        return np.sqrt((diff ** 2).sum(axis=2))

    def tour_length(self, tour: list) -> float:
        """
        Tính tổng độ dài của một tour.

        Parameters
        ----------
        tour : list[int]
            Hoán vị các chỉ số thành phố, ví dụ [0, 3, 1, 4, 2].

        Returns
        -------
        float
            Tổng khoảng cách (bao gồm cạnh quay về điểm xuất phát).
        """
        dm = self.dist_matrix
        total = sum(dm[tour[i], tour[i + 1]] for i in range(self.n - 1))
        total += dm[tour[-1], tour[0]]  # quay về điểm đầu
        return total

    def __repr__(self):
        return f"TSPInstance(name={self.name!r}, n={self.n})"


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def generate_random_instance(n: int, seed: int = None, name: str = None) -> TSPInstance:
    """
    Sinh ngẫu nhiên n thành phố trong mặt phẳng [0, 1000] × [0, 1000].

    Parameters
    ----------
    n : int
        Số thành phố.
    seed : int, optional
        Random seed để tái hiện kết quả.
    name : str, optional
        Tên bài toán.

    Returns
    -------
    TSPInstance
    """
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 1000, size=(n, 2))
    instance_name = name or f"random_{n}"
    return TSPInstance(coords, name=instance_name)


def load_tsplib(filepath: str) -> TSPInstance:
    """
    Parse file TSPLIB định dạng .tsp (EUC_2D).

    Parameters
    ----------
    filepath : str
        Đường dẫn đến file .tsp.

    Returns
    -------
    TSPInstance

    Raises
    ------
    FileNotFoundError
        Nếu file không tồn tại.
    ValueError
        Nếu định dạng file không hỗ trợ.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Không tìm thấy file: {filepath}")

    coords = []
    name = os.path.splitext(os.path.basename(filepath))[0]
    in_node_section = False

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("NAME"):
                name = line.split(":")[1].strip() if ":" in line else name
            elif line == "NODE_COORD_SECTION":
                in_node_section = True
            elif line == "EOF":
                break
            elif in_node_section:
                parts = line.split()
                if len(parts) >= 3:
                    # Format: <index> <x> <y>
                    coords.append([float(parts[1]), float(parts[2])])

    if not coords:
        raise ValueError(f"Không đọc được tọa độ từ file {filepath}. "
                         "Kiểm tra định dạng EUC_2D.")

    return TSPInstance(np.array(coords), name=name)


def random_tour(n: int, rng: np.random.Generator = None) -> list:
    """Sinh ngẫu nhiên một tour hợp lệ."""
    if rng is None:
        rng = np.random.default_rng()
    tour = list(range(n))
    rng.shuffle(tour)
    return tour
