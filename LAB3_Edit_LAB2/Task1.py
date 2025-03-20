from typing import Tuple
from PIL import Image
import numpy as np

def find_normal(x0: float, y0: float, z0: float, x1: float, y1: float, z1: float, x2: float, y2: float, z2: float) -> np.ndarray:
    n1 = (x1 - x2, y1 - y2, z1 - z2)
    n2 = (x1 - x0, y1 - y0, z1 - z0)

    return np.cross(n1, n2)

def bary(x0: float, y0: float, x1: float, y1: float, x2: float, y2: float, x: int, y: int) -> Tuple[float, float, float]:
    l0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    l1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    l2 = 1.0 - l0 - l1
    return l0, l1, l2

def draw_tr(x0: float, y0: float, z0: float, x1: float, y1: float, z1: float, x2: float, y2: float, z2: float, img_mat: np.ndarray, z_buffer: np.ndarray, H: int, W: int) -> np.ndarray:
    n = find_normal(x0, y0, z0, x1, y1, z1, x2, y2, z2)
    l = [0, 0, 1]

    norm_scalar_mult = (np.dot(n, l)) / (np.linalg.norm(n) * np.linalg.norm(l))
    
    x0_p, x1_p, x2_p = (5000 * i[0] / i[1] + H / 2 for i in ((x0, z0), (x1, z1), (x2, z2)))
    y0_p, y1_p, y2_p = (5000 * i[0] / i[1] + W / 2 for i in ((y0, z0), (y1, z1), (y2, z2)))

    xmin = int(min(x0_p, x1_p, x2_p))
    xmax = int(max(x0_p, x1_p, x2_p) + 1)
    ymin = int(min(y0_p, y1_p, y2_p))
    ymax = int(max(y0_p, y1_p, y2_p) + 1)

    xmin = max(xmin, 0)
    ymin = max(ymin, 0)
    xmax = min(xmax, img_mat.shape[1])
    ymax = min(ymax, img_mat.shape[0])
    color = (-255 * norm_scalar_mult, -255 * norm_scalar_mult, -255 * norm_scalar_mult)

    for i in range(xmin, xmax):
        for j in range(ymin, ymax):
            l0, l1, l2 = bary(x0_p, y0_p, x1_p, y1_p, x2_p, y2_p, i, j)
            if l0 >= 0 and l1 >= 0 and l2 >= 0 and norm_scalar_mult < 0:
                z = l0 * z0 + l1 * z1 + l2 * z2
                if z < z_buffer[j, i]:
                    img_mat[j, i] = color
                    z_buffer[j, i] = z

if __name__ == '__main__':
    img = np.zeros((1000, 1000), dtype=np.uint8)
    draw_tr(1.0, 50.0, 30.0, 15.0, 150.0, 150.0, img)
    img_pil = Image.fromarray(img)
    img_pil.save("test1.png")
