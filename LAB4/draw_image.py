from typing import Tuple
import numpy as np
from typing import List


def bary(
  x0: float, y0: float, x1: float, y1: float, x2: float, y2: float, x: int, y: int
) -> Tuple[float, float, float]:
  # Вычисление барицентрических координат
  l0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / (
    (x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2)
  )
  l1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / (
    (x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2)
  )
  l2 = 1.0 - l0 - l1
  return (l0, l1, l2)


def draw_triangle(
  x0: float,
  y0: float,
  z0: float,
  x1: float,
  y1: float,
  z1: float,
  x2: float,
  y2: float,
  z2: float,
  normals: np.ndarray,
  tex_coords: List[Tuple[float, float]],
  textures: np.ndarray,
  img_mat: np.ndarray,
  z_buffer: np.ndarray,
  scale: int,
  H: int,
  W: int,
) -> np.ndarray:
  t0, t1, t2 = tex_coords

  light_dir = np.array([0, 0, 1], dtype=np.float32)  # Направление света

  # Вычисление интенсивности света для каждой вершины
  I0, I1, I2 = (
    max(0, (np.dot(n, light_dir)) / (np.linalg.norm(n) * np.linalg.norm(light_dir)))
    for n in normals
  )

  # Преобразование координат вершин в экранные координаты
  x0_p, x1_p, x2_p = (
    scale * i[0] / i[1] + H / 2 for i in ((x0, z0), (x1, z1), (x2, z2))
  )
  y0_p, y1_p, y2_p = (
    scale * i[0] / i[1] + W / 2 for i in ((y0, z0), (y1, z1), (y2, z2))
  )

  # Определение границ треугольника
  xmin = max(0, int(min(x0_p, x1_p, x2_p)))
  xmax = min(img_mat.shape[1], int(max(x0_p, x1_p, x2_p) + 1))
  ymin = max(0, int(min(y0_p, y1_p, y2_p)))
  ymax = min(int(max(y0_p, y1_p, y2_p) + 1), img_mat.shape[0])

  for i in range(xmin, xmax):
    for j in range(ymin, ymax):
      # Вычисление барицентрических координат
      l0, l1, l2 = bary(x0_p, y0_p, x1_p, y1_p, x2_p, y2_p, i, j)
      if (
        l0 >= 0 and l1 >= 0 and l2 >= 0
      ):  # Проверка, находится ли точка внутри треугольника
        z = l0 * z0 + l1 * z1 + l2 * z2  # Интерполяция глубины
        if z < z_buffer[j, i]:  # Проверка буфера глубины
          if textures is not None:
            # Интерполяция текстурных координат
            u = l0 * t0[0] + l1 * t1[0] + l2 * t2[0]
            v = l0 * t0[1] + l1 * t1[1] + l2 * t2[1]

            # Преобразование текстурных координат в индексы текстуры
            t_i = min(int(u * textures.shape[1]), textures.shape[1] - 1)
            t_j = min(int(v * textures.shape[0]), textures.shape[0] - 1)

            color = textures[t_j, t_i]  # Получение цвета из текстуры

            # Применение интенсивности света к цвету
            intensity = l0 * I0 + l1 * I1 + l2 * I2
            img_mat[j, i] = (color * intensity).astype(np.uint8)  # Обновление изображения
            z_buffer[j, i] = z  # Обновление буфера глубины
          else:
            color = 255
            intensity = l0 * I0 + l1 * I1 + l2 * I2
            img_mat[j, i] = (color * intensity)
            z_buffer[j, i] = z
