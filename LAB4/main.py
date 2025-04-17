import numpy as np
import keyboard
from obj_parser import obj_parser
from config_parser import config_parser
from save_image import save_image
from draw_image import draw_triangle
from PIL import Image, ImageOps
from typing import Tuple, Dict

IMAGE_SIZE = (1000, 1000)

# Функция для построения матрицы поворота по углам (alpha, beta, gamma)
def build_rotation_matrix(a: float, b: float, g: float) -> np.ndarray:
  matrixRX = np.array(
    [[1, 0, 0], [0, np.cos(a), np.sin(a)], [0, -np.sin(a), np.cos(a)]]
  )
  matrixRY = np.array(
    [[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]]
  )
  matrixRZ = np.array(
    [[np.cos(g), np.sin(g), 0], [-np.sin(g), np.cos(g), 0], [0, 0, 1]]
  )
  return matrixRX @ matrixRY @ matrixRZ


# Функция для вычисления нормалей вершин на основе данных модели
def calculate_vertex_normals(
  data: Dict[str, list], rotation_matrix: np.ndarray
) -> np.ndarray:
  vn = np.zeros((len(data["v"]), 3), dtype=np.float32)

  for face in data["f"]:
    v_indices = [vi[0] - 1 for vi in face]
    vertices = np.array([data["v"][i] for i in v_indices])

    rotated_vertices = np.dot(vertices, rotation_matrix)

    edge1 = rotated_vertices[1] - rotated_vertices[0]
    edge2 = rotated_vertices[2] - rotated_vertices[0]
    face_normal = np.cross(edge2, edge1)

    for i in v_indices:
      vn[i] += face_normal

  norms = np.linalg.norm(vn, axis=1)
  return vn / norms[:, np.newaxis]


# Функция для загрузки текстуры из файла
def load_texture(filepath):
  if filepath != "None":
    try:
      texture_img = Image.open(filepath)
      return np.array(ImageOps.flip(texture_img))
    except FileNotFoundError:
      raise (f"Текстура {filepath} не найдена!")
  else:
    return None


# Функция для построения модели и рендера изображения
def build_model(data: Dict[str, list], matrix, z_buff, textures) -> np.ndarray:
  # Вычисление нормалей вершин
  vn = calculate_vertex_normals(data, rotation_matrix)
  # Scale
  scale = RENDER_CONFIG["scale"]

  # Обработка каждой грани модели
  for face in data["f"]:
    # Индексы вершин грани
    v_indices = [vi[0] - 1 for vi in face]
    # Координаты вершин
    vertices = np.array([data["v"][i] for i in v_indices])

    # Поворот и смещение вершин
    rotated_vertices = (
      np.dot(vertices, rotation_matrix) + RENDER_CONFIG["translation_offset"]
    )

    # Текстурные координаты
    tex_coords = [data["vt"][vi[1] - 1] for vi in face]

    # Отрисовка треугольника
    draw_triangle(
      *rotated_vertices.reshape(-1),
      vn[v_indices],
      tex_coords,
      textures,
      matrix,
      z_buff,
      scale,
      H,
      W,
    )

  return matrix


if __name__ == "__main__":
  # H, W - высота и ширина изображения
  H, W = IMAGE_SIZE

  # Инициализация матрицы изображения и Z-буфера
  matrix = np.full((H, W, 3), (0, 0, 0), dtype=np.uint8)
  z_buff = np.full((H, W), np.inf)

  while True:
    if keyboard.is_pressed('F10'):
      print("Обработка модели начата!")

      RENDER_CONFIG = config_parser("../data/config.txt")

      # Загрузка текстуры
      textures = load_texture(RENDER_CONFIG["texture_path"])

      # Построение матрицы поворота
      rotation_matrix = build_rotation_matrix(*RENDER_CONFIG["rotation_angles"])

      # Парсинг модели
      data = obj_parser(RENDER_CONFIG["input_path"])

      # Построение модели
      image_matrix = build_model(data, matrix, z_buff, textures)

      print("Изменения модели сохранены!")

    if keyboard.is_pressed('F11'):
      # Сохранение изображения
      save_image(image_matrix, RENDER_CONFIG["output_path"])
      print("Модель сохранена в виде изображения!")
      break