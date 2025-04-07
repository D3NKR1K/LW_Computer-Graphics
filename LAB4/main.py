import numpy as np
from obj_parser import obj_parser
from save_image import save_image
from draw_image import draw_triangle
from PIL import Image, ImageOps
from typing import Tuple, Dict

# Конфигурация рендера, задает параметры изображения, поворота, текстуры и смещения
RENDER_CONFIG = {
  "image_size": (1000, 1000),  # Размер изображения (в пикселях)
  "rotation_angles": (0.0, 0.0, 0.0),  # Углы поворота модели (в радианах)
  "input_path": "../data/model.obj",  # Путь к файлу модели
  "texture_path": "../data/bunny-atlas.jpg",  # Путь к текстуре
  "output_path": "../data/bunnies/bunnyFinal.png",  # Путь для сохранения результата
  "translation_offset": (0, -0.049, 1.0),  # Смещение модели
}


# Функция для поворота вершины с учетом матрицы поворота и смещения
def rotate_vertex(
  vertex: np.ndarray,
  rotation_matrix: np.ndarray,
  offset: Tuple[float, float, float] = RENDER_CONFIG["translation_offset"],
):
  return np.dot(vertex, rotation_matrix) + offset


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
def load_texture(filepath: str) -> np.ndarray:
  try:
    texture_img = Image.open(filepath)
    return np.array(ImageOps.flip(texture_img))
  except FileNotFoundError:
    raise (f"Текстура {filepath} не найдена!")


# Функция для построения модели и рендера изображения
def build_model(data: Dict[str, list]) -> np.ndarray:
  # H, W - высота и ширина изображения
  H, W = RENDER_CONFIG["image_size"]
  # Инициализация матрицы изображения и Z-буфера
  matrix = np.full((H, W, 3), (0, 0, 0), dtype=np.uint8)
  z_buff = np.full((H, W), np.inf)

  # Загрузка текстуры
  textures = load_texture(RENDER_CONFIG["texture_path"])
  # Построение матрицы поворота
  rotation_matrix = build_rotation_matrix(*RENDER_CONFIG["rotation_angles"])
  # Вычисление нормалей вершин
  vn = calculate_vertex_normals(data, rotation_matrix)

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
      H,
      W,
    )

  return matrix


if __name__ == "__main__":
  # Парсинг модели
  data = obj_parser(RENDER_CONFIG["input_path"])

  # Построение модели и сохранение изображения
  image_matrix = build_model(data)
  save_image(image_matrix, RENDER_CONFIG["output_path"])
