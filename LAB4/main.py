import numpy as np
import keyboard
from obj_parser import obj_parser
from config_parser import config_parser
from save_image import save_image
from draw_image import draw_triangle
from math import *
from quaternion_operations import *
from PIL import Image, ImageOps
from typing import Dict

IMAGE_SIZE = (1000, 1000)

# Функция для построения матрицы поворота по Эйлеру
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
  data: Dict[str, list], rotate_code
) -> np.ndarray:
    vn = np.zeros((len(data["v"]), 3), dtype=np.float32)

  quaternions = np.tile(np.array([1, 0, 0, 0]), (len(data["v"]), 1))

  if rotate_code != 0:
    q_rot = q_from_axis_angle(RENDER_CONFIG["axis_for_quater"], RENDER_CONFIG["rotation_angle_for_quater"])
    quaternions = np.tile(q_rot, (len(data["v"]), 1))

  for face in data["f"]:
    v_indices = [vi[0] - 1 for vi in face]
    vertices = np.array([data["v"][i] for i in v_indices])

    if rotate_code == 0:
      rotated_vertices = np.dot(vertices, rotation_matrix)
    else:
      rotated_vertices = np.array([
        rotate_vector_quaternion(quaternions[i], v)
        for i, v in zip(v_indices, vertices)
      ]) + RENDER_CONFIG["translation_offset"]

        edge1 = rotated_vertices[1] - rotated_vertices[0]
        edge2 = rotated_vertices[2] - rotated_vertices[0]
        face_normal = np.cross(edge2, edge1)

        for i in v_indices:
            vn[i] += face_normal

  norms = np.linalg.norm(vn, axis=1)
  return vn / norms[:, np.newaxis], quaternions


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
def build_model(data: Dict[str, list], matrix, z_buff, textures, rotate_code) -> np.ndarray:
  # Вычисление нормалей вершин
  vn, quaternions = calculate_vertex_normals(data, rotate_code)
  # Scale
  scale = RENDER_CONFIG["scale"]

  # Обработка каждой грани модели
  for face in data["f"]:
    # Индексы вершин грани
    v_indices = [vi[0] - 1 for vi in face]
    # Координаты вершин
    vertices = np.array([data["v"][i] for i in v_indices])

    if rotate_code == 0:
      # Поворот и смещение вершин
      rotated_vertices = (
              np.dot(vertices, rotation_matrix) + RENDER_CONFIG["translation_offset"]
      )
    else:
      rotated_vertices = np.array([
        rotate_vector_quaternion(quaternions[i], v)
        for i, v in zip(v_indices, vertices)
      ]) + RENDER_CONFIG["translation_offset"]

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

  print("F10 - обработка с поворотом по Эйлеру\n"
        "F11 - обработка с поворотом по кватернионам\n"
        "F12 - сохранение изображния\n")

  while True:
    if keyboard.is_pressed('F10'):
      print("Обработка модели начата!")
            RENDER_CONFIG = config_parser("../data/config.txt")

            # Загрузка текстуры
            textures = load_texture(RENDER_CONFIG["texture_path"])

      # Построение матрицы поворота
      rotation_matrix = build_rotation_matrix(*RENDER_CONFIG["rotation_angles_for_eiler"])

            # Парсинг модели
            data = obj_parser(RENDER_CONFIG["input_path"])

      image_matrix = build_model(data, matrix, z_buff, textures, 0)

            print("Изменения модели сохранены!")

    if keyboard.is_pressed('F11'):
      print("Обработка модели начата!")

      RENDER_CONFIG = config_parser("../data/config.txt")

      # Загрузка текстуры
      textures = load_texture(RENDER_CONFIG["texture_path"])

      # Парсинг модели
      data = obj_parser(RENDER_CONFIG["input_path"])

      # Построение модели
      image_matrix = build_model(data, matrix, z_buff, textures, 1)

      print("Изменения модели сохранены!")

    if keyboard.is_pressed('F12'):
      # Сохранение изображения
      save_image(image_matrix, RENDER_CONFIG["output_path"])
      print("Модель сохранена в виде изображения!")
      break
