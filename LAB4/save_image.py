from numpy import ndarray
from PIL import Image


def save_image(img: ndarray, filename: str) -> None:
  img_pil = Image.fromarray(img)  # Генерация изображения из массива
  img_pil.save(filename)  # Сохранение изображения с указанным именем
