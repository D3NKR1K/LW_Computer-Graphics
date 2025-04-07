def obj_parser(filename: str) -> dict:
  try:
    with open(filename) as file:
      lines = file.readlines()
  except FileNotFoundError:
    print(f"Ошибка: файл {filename} не найден.")
    return {}
  except Exception as e:
    print(f"Ошибка при обработке файла: {e}")
    return {}

  data = {"v": [], "vt": [], "vn": [], "f": []}

  for line in lines:
    parts = line.split()
    prefix = parts[0]

    if prefix in ("v", "vt", "vn"):
      coords = list(map(float, parts[1 : 3 if prefix == "vt" else 4]))
      data[prefix].append(tuple(coords))

    elif prefix == "f":
      f_numbers = []
      for part in parts[1:]:
        values = part.split("/")
        vertex_index = int(values[0])
        texture_index = int(values[1]) if len(values) > 1 and values[1] else None
        normal_index = int(values[2]) if len(values) > 2 and values[2] else None
        f_numbers.append((vertex_index, texture_index, normal_index))
      data["f"].append(tuple(f_numbers))

  return data
