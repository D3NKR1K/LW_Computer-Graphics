def obj_parser(filename: str) -> dict:
    try:
        with open(filename) as file:
            lines = file.readlines()
    except FileNotFoundError:
        raise (f"Ошибка: файл {filename} не найден.")
    except Exception as e:
        raise (f"Ошибка при обработке файла: {e}")

    data = {"v": [], "vt": [], "vn": [], "f": []}

    for line in lines:
        parts = line.split()
        if len(parts) == 0:
            continue
        prefix = parts[0]

        if prefix in ("v", "vt", "vn"):
            coords = list(map(float, parts[1 : 3 if prefix == "vt" else 4]))
            data[prefix].append(tuple(coords))

        elif prefix == "f":
            vertices = parts[1:]

            def parse_vertex(part: str) -> tuple:
                values = part.split("/")
                vertex_index = int(values[0]) if values[0] else None
                texture_index = (
                    int(values[1]) if len(values) > 1 and values[1] else None
                )
                normal_index = int(values[2]) if len(values) > 2 and values[2] else None
                return (vertex_index, texture_index, normal_index)

            if len(vertices) > 3:
                v0 = parse_vertex(vertices[0])
                for i in range(1, len(vertices) - 1):
                    v1 = parse_vertex(vertices[i])
                    v2 = parse_vertex(vertices[i + 1])
                    data["f"].append((v0, v1, v2))
                continue

            # Обработка треугольника
            f_numbers = [parse_vertex(v) for v in vertices]
            data["f"].append(tuple(f_numbers))

    return data
