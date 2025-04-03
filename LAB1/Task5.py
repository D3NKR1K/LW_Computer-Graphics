from numpy.ma.core import append


def obj_parser(filename: str) -> dict:
    """Парсер строк, содержащих информацию о вершинах объекта и о ребрах объекта в файле типа OBJ"""

    lines = []  # Массив для хранения строк из файла

    try:
        # Открываем и построчно читаем файл
        with open(filename) as file:
            lines = file.readlines()

    except FileNotFoundError:
        print(f"Ошибка: файл {filename} не найден.")
    except Exception as e:
        print(f"Ошибка при обработке файла: {e}")

    parser_dict = dict(v = [], vt = [], vn = [], f = [])

    for line in lines:
        if line.startswith('v '):
            # Разбиваем строку по пробелам на подстроки
            parts = line.strip().split()
            # Приводим строки к вещественным числам и заносим в массив
            v_numbers = [float(parts[1]), float(parts[2]), float(parts[3])]

            parser_dict['v'].append(v_numbers)

        if line.startswith('vt '):
            parts = line.strip().split()
            vt_numbers = [float(parts[1]), float(parts[2])]

            parser_dict['vt'].append(vt_numbers)

        if line.startswith('vn '):
            parts = line.strip().split()
            vn_numbers = [float(parts[1]), float(parts[2]), float(parts[3])]

            parser_dict['vn'].append(vn_numbers)

        if line.startswith('f '):
            parts = line.strip().split() # Разбиваем строку по пробелам на подстроки
            f_numbers = []

            for part in parts[1:]: # Пропускаем первую часть 'f'
                f_v_value = part.split('/')[0] # Делим части по '/' и выделяем только 1-ю часть
                f_vt_value = part.split('/')[1]
                f_vn_value = part.split('/')[2]
                f_numbers.append(int(f_v_value)) # Приводим строку к целому числу и заносим в массив
                f_numbers.append(int(f_vt_value))
                f_numbers.append(int(f_vn_value))

            parser_dict['f'].append(f_numbers)

    return parser_dict