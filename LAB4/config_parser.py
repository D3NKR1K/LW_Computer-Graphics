import math

def config_parser(file_path):
    config_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue  # Пропускаем пустые строки

            # Разделяем ключ и значение
            key_part, value_part = line.split(':', 1)
            key = key_part.strip().strip('"')
            value_str = value_part.strip()

            # Обработка значений
            if value_str.startswith('(') and value_str.endswith(')'):
                # Парсим кортеж с заменой pi на math.pi
                try:
                    value = eval(
                        value_str,
                        {"__builtins__": None},
                        {"pi": math.pi}
                    )
                except:
                    value = value_str
            else:
                # Пытаемся определить тип значения
                try:
                    value = int(value_str)
                except ValueError:
                    try:
                        value = float(value_str)
                    except ValueError:
                        # Убираем кавычки у строк
                        if (value_str.startswith('"') and value_str.endswith('"')) or \
                                (value_str.startswith("'") and value_str.endswith("'")):
                            value = value_str[1:-1]
                        else:
                            value = value_str

            config_dict[key] = value

    return config_dict