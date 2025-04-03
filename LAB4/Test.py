from LAB1.Task5 import obj_parser
from Task0 import find_vn

if __name__ == '__main__':
    dict = obj_parser("model_1.obj")

    vn = find_vn(dict)

    print(vn[0])
    print(vn[1])
    print(vn[2])


