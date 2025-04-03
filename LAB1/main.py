from Task6 import build_model
from Task5 import obj_vf_parser
from Task2 import save_image

if __name__ == '__main__':
    vf_dict = obj_vf_parser('model.obj')
    matrix = build_model(vf_dict)
    save_image(matrix, 'bunnies/bunny0.png')