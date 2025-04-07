from LAB1.obj_parser import obj_vf_parser
from Task2 import bresenham_line
from Task1 import save_image
from numpy import ndarray, zeros, uint8, rot90

def build_model(vf: dict) -> ndarray:
    matrix = zeros((1000, 1000), dtype = uint8)
    for i in range(len(vf['f'])):
        p1, p2, p3 = [vi - 1 for vi in vf['f'][i]]

        x1 = round(9900 * vf['v'][p1][0] + 500)
        y1 = round(9900 * vf['v'][p1][1] + 5)

        x2 = round(9900 * vf['v'][p2][0] + 500)
        y2 = round(9900 * vf['v'][p2][1] + 5)

        x3 = round(9900 * vf['v'][p3][0] + 500)
        y3 = round(9900 * vf['v'][p3][1] + 5)

        bresenham_line(matrix, x1, y1, x2, y2, 255)
        bresenham_line(matrix, x2, y2, x3, y3, 255)
        bresenham_line(matrix, x3, y3, x1, y1, 255)

    return rot90(matrix)

if __name__ == '__main__':
    vf_dict = obj_vf_parser("model.obj")
    matrix = build_model(vf_dict)
    save_image(matrix, 'bunnies/poly_bunny.png')
