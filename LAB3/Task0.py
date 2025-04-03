import numpy as np
from math import *
from LAB1.Task1 import save_image
from LAB1.Task5 import obj_vf_parser
from LAB2.Task1 import draw_tr

def build_model(vf, h, w):
    matrix = np.full((h, w, 3), (0, 0, 0), dtype=np.uint8)
    z_buff = np.full((h, w), np.inf)

    # Поворот
    alpha = 0
    beta = 0
    gamma = 0

    Rx = np.array([[1, 0, 0], [0, cos(alpha), sin(alpha)], [0, -sin(alpha), cos(alpha)]])
    Ry = np.array([[cos(beta), 0, sin(beta)], [0, 1, 0], [-sin(beta), 0, cos(beta)]])
    Rz = np.array([[cos(gamma), sin(gamma), 0], [-sin(gamma), cos(gamma), 0], [0, 0, 1]])

    R = Rx @ Ry @ Rz

    for face in vf['f']:
        v1, v2, v3 = [vi - 1 for vi in face]

        def rotate_vertex(vertex_idx):
            t_vector = np.array([0, -0.049, 0.1]).T
            rotated = np.dot(R, vf['v'][vertex_idx]) + t_vector
            return rotated[0], rotated[1], rotated[2]

        x1, y1, z1 = rotate_vertex(v1)
        x2, y2, z2 = rotate_vertex(v2)
        x3, y3, z3 = rotate_vertex(v3)

        draw_tr(x1, y1, z1, x2, y2, z2, x3, y3, z3, h, w, matrix, z_buff)

    return np.rot90(matrix, 2)

if __name__ == '__main__':
    h, w = 1000, 1000
    vf_dict = obj_vf_parser("model_1.obj")
    matrix = build_model(vf_dict, h, w)
    save_image(matrix, 'bunny6.png')