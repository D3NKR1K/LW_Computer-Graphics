import numpy as np
from PIL import Image, ImageOps
from math import *
from LAB1.Task1 import save_image
from LAB1.Task5 import obj_parser
from LAB2.Task1 import draw_tr

def find_normal(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    n1 = (x1 - x2, y1 - y2, z1 - z2)
    n2 = (x1 - x0, y1 - y0, z1 - z0)

    return np.cross(n1, n2)


def rotate_vertex(R, vertex_idx):
    t_vector = np.array([0, -0.049, 0.1]).T
    rotated = np.dot(R, parser_dict['v'][vertex_idx]) + t_vector
    return rotated[0], rotated[1], rotated[2]

def find_vn(parser_dict, R):
    vn = np.zeros((len(parser_dict['v']), 3), dtype=float)

    for face in parser_dict['f']:
        v1 = face[0] - 1
        v2 = face[3] - 1
        v3 = face[6] - 1

        x0, y0, z0 = rotate_vertex(R, v1)
        x1, y1, z1 = rotate_vertex(R, v2)
        x2, y2, z2 = rotate_vertex(R, v3)

        n = find_normal(x0, y0, z0, x1, y1, z1, x2, y2, z2)
        vn[v1] += n
        vn[v2] += n
        vn[v3] += n

    return vn / np.linalg.norm(vn, axis=1)[:, np.newaxis]

def build_model(parser_dict, h, w):
    matrix = np.full((h, w, 3), (0, 0, 0), dtype=np.uint8)
    z_buff = np.full((h, w), np.inf)
    texture_image = Image.open("texture.jpg")
    texture_image = ImageOps.flip(texture_image)
    texture_array = np.array(texture_image)

    # Поворот
    alpha = 0
    beta = -pi/2
    gamma = 0

    Rx = np.array([[1, 0, 0], [0, cos(alpha), sin(alpha)], [0, -sin(alpha), cos(alpha)]])
    Ry = np.array([[cos(beta), 0, sin(beta)], [0, 1, 0], [-sin(beta), 0, cos(beta)]])
    Rz = np.array([[cos(gamma), sin(gamma), 0], [-sin(gamma), cos(gamma), 0], [0, 0, 1]])

    R = Rx @ Ry @ Rz

    vn = find_vn(parser_dict, R)

    for face in parser_dict['f']:
        v1 = face[0] - 1
        v2 = face[3] - 1
        v3 = face[6] - 1

        vt1 = face[1] - 1
        vt2 = face[4] - 1
        vt3 = face[7] - 1

        x1, y1, z1 = rotate_vertex(R, v1)
        x2, y2, z2 = rotate_vertex(R, v2)
        x3, y3, z3 = rotate_vertex(R, v3)

        normal1 = vn[v1]
        normal2 = vn[v2]
        normal3 = vn[v3]

        texture1 = parser_dict['vt'][vt1]
        texture2 = parser_dict['vt'][vt2]
        texture3 = parser_dict['vt'][vt3]

        draw_tr(x1, y1, z1, x2, y2, z2, x3, y3, z3, normal1, normal2, normal3, texture1, texture2, texture3, texture_array, h, w, matrix, z_buff)

    return np.rot90(matrix, 2)

if __name__ == '__main__':
    h, w = 1000, 1000
    parser_dict = obj_parser("model_1.obj")
    matrix = build_model(parser_dict, h, w)
    save_image(matrix, 'bunny2.png')