import numpy as np
from math import cos, sin, pi
from LAB1.Task1 import save_image
from LAB1.Task5 import obj_vf_parser
from LAB3_Edit_LAB2.Task1 import draw_tr

def build_model(vf, H: int, W: int):
    matrix = np.full((H, W, 3), (0, 0, 0), dtype=np.uint8)
    z_buff = np.full((H, W), np.inf)
    
    alpha = 0
    beta = pi / 2
    gamma = 0
    
    matrixRX = np.array([
      [1, 0, 0],
      [0, cos(alpha), sin(alpha)],
      [0, -sin(alpha), cos(alpha)]
    ])	
    
    matrixRY = np.array([
        [cos(beta), 0, sin(beta)],
        [0, 1, 0],
        [-sin(beta), 0, cos(beta)]
    ])
    
    matrixRZ = np.array([
        [cos(gamma), sin(gamma), 0],
        [-sin(gamma), cos(gamma), 0],
        [0, 0, 1]
    ])	
    
    matrixR = matrixRX @ matrixRY @ matrixRZ
    vectorT = np.array([0, -0.049, 1.0]).T
    
    for face in vf['f']:
        v1, v2, v3 = [vi - 1 for vi in face]
        
        def rotate_vertex(vertex_idx):
            rotated = np.dot(vf['v'][vertex_idx], matrixR) + vectorT
            return rotated[0], rotated[1], rotated[2]

        x1, y1, z1 = rotate_vertex(v1)
        x2, y2, z2 = rotate_vertex(v2)
        x3, y3, z3 = rotate_vertex(v3)

        draw_tr(x1, y1, z1, x2, y2, z2, x3, y3, z3, matrix, z_buff, H, W)

    return np.rot90(matrix, 2)

if __name__ == '__main__':
    vf_dict = obj_vf_parser("data/model.obj")
    matrix = build_model(vf_dict, 2000, 2000)
    save_image(matrix, 'data/bunnies/bunny0.png')
