import numpy as np
from LAB1.Task1 import save_image
from LAB1.Task5 import obj_vf_parser
from LAB2.Task1 import draw_tr

def build_model(vf):
    matrix = np.full((1000, 1000, 3), (0, 0, 0), dtype=np.uint8)
    
    z_buff = np.full((1000, 1000), np.inf)
    for i in range(len(vf['f'])):
        p1, p2, p3 = [vi - 1 for vi in vf['f'][i]]

        x1 = vf['v'][p1][0]
        y1 = vf['v'][p1][1]
        z1 = vf['v'][p1][2]

        x2 = vf['v'][p2][0]
        y2 = vf['v'][p2][1]
        z2 = vf['v'][p2][2]

        x3 = vf['v'][p3][0]
        y3 = vf['v'][p3][1]
        z3 = vf['v'][p3][2]

        draw_tr(x1, y1, z1, x2, y2, z2, x3, y3, z3, matrix, z_buff)

    return np.rot90(np.rot90(matrix))

if __name__ == '__main__':
    vf_dict = obj_vf_parser("data/model.obj")
    matrix = build_model(vf_dict)
    save_image(matrix, 'data/bunnies/bunny.png')