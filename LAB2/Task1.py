import numpy as np
from math import *
from PIL import Image

def bary(x0, y0, x1, y1, x2, y2, x, y):
    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda2 = 1.0 - lambda0 - lambda1

    return [lambda0, lambda1, lambda2]

def draw_tr(x0, y0, z0, x1, y1, z1, x2, y2, z2, n0, n1, n2, t1, t2, t3, t_array, h, w, img_matrix, z_buffer):
    l = [0, 0, 1]
    I0 = (np.dot(n0, l)) / (np.linalg.norm(n0) * np.linalg.norm(l))
    I1 = (np.dot(n1, l)) / (np.linalg.norm(n1) * np.linalg.norm(l))
    I2 = (np.dot(n2, l)) / (np.linalg.norm(n2) * np.linalg.norm(l))

    x0_p, x1_p, x2_p = [5000 * i[0]/i[1] + w / 2 for i in [(x0, z0), (x1, z1), (x2, z2)]]
    y0_p, y1_p, y2_p = [5000 * i[0]/i[1] + h / 2 for i in [(y0, z0), (y1, z1), (y2, z2)]]

    x_min, x_max = int(min(x0_p, x1_p, x2_p)), int(max(x0_p, x1_p, x2_p) + 1)
    if x_min < 0: x_min = 0
    if x_max > img_matrix.shape[0]: x_max = img_matrix.shape[1]

    y_min, y_max = int(min(y0_p, y1_p, y2_p)), int(max(y0_p, y1_p, y2_p) + 1)
    if y_min < 0: y_min = 0
    if y_max > img_matrix.shape[1]: y_max = img_matrix.shape[0]

    for i in range(x_min, x_max):
        for j in range(y_min, y_max):
            l0, l1, l2 = bary(x0_p, y0_p, x1_p, y1_p, x2_p, y2_p, i, j)
            if l0 >= 0 and l1 >= 0 and l2 >= 0:
                z = l0 * z0 + l1 * z1 + l2 * z2
                if z < z_buffer[j, i]:
                    if (l0 * I0 + l1 * I1 + l2 * I2) <= 0:
                        t_i = int(1024*(l0*t1[1]+l1*t2[1]+l2*t3[1]))
                        t_j = int(1024*(l0*t1[0]+l1*t2[0]+l2*t3[0]))
                        color = t_array[t_i][t_j]
                        img_matrix[j, i] = -(l0 * I0 + l1 * I1 + l2 * I2) * color
                        z_buffer[j, i] = z