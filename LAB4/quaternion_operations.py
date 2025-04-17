import numpy as np

def q_mult(p: np.array, q: np.array) -> np.array:
    return np.array([p[0]*q[0]-p[1]*q[1]-p[2]*q[2]-p[3]*q[3], p[0]*q[1]+q[0]*p[1]+p[2]*q[3]-p[3]*q[2], p[0]*q[2]-p[1]*q[3]+p[2]*q[0]+p[3]*q[1], p[0]*q[3]+p[1]*q[2]-p[2]*q[1]+p[3]*q[0]])

def q_conj(p):
    return np.array([p[0], -p[1], -p[2], -p[3]])

def q_from_axis_angle(axis, angle):
    axis = np.array(axis) / np.linalg.norm(axis)
    w = np.cos(angle / 2)
    x, y, z = np.sin(angle / 2) * axis
    return np.array([w, x, y, z])

def rotate_vector_quaternion(q, p):
    p = np.array([0, *p])

    q_conjj = q_conj(q)
    qp = q_mult(q, p)
    qpq = q_mult(qp, q_conjj)

    return qpq[1:]