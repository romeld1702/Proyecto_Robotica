import numpy as np
import rospy

import PyKDL as kdl
from urdf_parser_py.urdf import URDF
from kdl_parser_py.urdf import treeFromUrdfModel
from copy import copy
from pyquaternion import Quaternion
cos = np.cos
sin = np.sin
pi = np.pi

# === DH Parameters
# L = [L0, L1, L2, L3, L4, L5, L6, L7] in meters
L = [0.1564, 0.1284, 0.2104, 0.2104, 0.2084, 0.0064, 0.1059, 0.0615]

# Table DH: [d_i, theta_i, a_i, alpha_i]
def dh(d, theta, a, alpha):
    ct = cos(theta)
    st = sin(theta)
    ca = cos(alpha)
    sa = sin(alpha)

    T = np.array([[ct, -st * ca, st * sa, a * ct],
                  [st, ct * ca, -ct * sa, a * st],
                  [0, sa, ca, d],
                  [0, 0, 0, 1]])
    return T


def fkine(q):
    q1 = q[0]
    q2 = q[1]
    q3 = q[2]
    q4 = q[3]
    q5 = q[4]
    q6 = q[5]


    # --- Matriz del riel (traslación en X) ---
    T0 = np.eye(4)
    T0[0:3, 3] = np.array([0.030, -0.423, 0.0])  # el desplazamiento fijo
    T1 = dh(0.15643+0.12838,     q1,         0,    3*np.pi/2)
    T2 = dh(-0.00538,  q2 + 3*np.pi/2,    0.41,       np.pi)
    T3 = dh(-0.00638, q3 + 3*np.pi/2,      0,        np.pi/2)
    T4 = dh(-(0.20843+0.10593),   q4 + np.pi,   0,    np.pi/2)
    T5 = dh(0,    q5 + np.pi,         0,            np.pi/2)
    T6 = dh(-(0.10593 + 0.06153),   q6 + np.pi, 0,     np.pi)

    # --- Transformación total ---
    T = T0 @ T1 @ T2 @ T3 @ T4 @ T5 @ T6

    return T


def TF2xyzquat(T):
    """
    Convierte matriz homogénea 4x4 a [x, y, z, qx, qy, qz, qw]
    """
    pos = T[0:3, 3]
    rot = T[0:3, 0:3]
    quat = Quaternion(matrix=rot)
    return list(pos) + [quat.x, quat.y, quat.z, quat.w]


def jacobian_position(q, delta=0.0001):
    """
    Jacobiano de posición por diferencias finitas.
    """
    n = q.size
    J = np.zeros((3, n))
    
    # FK en la configuración actual
    T = fkine(q)
    x = T[0:3, 3]

    # Por cada articulación
    for i in range(n):
        dq = np.copy(q)
        dq[i] += delta
        T_delta = fkine(dq)
        x_delta = T_delta[0:3, 3]

        J[:, i] = (x_delta - x) / delta

    return J


def ikine(xdes, q0, epsilon=1e-3, max_iter=100, k=0.5):
    """
    Resuelve la cinemática inversa usando método numérico (pseudoinversa del Jacobiano).
    
    Parámetros:
    - xdes: vector objetivo (posición deseada del efector final)
    - q0: configuración articular inicial
    - epsilon: tolerancia mínima para el error
    - max_iter: número máximo de iteraciones
    - k: ganancia del controlador
    
    Retorna:
    - q: configuración articular que aproxima la posición deseada
    """
    q = q0.copy()
    
    for i in range(max_iter):
        T = fkine(q)                    # Cinemática directa
        x = T[0:3, 3]                   # Posición actual
        e = xdes - x                    # Error de posición
        
        if np.linalg.norm(e) < epsilon:
            return q
        
        J = jacobian_position(q)       # Jacobiano de posición
        qdot = k * np.dot(np.linalg.pinv(J), e)  # Velocidad articular
        q = q + qdot                   # Integración por Euler
    
    rospy.logwarn("IK no encontró solución")
    return q
