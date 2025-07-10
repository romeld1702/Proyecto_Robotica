import numpy as np
import rospy

import PyKDL as kdl
from urdf_parser_py.urdf import URDF
from kdl_parser_py.urdf import treeFromUrdfModel
from copy import copy
from pyquaternion import Quaternion

from numpy import sin, cos, pi


# Cargar el modelo URDF generado
robot = URDF.from_xml_file("/home/romel/gen3_ws/src/ros_kortex/kortex_description/arms/gen3/6dof/urdf/gen3_on_rail.urdf")
#print("Link raíz:", robot.get_root())

ok, tree = treeFromUrdfModel(robot)



# Crear cadena entre riel y efector final
chain = tree.getChain("base_fixed", "end_effector_link")



def dh(d, theta, a, alpha):
    ct = cos(theta)
    st = sin(theta)
    ca = cos(alpha)
    sa = sin(alpha)
    return np.array([
        [ct, -st*ca, st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,    sa,    ca,    d],
        [0,     0,     0,    1]
    ])

def fkine(q):
    q0, q1, q2, q3, q4, q5, q6 = q

    T1 = dh(q0,         0,             0.048803,         0)
    T2 = dh(0,   3*pi/2,             0.15643 + 0.12838,  q1)
    T3 = dh(0.41, pi,               -0.00538,            q2 + 3*pi/2)
    T4 = dh(0,    pi,               -0.00638,            q3 + 3*pi/2)
    T5 = dh(0,   pi/2,              -(0.20843 + 0.10593), q4 + pi)
    T6 = dh(0,   pi/2,              0,                   q5 + pi)
    T7 = dh(0,    pi,               -(0.10593 + 0.06153), q6 + pi)

    T = T1 @ T2 @ T3 @ T4 @ T5 @ T6 @ T7
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

def ikine(xdes, q0):
    """
    IK con orientación canónica y posición usando PyKDL LMA
    """
    # Crear solucionador
    solver = kdl.ChainIkSolverPos_LMA(chain)

    # Inicial
    q_init = kdl.JntArray(chain.getNrOfJoints())
    for i in range(chain.getNrOfJoints()):
        q_init[i] = q0[i]

    # Pose deseada: solo posición, orientación canónica
    R = kdl.Rotation.Identity()
    p = kdl.Vector(xdes[0], xdes[1], xdes[2])
    frame_des = kdl.Frame(R, p)

    # Salida
    q_out = kdl.JntArray(chain.getNrOfJoints())
    success = solver.CartToJnt(q_init, frame_des, q_out)

    if success < 0:
        rospy.logwarn("IK no encontró solución")
        return q0  # o retornar None

    # Convertir a numpy
    return np.array([q_out[i] for i in range(chain.getNrOfJoints())])




def ikine_limited(xdes, q0, qmin=-0.9, qmax=0.9, tol=1e-4, max_iter=1000000, alpha=0.1):
    """
    IK iterativo limitado solo en q[0] (riel prismático)
    xdes: posición deseada (3,)
    q0: configuración inicial (7,)
    qmin, qmax: límites de q[0]
    """
    q = q0.copy().astype(float)

    for _ in range(max_iter):
        # FK y error de posición
        T = fkine(q)
        x = T[0:3, 3]
        e = xdes - x

        if np.linalg.norm(e) < tol:
            break

        # Jacobiano de posición
        J = jacobian_position(q)

        # Pseudo-inversa del Jacobiano
        dq = alpha * np.dot(np.linalg.pinv(J), e)

        q += dq

        # Aplicar límites SOLO a q[0]
        q[0] = np.clip(q[0], qmin, qmax)

    return q


def jacobian_geometric(q, delta=1e-5):
    """
    Jacobiano geométrico completo (posición + orientación) por diferencias finitas.
    Devuelve una matriz de 6x7: 3 filas para velocidad lineal y 3 para velocidad angular.
    """
    n = q.size
    J = np.zeros((6, n))

    # FK en la configuración actual
    T = fkine(q)
    R = T[0:3, 0:3]
    x = T[0:3, 3]

    for i in range(n):
        dq = np.copy(q)
        dq[i] += delta

        T_delta = fkine(dq)
        R_delta = T_delta[0:3, 0:3]
        x_delta = T_delta[0:3, 3]

        # Velocidad lineal (posición)
        J[0:3, i] = (x_delta - x) / delta

        # Velocidad angular (rotación)
        R_err = R_delta @ R.T
        dtheta = 0.5 * np.array([
            R_err[2,1] - R_err[1,2],
            R_err[0,2] - R_err[2,0],
            R_err[1,0] - R_err[0,1]
        ]) / delta

        J[3:6, i] = dtheta

    return J
