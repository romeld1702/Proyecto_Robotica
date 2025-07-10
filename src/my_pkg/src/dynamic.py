import PyKDL
from urdf_parser_py.urdf import URDF
from kdl_parser_py.urdf import treeFromUrdfModel
import numpy as np

if _name_ == '_main_':

    # Lectura del modelo del robot Gen3 6DOF + riel
    robot = URDF.from_xml_file('/home/romel/gen3_ws/src/ros_kortex/kortex_description/arms/gen3/6dof/urdf/gen3_on_rail.urdf')
    ok, tree = treeFromUrdfModel(robot)
    chain = tree.getChain('base_fixed', 'end_effector_link')

    ndof = chain.getNrOfJoints()

    # Configuración articular inicial (ajustar según rango de cada articulación si es necesario)
    q0 = np.array([-0.8,-np.pi/2,-np.pi/4,np.pi/2,0,-np.pi/4,0])
    dq0 = np.zeros(ndof)
    ddq0 = np.zeros(ndof)

    q = PyKDL.JntArray(ndof)
    dq = PyKDL.JntArray(ndof)
    ddq = PyKDL.JntArray(ndof)

    for i in range(ndof):
        q[i] = q0[i]
        dq[i] = dq0[i]
        ddq[i] = ddq0[i]

    zeros = PyKDL.JntArray(ndof)
    tau = PyKDL.JntArray(ndof)
    g = PyKDL.JntArray(ndof)
    c = PyKDL.JntArray(ndof)
    M = PyKDL.JntSpaceInertiaMatrix(ndof)

    gravity = PyKDL.Vector(0.0, 0.0, -9.81)

    dyn_solver = PyKDL.ChainDynParam(chain, gravity)

    dyn_solver.JntToGravity(q, g)
    dyn_solver.JntToCoriolis(q, dq, c)
    dyn_solver.JntToMass(q, M)

    g_list = [round(g[i], 3) for i in range(ndof)]
    c_list = [round(c[i], 3) for i in range(ndof)]
    M_list = [[round(M[i, j], 3) for j in range(ndof)] for i in range(ndof)]

    print('g(q) =', g_list)
    print('c(q, dq) =', c_list)
    print('M(q):')
    for row in M_list:
        print(row)

    # Calcular tau = M*ddq + c + g
    for i in range(ndof):
        tau[i] = c[i] + g[i]
        for j in range(ndof):
            tau[i] += M[i, j] * ddq[j]

    tau_list = [round(tau[i], 3) for i in range(ndof)]
    print('tau =', tau_list)
