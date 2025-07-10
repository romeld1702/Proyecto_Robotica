#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import JointState
import numpy as np
from numpy.linalg import pinv
from markers import *
from robotfunctions import *


if __name__ == '__main__':

    # Initialize the node
    rospy.init_node("testKinematicControlPosition")
    print('starting motion ... ')

    # Publisher: publish to the joint_states topic
    pub = rospy.Publisher('joint_states', JointState, queue_size=1)

    # Files for the logs
    fxcurrent = open("xcurrent.txt", "w")
    fxdesired = open("xdesired.txt", "w")
    fq = open("q.txt", "w")

    # Markers for the current and desired positions
    bmarker_current  = BallMarker(color['RED'])
    bmarker_desired  = BallMarker(color['GREEN'])

    # Joint names
    jnames = [
    'prismatic_joint_rail',
    'joint_1', 'joint_2', 'joint_3',
    'joint_4', 'joint_5', 'joint_6'
    ]


    # Desired position
    xd = np.array([-0.45, -0.2, 0.6])

    # Initial configuration
    q0 = np.array([0, 0, 0, 0, 0, 0, 0])

    # JointState message
    jstate = JointState()
    jstate.name = jnames

    # Force robot to initial configuration
    rate = rospy.Rate(50)
    for _ in range(100):
        jstate.header.stamp = rospy.Time.now()
        jstate.position = q0
        pub.publish(jstate)
        rate.sleep()

    rospy.sleep(1.5)

    # Evaluate initial pose after sync
    print("----- VERIFICACIÓN INICIAL -----")
    print("q0 =", q0)
    T = fkine(q0)
    x0 = T[0:3, 3]
    print("fkine(q0) → x0 =", x0)
    print("xd =", xd)
    print("----------------------------------")

    # Set initial position
    bmarker_current.xyz(x0)
    bmarker_desired.xyz(xd)

    # Control parameters
    freq = 50
    dt = 1.0 / freq
    rate = rospy.Rate(freq)

    # Initial joint configuration
    q = q0.copy()

    # Main loop
    while not rospy.is_shutdown():
        print("q (actual) =", q)
        # Update joint state
        jstate = JointState()
        jstate.name = jnames
        jstate.header.stamp = rospy.Time.now()
        jstate.position = q

        # Compute FK and error
        T = fkine(q)
        x = T[0:3, 3]
        e = xd - x

        # Compute Jacobian and joint velocity
        J = jacobian_position(q)
        k = 0.1
        dq = np.dot(pinv(J), k * e)
        dq = np.clip(dq, -0.2, 0.2)  # Limitar velocidad articular
        # Alert if first error is too large
        if np.linalg.norm(e) > 1.0:
            print("[ADVERTENCIA] Error inicial alto (>1m):", np.linalg.norm(e))
            print("  x =", x)
            print("  xd =", xd)
            print("  e =", e)
            print("  J =\n", J)
            input("Presiona ENTER para continuar...")

        # Integrate joint velocities
        q = q + dq * dt

        # Logs
        fxcurrent.write(f"{x[0]} {x[1]} {x[2]}\n")
        fxdesired.write(f"{xd[0]} {xd[1]} {xd[2]}\n")
        fq.write(" ".join([str(val) for val in q]) + "\n")

        # Publish joint state and update markers
        pub.publish(jstate)
        bmarker_desired.xyz(xd)
        bmarker_current.xyz(x)

        # Exit condition
        if np.linalg.norm(e) < 1e-05:
            print("Objetivo alcanzado")
            print("Posición deseada (xd):", xd)
            print("Posición actual (x):", x)
            break

        rate.sleep()

    print('ending motion ...')
    fxcurrent.close()
    fxdesired.close()
    fq.close()
