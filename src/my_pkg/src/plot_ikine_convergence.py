import numpy as np
import matplotlib.pyplot as plt
from robotfunctions import fkine, ikine_limited, jacobian_position

def evaluar_convergencia(xdes_list, q0, max_iter=1500):
    errores_todos = []

    for xdes in xdes_list:
        errores = []
        q = q0.copy()
        for _ in range(max_iter):
            # FK actual
            T = fkine(q)
            x = T[0:3, 3]
            e = xdes - x
            errores.append(np.linalg.norm(e))

            # Jacobiano
            J = jacobian_position(q)
            dq = 0.1 * np.dot(np.linalg.pinv(J), e)
            q += dq
            q[0] = np.clip(q[0], -0.9, 0.9)

        if len(errores) < max_iter:
            errores += [errores[-1]] * (max_iter - len(errores))

        
        errores_todos.append((xdes, errores))

    return errores_todos


if __name__ == "__main__":
    # Configuración inicial
    q0 = np.zeros(7)

    # Varias posiciones deseadas
    xdes_list = [
        np.array([1.5, 0.4, 0.7]),
        np.array([1.2, -0.3, 0.5]),
        np.array([1.0, 0.0, 0.4]),
        np.array([1.6, 0.2, 0.9]),   # más lejana
        np.array([1.8, -0.5, 1.0]),  # fuera del workspace posiblemente
    ]

    resultados = evaluar_convergencia(xdes_list, q0)

    # Graficar
    plt.figure(figsize=(10, 6))
    for xdes, errores in resultados:
        label = f"xdes: [{xdes[0]:.2f}, {xdes[1]:.2f}, {xdes[2]:.2f}]"
        plt.plot(errores, label=label)

    plt.title("Convergencia del método de IK por Jacobiano numérico")
    plt.xlabel("Iteración")
    plt.ylabel("Error ‖x_des - x_actual‖ [m]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
