import numpy as np
import matplotlib.pyplot as plt
from robotfunctions import fkine, jacobian_position


def evaluar_errores_por_eje_por_config(xdes_list, q0, max_iter=1500):
    """
    Retorna listas por eje: err_xs, err_ys, err_zs
    Cada elemento es una lista de errores por iteración para una configuración
    """
    err_xs, err_ys, err_zs = [], [], []

    for xdes in xdes_list:
        err_x, err_y, err_z = [], [], []
        q = q0.copy()

        for _ in range(max_iter):
            T = fkine(q)
            x = T[0:3, 3]
            e = xdes - x

            err_x.append(abs(e[0]))
            err_y.append(abs(e[1]))
            err_z.append(abs(e[2]))

            J = jacobian_position(q)
            dq = 0.1 * np.dot(np.linalg.pinv(J), e)
            q += dq
            q[0] = np.clip(q[0], -0.9, 0.9)

        for arr in [err_x, err_y, err_z]:
            if len(arr) < max_iter:
                arr += [arr[-1]] * (max_iter - len(arr))

        err_xs.append((xdes, err_x))
        err_ys.append((xdes, err_y))
        err_zs.append((xdes, err_z))

    return err_xs, err_ys, err_zs


if __name__ == "__main__":
    q0 = np.zeros(7)
    xdes_list = [
        np.array([1.5, 0.4, 0.7]),
        np.array([1.2, -0.3, 0.5]),
        np.array([1.0, 0.0, 0.4]),
        np.array([1.6, 0.2, 0.9]),   # más lejana
        np.array([1.8, -0.5, 1.0]),  # fuera del workspace posiblemente
    ]

    err_xs, err_ys, err_zs = evaluar_errores_por_eje_por_config(xdes_list, q0)

    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    for xdes, err in err_xs:
        axs[0].plot(err, label=f"xdes: [{xdes[0]:.2f}, {xdes[1]:.2f}, {xdes[2]:.2f}]")
    axs[0].set_ylabel("Error en X [m]")
    axs[0].legend()
    axs[0].grid(True)

    for xdes, err in err_ys:
        axs[1].plot(err, label=f"xdes: [{xdes[0]:.2f}, {xdes[1]:.2f}, {xdes[2]:.2f}]")
    axs[1].set_ylabel("Error en Y [m]")
    axs[1].legend()
    axs[1].grid(True)

    for xdes, err in err_zs:
        axs[2].plot(err, label=f"xdes: [{xdes[0]:.2f}, {xdes[1]:.2f}, {xdes[2]:.2f}]")
    axs[2].set_ylabel("Error en Z [m]")
    axs[2].set_xlabel("Iteración")
    axs[2].legend()
    axs[2].grid(True)

    fig.suptitle("Convergencia por componente para distintas configuraciones", fontsize=14)
    plt.tight_layout()
    plt.show()
