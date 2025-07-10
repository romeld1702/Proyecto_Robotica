import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # NECESARIO para activar '3d'

# Cargar los datos
xcurrent = np.loadtxt('xcurrent.txt')
xdesired = np.loadtxt('xdesired.txt')

# --- GRAFICA 3D ---
fig1 = plt.figure(1)
ax = fig1.add_subplot(111, projection='3d')
ax.plot(xdesired[:,0], xdesired[:,1], xdesired[:,2], 'g-', label='Deseada')
ax.plot(xcurrent[:,0], xcurrent[:,1], xcurrent[:,2], 'r-', label='Actual')
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
ax.legend()
ax.set_title('Trayectoria en el espacio 3D')

# --- GRAFICA 2D ---
fig2 = plt.figure(2)
t = np.arange(xcurrent.shape[0])
plt.subplot(3,1,1)
plt.plot(t, xdesired[:,0], 'g-', label='X deseada')
plt.plot(t, xcurrent[:,0], 'r-', label='X actual')
plt.ylabel('X [m]')
plt.legend()

plt.subplot(3,1,2)
plt.plot(t, xdesired[:,1], 'g-', label='Y deseada')
plt.plot(t, xcurrent[:,1], 'r-', label='Y actual')
plt.ylabel('Y [m]')
plt.legend()

plt.subplot(3,1,3)
plt.plot(t, xdesired[:,2], 'g-', label='Z deseada')
plt.plot(t, xcurrent[:,2], 'r-', label='Z actual')
plt.ylabel('Z [m]')
plt.xlabel('Muestras')
plt.legend()

plt.suptitle('Trayectoria en X, Y, Z vs tiempo')

plt.show()
