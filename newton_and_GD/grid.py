import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Lennard-Jones potential function
def lj_potential(r, epsilon=1.0, sigma=1.0):
    with np.errstate(divide='ignore'):
        return 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)

# Fixed atom at the origin
atom1 = np.array([0.0, 0.0])

# Define the grid for the moving atom
x = np.linspace(-1.5, 1.5, 200)
y = np.linspace(-1.5, 1.5, 200)
X, Y = np.meshgrid(x, y)

# Distance from the moving atom to the fixed atom at the origin
R = np.sqrt(X**2 + Y**2)

# Compute the Lennard-Jones potential
V = lj_potential(R)

# Clip extreme values to avoid singularity at r=0
V = np.clip(V, -2, 2)

# Plotting the 3D surface
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, V, cmap='plasma', edgecolor='none', linewidth=0, antialiased=False)

# Mark the fixed atom at the origin
ax.scatter(atom1[0], atom1[1], lj_potential(1.0), color='white', s=100, label='Fixed Atom at Origin')

# Labels and colorbar
ax.set_title('3D Lennard-Jones Potential: One Fixed Atom at Origin')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Potential Energy')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='LJ Potential')

plt.legend()
plt.show()
plt.savefig("plot.png")