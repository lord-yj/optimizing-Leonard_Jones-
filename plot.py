import numpy as np
import matplotlib.pyplot as plt

# Define a symmetric matrix A and vector b
A = np.array([[2, 1], [1, 3]])
b = np.array([1, 2])

# Define the function f(x)
def f(x, y):
    vec = np.array([x, y])
    return 12 * vec @ A @ vec - b @ vec

# Create a grid of x and y values
x_vals = np.linspace(-3, 3, 200)
y_vals = np.linspace(-3, 3, 200)
X, Y = np.meshgrid(x_vals, y_vals)

# Evaluate f over the grid
Z = np.array([[f(x, y) for x in x_vals] for y in y_vals])

# Plot the contour
plt.figure(figsize=(8, 6))
cp = plt.contourf(X, Y, Z, levels=50, cmap='viridis')
plt.colorbar(cp)
plt.title(r'$f(x) = 12x^T A x - b^T x$')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.savefig("(0 plot)")
