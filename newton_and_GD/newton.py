import jax
import jax.numpy as jnp
from jax import grad
from jax.scipy.linalg import solve
import numpy as np
import time
def get_energy(x_flat):
    x = x_flat.reshape(-1, 3)  # Reshape to (N-1, 3)
    x = jnp.vstack([jnp.zeros((1, 3)), x])  # Add fixed atom at origin
    N = x.shape[0]
    energy = 0.0
    for i in range(N):
        for j in range(i+1, N):
            r = jnp.linalg.norm(x[i] - x[j])
            energy += (1 / r**12) - (2 / r**6)
    return energy

def newton_method(f, x0, tol=1e-6, max_iter=100,reg=1e-6):
    grad_f = grad(f)
    hess_f = jax.hessian(f)
    x = x0.copy()
    for i in range(max_iter):
        g = grad_f(x)
        H = hess_f(x)
        
        # Compute Newton step direction: solve H dx = g
        H_reg = H + reg * jnp.eye(H.shape[0])
        try:
            dx = solve(H_reg, g)
        except np.linalg.LinAlgError:
            print("Singular Hessian, cannot solve.")
            return x
        
        # Determine descent direction (Newton step is dx, direction is -dx)
        direction = -dx
        
        # Line search parameters
        alpha = 1.0
        f_current = f(x)
        sufficient_decrease = False
        
        # Armijo line search
        while True:
            x_new = x + alpha * direction
            f_new = f(x_new)
            # Armijo condition: f_new <= f_current + c*alpha*g.dot(direction)
            if f_new <= f_current + 1e-4 * alpha * jnp.dot(g, direction):
                sufficient_decrease = True
                break
            alpha *= 0.5
        
        if not sufficient_decrease:
            print("Line search failed to find sufficient decrease.")
            return x
        
        # Update position and check convergence
        step = x_new - x
        x = x_new
        
        step_norm = jnp.linalg.norm(step)
        
        if step_norm < tol:
            print(f"Converged in {i+1} iterations.")
            return x
    
    print("Did not converge within max iterations.")
    return x

# Example usage
# Equilateral triangle initialization
key = jax.random.PRNGKey(int(time.time()))
N = 2
init_positions = jax.random.uniform(key, (N - 1, 3), minval=-0.5, maxval=0.5)
x0 = init_positions.flatten()
 # Initial position for the movable atom
print(init_positions)
x0 = init_positions.flatten()

result = newton_method(get_energy, x0)
final_positions = jnp.vstack([jnp.zeros((1, 3)), result.reshape(-1, 3)])
print("Final atom positions:\n", final_positions)
print("Final energy:", get_energy(result))