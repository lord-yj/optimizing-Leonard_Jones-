import numpy as np
import jax
import matplotlib.pyplot as plt
import math
from scipy import stats
from jax import grad
from jax.scipy.linalg import solve
norm = np.linalg.norm

    
def potential_and_gradient(X):
    """
    X a matrix with shape (N, 3) to encode the position of each atoms
    We return the potential and the gradient matrix
    """
    N = X.shape[0]
    V = 0
    G = np.zeros_like(X)
    for i in range(N):
        for j in range(i + 1, N):
            d = X[i] - X[j]

            r2 = np.dot(d, d)
            r = np.sqrt(r2)
            r6 = r2 ** 3
            r12 = r2**6

            potential= (1/(r12) - 2/(r6))
            V += potential
            deriv = (-12 / (r2**6.5)) + (12 / (r2**3.5))/r
            grad = deriv*d
            G[i] += grad
            G[j] -= grad 
    
    G[0] = 0.0

    return V, G

def gradient_descent_LJ(X_init, tol=1e-4, max_iters=1000,
                        alpha0=1.0, rho=0.5, c=1e-4):

    X = X_init.copy()
    history = []
    
    for k in range(max_iters):
        E, G = potential_and_gradient(X)
        grad_norm = np.linalg.norm(G)
        history.append((E, grad_norm))
        
        if grad_norm < tol:
            print(f"Converged in {k} iterations: E={E:.6f}, ||grad||={grad_norm:.2e}")
            break
        
        P = -G  # descent direction
        
        alpha = alpha0
        E0 = E
        dot_GP = np.sum(G * P)
        while True:
            X_new = X + alpha * P
            X_new[0, :] = 0.0  
            E_new, _ = potential_and_gradient(X_new)
            if E_new <= E0 + c * alpha * dot_GP:
                break
            alpha *= rho
            if alpha < 1e-12:
                raise RuntimeError("Line search failed to find a suitable step size")
        
        X = X_new
    else:
        print("Reached max iterations without converging")
    return X, history
# def gradient_descent(x, A, tol=1e-4, max_iters=1000):
#     x_old = x
#     historu 
#     for _ in range(max_iters):
#         gradient = 2*np.multiply(A, x)
#         x_curr = x_old - (1/2)*gradient
#         if gradient < tol:
#             break 
#         x_old = x_curr/np.linalg.norm(x_curr)        
#     pass 
# N = 5
# X = np.random.normal(0, 1,(N, 3))
# X[0] = 0
# print(X)
# X_new, history= gradient_descent_LJ(X)
# potential,_ = potential_and_gradient(X_new)
# print("Potential is")
# print(potential)
# print(X_new)


# # unpack history
# energies, grad_norms = zip(*history)
# iters = np.arange(len(grad_norms))
# slope, intercept, _, _, _ = stats.linregress(iters, grad_norms)
# plt.figure()
# plt.scatter(iters, grad_norms)
# plt.yscale('log')
# plt.xlabel('Iteration')
# plt.ylabel('Gradient Norm')
# plt.title('Gradient Norm Convergence [scatter]')
# plt.grid(True)
# plt.tight_layout()
# plt.savefig('(18)gradnorm_convergence_scatter.png')
# print("slope is " + str(slope))


import numpy as np
norm = np.linalg.norm
def gradient_descent(A, x0, alpha, num_iters, tol=1e-6):

    H = 2*A
    x = x0.copy()/np.linalg.norm(x0)
    history = [norm(x)]
    
    for k in range(1, num_iters + 1):
        grad = H.dot(x)
        grad_t = grad - (x.dot(grad)) * x
        grad_norm = np.linalg.norm(grad_t)
        
        
        # Check convergence
        if grad_norm < tol:
            print(f"Converged at iteration {k-1} (||grad||={grad_norm:.2e})")
            break
        
        x = x - alpha * grad_t
        x = x/np.linalg.norm(x)
        history.append(norm(x))
    
    else:
        # If we never broke, k is max
        print(f"Reached maximum iterations ({num_iters}) without full convergence.")
        k = num_iters
    
    return x, np.array(history), k

def generate_spd(n):
    # Generate eigenvalues in descending order: 1 >= λ₁ > λ₂ > ... > λₙ > 0
    eigenvalues = np.linspace(1, 0.1, n)  # Adjust 0.1 for smallest eigenvalue
    
    # Generate a random orthogonal matrix using QR decomposition
    random_matrix = np.random.randn(n, n)
    Q, _ = np.linalg.qr(random_matrix)
    
    # Construct SPD matrix: A = Q * diag(λ) * Q^T
    D = np.diag(eigenvalues)
    A = Q @ D @ Q.T
    
    # Ensure symmetry (due to numerical precision, sometimes needed)
    A = (A + A.T) / 2
    
    return A

# Example usage
n = 4
A = generate_spd(n)
x0 = np.random.randn(n)

alpha = 0.5
max_iters = 1000
tol = 1e-8

x_star, history, num_used = gradient_descent(A, x0, alpha, max_iters, tol)
history = [math.log(val) for val in history if val != 0] 

plt.figure()
plt.scatter(range(num_used), history)
plt.xlabel("Iteration")
plt.ylabel("||∇f(x_k)||")
plt.title(f"Convergence of Gradient Norm (stopped at {num_used} iterations)")
plt.show()
plt.savefig("(3).png")
