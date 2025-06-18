import numpy as np

n = 5
d = -np.random.rand(n)
Q, _ = np.linalg.qr(np.random.randn(n, n))
A = Q @ np.diag(d) @ Q.T

vec = A[:, 0]
vec = vec / np.linalg.norm(vec)
res = 0.5 * vec.T @ A @ vec - vec.T @ vec
print(res)
print()
for i in range(A.shape[1]):
    vec = A[:, i]
    vec = vec / np.linalg.norm(vec)
    res = 0.5 * vec.T @ A @ vec - vec.T @ vec
    print(res)