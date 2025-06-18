import numpy as np
import time
import matplotlib.pyplot as plt
from math import *
def make_H_matrix(n):
    # build the list of allowed values
    A = np.random.normal(0, 1, (n, n))
    r, c = A.shape
    for i in range(r):
        for j in range(c):
            if i > j + 1:
                A[i][j] = 0
    return A

def modified_H_matrix(n):
    H = np.random.normal(0, 1, (n, n))
    r, c = H.shape
    for i in range(r):
        for j in range(c):
            if i <= j:
                H[i][j] = 1
            elif i == j + 1:
                H[i][j] = -1
            else:
                H[i][j] = 0
    return H


# L = np.array(
#     [
#         [1,0,0,0,0],
#         [1,1,0,0,0],
#         [0,-1/5,1,0,0],
#         [0,0,5/7,1,0],
#         [0,0,0,21/27,1]
#     ]
# )
# U = np.array(
#     [
#         [1,3,2,3,2],
#         [0,-5,2,-9,3],
#         [0,0,7/5,1/5,8/5],
#         [0,0,0,27/7,13/7],
#         [0,0,0,0,5/9]
#     ]
# )

def hessenberg_lu(H, n):
    H = np.asarray(H)
    n = H.shape[0]
    U = H.copy()
    L = np.eye(n)

    for i in range(n-1):
        if U[i, i] == 0:
            raise ZeroDivisionError(f"Zero pivot encountered at U[{i},{i}]")
        # multiplier for the (i+1)-th row
        l = U[i+1, i] / U[i, i]
        L[i+1, i] = l
        # subtract l * (i-th row of U) from (i+1)-th row of U
        for j in range(i, n):
            U[i+1, j] -= l * U[i, j]

    return L, U
def partial_pivoted_hessenberg(H, n):
    H = H.copy().astype(float)
    n = H.shape[0]

    P = np.eye(n)
    L = np.zeros((n, n))
    U = H.copy()

    for k in range(n - 1):
        if abs(U[k + 1, k]) > abs(U[k, k]):
            U[[k, k + 1], k:] = U[[k + 1, k], k:]
            P[[k, k + 1], :] = P[[k + 1, k], :]
            if k > 0:
                L[[k, k + 1], :k] = L[[k + 1, k], :k]
        L[k + 1, k] = U[k + 1, k] / U[k, k]
        U[k + 1, k:] -= L[k + 1, k] * U[k, k:]

    np.fill_diagonal(L, 1.0)
    return P, L, U


A = modified_H_matrix(5)
print(A)
print()
L, U = hessenberg_lu(A, 5)
print(L)
print()
print(U)
print()
print(L @ U)
# We now test it for random matrices satisfying the property
def time_complexity_tester_and_plotter():
    sizes = [2**i for i in range(1, 12)]
    times = []

    for n in sizes:
        # run the factorization many times so it's never underflow‐to‐zero
        reps = max(1, int(1e6 / (n**2)))   # e.g. ~1e6 total ops
        t0 = time.perf_counter()
        for _ in range(reps):
            H = make_H_matrix(n)
            partial_pivoted_hessenberg(H, n)
        t1 = time.perf_counter()

        # average per‐run time
        times.append((t1 - t0) / reps)

    log_n = np.log2(sizes)
    log_t = np.log2(times)

    p, C = np.polyfit(log_n, log_t, 1)

    print(f"Empirical slope on log–log plot: p ≈ {p:.2f}")

    plt.figure()
    plt.loglog(sizes, times, 'o-')         # log–log plot
    plt.xlabel('Matrix size $n$ (log₂ scale)')
    plt.ylabel('Time per run $T(n)$ (log₂ scale)')
    plt.grid(True, which='both', ls='--')
    plt.tight_layout()
    plt.savefig('(11_teta)complexity_loglog.png')

# time_complexity_tester_and_plotter()