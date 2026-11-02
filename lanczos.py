# Copy of lanczos_iteration from handin.ipynb ("Lanczos Iteration" section), so that map_test.ipynb can import it.

import numpy as np
import scipy.linalg as la


def lanczos_iteration(L, m, K=4, which="smallest", v1=None, seed=0, reorth=False, tol=1e-12):
    """
    Compute the Ritz pairs of a symmetric matrix L using the Lanczos algorithm.

    Parameters:
        L: The graph Laplacian, symmetric (numpy array or scipy sparse matrix), shape (n, n)
        m: The number of Lanczos steps
        K: The number of Ritz pairs to return. If None, all m pairs are returned
        which: "smallest" (for spectral clustering) or "largest"
        v1: Initial vector. If None, a random vector is used
        seed: The seed for the random number generator (used if v1 is None)
        reorth: If True, w_j is re-orthogonalized against all previous v_i (not part of Alg. 6.15,
                but prevents the loss of orthogonality / ghost eigenvalues in floating point)
        tol: beta_{j+1} below tol * ||L v_j|| is treated as beta_{j+1} = 0 (breakdown).
             If the algorithm stops early at step j, m is reduced to j.

    Returns:
        theta: Ritz values theta_i^(m), shape (K,)
        U: Ritz vectors u_i^(m) = V_m y_i^(m) as columns, shape (n, K)
        res: Residual norms |beta_m (e_m^T y_i^(m))|, shape (K,)
    """

    n = L.shape[0]

    # 1. Initial vector of 2-norm unity, beta_1 = 0, v_0 = 0
    if v1 is None:
        v1 = np.random.default_rng(seed).random(n)
    V = np.zeros((n, m))
    V[:, 0] = v1 / la.norm(v1)
    alphas = np.zeros(m)
    betas = np.zeros(m)
    beta = 0.0
    v_prev = np.zeros(n)

    for j in range(m):
        v = V[:, j]
        w = L @ v - beta * v_prev                     # 3. w_j = L v_j - beta_j v_{j-1}
        alpha = w @ v                                 # 4. alpha_j = (w_j, v_j)
        w = w - alpha * v                             # 5. w_j = w_j - alpha_j v_j
        if reorth:
            w = w - V[:, :j + 1] @ (V[:, :j + 1].T @ w)
        beta_next = la.norm(w)                        # 6. beta_{j+1} = ||w_j||_2

        alphas[j] = alpha
        betas[j] = beta_next

        if beta_next <= tol * max(abs(alpha), beta, 1.0):   # 6. If beta_{j+1} = 0 then Stop
            betas[j] = 0.0
            m = j + 1
            break
        if j + 1 < m:
            V[:, j + 1] = w / beta_next               # 7. v_{j+1} = w_j / beta_{j+1}
        v_prev, beta = v, beta_next

    V_m, alphas, betas = V[:, :m], alphas[:m], betas[:m]
    T_m = np.diag(alphas) + np.diag(betas[:-1], 1) + np.diag(betas[:-1], -1)
    beta_m = betas[-1]        # coefficient of v_{m+1} in L V_m = V_m T_m + beta_m v_{m+1} e_m^T


    # Ritz values theta^(m) in sigma(T_m) and eigenvectors y^(m) of T_m (ascending order)
    theta, Y = la.eigh(T_m)
    K = m if K is None else min(K, m)
    idx = np.arange(K) if which == "smallest" else np.arange(m - 1, m - K - 1, -1)
    theta, Y = theta[idx], Y[:, idx]

    # Ritz vectors u^(m) = V_m y^(m)
    U = V_m @ Y

    # Residual ||(L - theta^(m) I) u^(m)||_2 = |beta_m (e_m^T y^(m))|, where e_m^T y^(m) = Y[-1, :]
    e_m = np.zeros(m)
    e_m[-1] = 1.0
    res = np.abs(beta_m * (e_m @ Y))

    return theta, U, res
