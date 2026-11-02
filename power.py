# Copy of power_iteration from handin.ipynb (Part 2, power iteration), so that map_test.ipynb can import it.

import numpy as np
import scipy.linalg as la


def power_iteration(L, tol=1e-10, maxiter=1000, seed=0, c=None, found=None, callback=None):
    """
    Power iteration algorithm to compute the dominant eigenvalue and eigenvector of the graph Laplacian L.

    Parameters:
        L: The input matrix (sparse or dense)
        tol: The tolerance for convergence
        maxiter: The maximum number of iterations
        seed: The seed for the random number generator
        c: The shift. If None, power iteration is applied to L. If given, it is applied to c*I - L (requires c >= lambda_max(L)), which gives the smallest eigenpair of L.
        found: Optional list of previously computed eigenvectors (orthonormal). If given, the iterates are kept
               orthogonal to these vectors (deflation), so the method converges to the next eigenpair.
        callback: Optional function callback(lambda, v), called in every iteration with the current eigenvalue
                  estimate of L (shifted back if c is given) and the iterate v^(k). Used for the error plots.

    Returns:
        lambda^(k): The dominant eigenvalue after k iterations
        v^(k): The dominant eigenvector after k iterations
        k: The number of iterations performed
        converged: A boolean indicating whether the algorithm converged within the specified tolerance
        history: A list of residuals at each iteration
    """

    def apply(x):
        # Matrix-vector multiplication with L, or with the shifted matrix c*I - L
        return L @ x if c is None else c * x - L @ x

    # Deflation: matrix X with the found eigenvectors as columns
    X = np.column_stack(found) if found else None

    def deflate(x):
        # Deflation: remove the components of x along the found eigenvectors, x - X X^T x
        return x if X is None else x - X @ (X.T @ x)

    # Initialize a random vector v^(0), orthogonal to the found eigenvectors
    rng = np.random.default_rng(seed)
    v = deflate(rng.standard_normal(L.shape[0]))
    v = v / la.norm(v) # Normalize the vector

    w = deflate(apply(v))   # Initial matrix-vector multiplication: w = A v^(0), deflated
    converged = False  # Flag to check for convergence
    history = []

    for k in range(1, maxiter+1):
        v = w / la.norm(w)  # Normalize the vector, v^(k) = A v^(k-1) / ||A v^(k-1)||

        # Deflation in every iteration: rounding errors would otherwise reintroduce the found directions
        w = deflate(apply(v))  # Matrix-vector multiplication, A v^(k), reused in the next iteration

        lambda_ = v @ w  # Rayleigh quotient to estimate the eigenvalue, lambda^(k) = (v^(k))^T A v^(k)
        res = la.norm(w - lambda_ * v)  # Compute the residual of the pair (lambda^(k), v^(k))
        history.append(res)  # Store the residual for analysis
        if callback is not None:
            callback(lambda_ if c is None else c - lambda_, v)
        if res < tol: # Check for convergence
            converged = True
            break

    if c is not None:
        lambda_ = c - lambda_  # Shift back: eigenvalue of L = c - eigenvalue of c*I - L

    return lambda_, v, k, converged, history
