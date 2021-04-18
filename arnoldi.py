import numpy as np
from hh_funcs import householder_ls
from lu_funcs import solve_U


def arnoldi_step(A, Q, H, i, apply_pc=None, normalise=False):
    """
    Perform a single step of the Arnoldi algorithm.
    """
    v = A.dot(Q[:,i])

    if apply_pc:
        v = apply_pc(v)

    H[:i+1,i] = Q[:,:i+1].conj().T.dot(v)

    if normalise:
        H[:i+1,i] /= A.shape[0]

    v -= Q[:,:i+1].dot(H[:i+1,i])

    H[i+1,i] = np.linalg.norm(v)

    if normalise:
        H[i+1,i] /= np.sqrt(A.shape[0])

    Q[:,i+1] = v / H[i+1,i]


def arnoldi(A, b, k, normalise=None):
    """
    For a matrix A, apply k iterations of the Arnoldi algorithm,
    using b as the first basis vector.

    :param A: an mxm numpy array
    :param b: m dimensional numpy array, the starting vector
    :param k: integer, the number of iterations
    :param normalise: choose whether the columns of H should be normalised (used
                      in Arnoldi polynomial interpolation)

    :return Q: an mx(k+1) dimensional numpy array containing the orthonormal basis
    :return H: a (k+1)xk dimensional numpy array containing the upper \
    Hessenberg matrix
    """
    m = A.shape[0]

    Q = np.zeros((m, k+1), dtype=A.dtype)
    H = np.zeros((k+1, k), dtype=A.dtype)

    Q[:,0] = b

    # During polynomial interpolation we don't want to normalise here
    if not normalise:
         Q /= np.linalg.norm(b)

    for i in range(k):
        arnoldi_step(A, Q, H, i, normalise=normalise)

    return Q, H


def GMRES(A, b, maxit, tol, x0=None, return_residual_norms=False,
          return_residuals=False, apply_pc=None):
    """
    For a matrix A, solve Ax=b using the basic GMRES algorithm.

    :param A: an mxm numpy array
    :param b: m dimensional numpy array
    :param maxit: integer, the maximum number of iterations
    :param tol: floating point number, the tolerance for termination
    :param x0: the initial guess (if not present, use b)
    :param return_residual_norms: logical
    :param return_residuals: logical

    :return x: an m dimensional numpy array, the solution
    :return nits: if converged, the number of iterations required, otherwise \
    equal to -1
    :return rnorms: nits dimensional numpy array containing the norms of \
    the residuals at each iteration
    :return r: mxnits dimensional numpy array, column k contains residual \
    at iteration k
    """

    def calc_givens(v1, v2): # ROTATE
        t = np.sqrt(v1**2 + v2**2)
        return v2 / t, v1 / t # sin(theta), cos(theta)

    if x0 is None:
        x0 = b

    if apply_pc is None:
        # If no preconditioner, set to it the identity function
        apply_pc = lambda x: x

    m = A.shape[0]
    maxit = min(maxit, m)

    Q = np.zeros((m, maxit+1), dtype=A.dtype)
    H = np.zeros((maxit+1, maxit), dtype=A.dtype)

    r = np.zeros((m, maxit), dtype=A.dtype)
    rnorms = np.zeros(maxit)

    rots = np.zeros((maxit, 2)) # cols: sin, cos

    r0 = apply_pc(b - A.dot(x0))

    be1 = np.zeros(maxit+1)
    be1[0] = np.linalg.norm(r0)

    Q[:,0] = r0 / np.linalg.norm(r0)

    for i in range(maxit):
        # Add new columns to Q and H using Arnoldi
        v = apply_pc(A.dot(Q[:,i]))

        H[:i+1,i] = Q[:,:i+1].conj().T.dot(v)
        v -= Q[:,:i+1].dot(H[:i+1,i])

        H[i+1,i] = np.linalg.norm(v)
        Q[:,i+1] = v / H[i+1,i]

        # Update new column with previous rotations
        for k in range(i):
            H[k,i], H[k+1,i] = rots[k,1] * H[k,i] + rots[k,0] * H[k+1,i], \
                                -rots[k,0] * H[k,i] + rots[k,1] * H[k+1,i]

        # Get current rotations
        rots[i,0], rots[i,1] = calc_givens(H[i,i], H[i+1,i])

        # Apply new rotation to be1
        be1[i], be1[i+1] = rots[i,1]*be1[i], -rots[i,0]*be1[i]

        # Apply new rotation to the final pair of H values
        H[i,i] = rots[i,1]*H[i,i] + rots[i,0]*H[i+1,i]
        H[i+1,i] = 0


        # Solve resulting triangular system then change variables back to y
        y = solve_U(H[:i+1,:i+1], be1[:i+1])
        x = x0 + Q[:,:i+1].dot(y)

        # Compute residuals
        r[:,i] = A.dot(x) - b
        rnorms[i] = np.linalg.norm(r[:,i])

        if rnorms[i] < tol:
            rnorms = rnorms[:i+1]
            r = r[:,:i+1]
            break

    else:
        i = -2

    if return_residual_norms and return_residuals:
        return x, i+1, rnorms, r
    elif return_residual_norms:
        return x, i+1, rnorms
    elif return_residuals:
        return x, i+1, r
    else:
        return x, i+1
