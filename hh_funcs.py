import numpy as np
from lu_funcs import solve_U


def householder(A, kmax=None):
    """
    Given a real mxn matrix A, find the reduction to upper triangular matrix R
    using Householder transformations.

    :param A: an mxn-dimensional numpy array
    :param kmax: an integer, the number of columns of A to reduce \
    to upper triangular. If not present, will default to n.

    :return R: an mxn-dimensional numpy array containing the upper \
    triangular matrix
    """

    m, n = A.shape
    if kmax is None:
        kmax = n

    def sign(x): # since NumPy's sign function doesn't quite do what we want
        return 1 if x == 0 else x / np.abs(x)

    for k in range(kmax):
        v = A[k:,k].copy()
        v[0] += sign(v[0]) * np.linalg.norm(v)
        v /= np.linalg.norm(v)
        A[k:,k:] -= 2 * np.outer(v, v.conj().dot(A[k:,k:]))

    return A


def householder_solve(A, b):
    """
    Given a real mxm matrix A, use the Householder transformation to solve
    Ax_i=b_i, i=1,2,...,k.

    :param A: an mxm-dimensional numpy array
    :param b: an mxk-dimensional numpy array whose columns are the \
    right-hand side vectors b_1,b_2,...,b_k.

    :return x: an mxk-dimensional numpy array whose columns are the \
    right-hand side vectors x_1,x_2,...,x_k.
    """
    m = A.shape[0]
    R_b_hat = householder(np.c_[A, b], kmax=m)

    return solve_U(R_b_hat[:,:m], R_b_hat[:,m:])


def householder_qr(A):
    """
    Given a real mxn matrix A, use the Householder transformation to find
    the QR factorisation of A.

    :param A: an mxn-dimensional numpy array

    :return Q: an mxm-dimensional numpy array
    :return R: an mxn-dimensional numpy array
    """

    m, n = A.shape

    identity = np.eye(m, dtype=A.dtype)
    Ahat = np.hstack((A, identity))
    Ahat_hh = householder(Ahat, kmax=n)

    R = Ahat_hh[:,:n]
    Q = Ahat_hh[:,n:].conj().T

    return Q, R


def householder_ls(A, b):
    """
    Given a real mxn matrix A and an m dimensional vector b, find the
    least squares solution to Ax = b.

    :param A: an mxn-dimensional numpy array
    :param b: an m-dimensional numpy array

    :return x: an n-dimensional numpy array
    """

    m, n = A.shape

    Ahat = np.c_[A, b] # sticks b to the end of A
    Ahat_hh = householder(Ahat, kmax=n)

    return solve_U(Ahat_hh[:n,:-1], Ahat_hh[:n,-1])
