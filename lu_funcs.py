import numpy as np


def get_Lk(m, lvec):
    """Compute the lower triangular row operation mxm matrix L_k
    which has ones on the diagonal, and below diagonal entries
    in column k given by lvec (k is inferred from the size of lvec).

    :param m: integer giving the dimensions of L.

    :param lvec: a m-k-1 dimensional numpy array.

    :return Lk: an mxm dimensional numpy array.

    """

    s = lvec.size

    L = np.eye(m)
    L[m-s:,m-s-1] = lvec

    return L


def LU_inplace(A, return_iterations=False):
    """Compute the LU factorisation of A, using the in-place scheme so
    that the strictly lower triangular components of the array contain
    the strictly lower triangular components of L, and the upper
    triangular components of the array contain the upper triangular
    components of U.

    :param A: an mxm-dimensional numpy array

    :param return_iterations: boolean - choose whether to return an mxmxm array,
                              with slices corresponding to the value of A after
                              each iteration

    """

    m = A.shape[0]

    if return_iterations:
        As = np.zeros((m, m, m), dtype=A.dtype)
        As[:,:,0] = A

    for k in range(m-1):
        A[k+1:,k] /= A[k,k]
        A[k+1:,k+1:] -= np.outer(A[k+1:,k], A[k,k+1:])

        if return_iterations:
            As[:,:,k+1] = A

    if return_iterations:
        return As


def solve_L(L, b, diagonal_ones=False):
    """
    Solve systems Lx_i=b_i for x_i with L lower triangular, i=1,2,\\ldots,k

    :param L: an mxm-dimensional numpy array, assumed lower triangular
    :param b: an mxk-dimensional numpy array, with ith column containing \
    b_i
    :param diagonal_ones: an optional parameter to specify whether the
    diagonal entries should all be assumed to be 1 (leads to cleaner
    and more efficient code when it comes to solving LU systems)

    :return x: an mxk-dimensional numpy array, with ith column containing \
    the solution x_i

    """
    # doing it like this allows b to be 1-dim arrays
    m = b.shape[0]
    x = np.zeros_like(b, dtype=L.dtype)

    for i in range(m):
        x[i] = b[i] - L[i,:i].dot(x[:i])

        if not diagonal_ones:
            x[i] /= L[i,i]

    return x


def solve_U(U, b):
    """
    Solve systems Ux_i=b_i for x_i with U upper triangular, i=1,2,\\ldots,k

    :param U: an mxm-dimensional numpy array, assumed upper triangular
    :param b: an mxk-dimensional numpy array, with ith column containing \
    b_i

    :return x: an mxk-dimensional numpy array, with ith column containing \
    the solution x_i

    """
    # doing it like this allows b to be 1-dim arrays
    m = b.shape[0]
    x = np.zeros_like(b, dtype=U.dtype)

    for i in range(m-1, -1, -1):
        x[i] = (b[i] - U[i,i+1:].dot(x[i+1:])) / U[i,i]

    return x


def solve_LU(A, b):
    """
    Solves Ax = b using LU factorisation, then forwards and backwards
    substitution.
    """
    LU_inplace(A)
    return solve_U(A, solve_L(A, b, diagonal_ones=True))


def inverse_LU(A):
    """
    Form the inverse of A via LU factorisation.

    :param A: an mxm-dimensional numpy array.

    :return Ainv: an mxm-dimensional numpy array.

    """
    LU_inplace(A)

    return solve_U(A, solve_L(A, np.eye(*A.shape), diagonal_ones=True))
