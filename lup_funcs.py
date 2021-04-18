import numpy as np
from lu_funcs import solve_U, solve_L

def perm(p, i, j):
    """
    For p representing a permutation P, i.e. Px[i] = x[p[i]],
    replace with p representing the permutation P_{i,j}P, where
    P_{i,j} exchanges rows i and j.

    :param p: an m-dimensional numpy array of integers.
    """
    p[i], p[j] = p[j], p[i]


def LUP_inplace(A, count_swaps=False):
    """
    Compute the LUP factorisation of A with partial pivoting, using the
    in-place scheme so that the strictly lower triangular components
    of the array contain the strictly lower triangular components of
    L, and the upper triangular components of the array contain the
    upper triangular components of U.

    :param A: an mxm-dimensional numpy array

    :return p: an m-dimensional integer array describing the permutation \
    i.e. (Px)[i] = x[p[i]]
    """

    m = A.shape[0]

    p = np.arange(m)

    num_swaps = 0

    for k in range(m-1):
        i = k + np.argmax(np.abs(A[k:,k]))

        A[[k,i]] = A[[i,k]]
        perm(p, i, k)

        if i != k:
            num_swaps += 1

        A[k+1:,k] /= A[k,k]
        A[k+1:,k+1:] -= np.outer(A[k+1:,k], A[k,k+1:])

    if count_swaps:
        return p, num_swaps
    else:
        return p


def solve_LUP(A, b):
    """
    Solve Ax=b using LUP factorisation.

    :param A: an mxm-dimensional numpy array
    :param b: an m-dimensional numpy array

    :return x: an m-dimensional numpy array
    """
    p = LUP_inplace(A)

    return solve_from_LUP(A, b, p)


def solve_from_LUP(LU, b, p=None):
    """
    My own helper function to solve systems already factorised into compact
    LUP form.
    """
    if p is not None:
        b = b[p]

    return solve_U(LU, solve_L(LU, b, diagonal_ones=True))


def det_LUP(A):
    """
    Find the determinant of A using LUP factorisation.

    :param A: an mxm-dimensional numpy array

    :return detA: floating point number, the determinant.
    """

    _, num_swaps = LUP_inplace(A, count_swaps=True)

    return np.prod(np.diag(A)) * (-1) ** (num_swaps % 2)
