import numpy as np
from functools import partial
import timeit


def orthog_cpts(v, Q):
    """
    Given a vector v and an orthonormal set of vectors q_1,...q_n,
    compute v = r + u_1q_1 + u_2q_2 + ... + u_nq_n
    for scalar coefficients u_1, u_2, ..., u_n and
    residual vector r

    :param v: an m-dimensional numpy array
    :param Q: an mxn-dimensional numpy array whose columns are the \
    orthonormal vectors

    :return r: an m-dimensional numpy array containing the residual
    :return u: an n-dimensional numpy array containing the coefficients
    """
    u = Q.T.dot(v)
    r = v - Q.dot(u)

    return r, u


def solveQ(Q, b):
    """
    Given a unitary mxm matrix Q and a vector b, solve Qx=b for x.

    :param Q: an mxm dimensional numpy array containing the unitary matrix
    :param b: the m dimensional array for the RHS

    :return x: m dimensional array containing the solution.
    """

    return Q.conj().T.dot(b)


def time_solveQ():
    """
    Compare solveQ's performance with Numpy's general purpose algorithm
    """
    for m in [100, 200, 400, 1000]:
        A = np.random.randn(m, m) + np.random.randn(m, m)*1j
        Q, _ = np.linalg.qr(A)
        b = np.random.randn(m) + np.random.randn(m)*1j

        solveq_time = timeit.Timer(partial(solveQ, Q, b)).timeit(number=1)
        numpy_time = timeit.Timer(partial(np.linalg.solve, Q, b)).timeit(number=1)

        print(f"m={m}")
        print(f"solveQ time: {solveq_time}")
        print(f"Numpy time: {numpy_time}")

    # we see solveQ is substantially quicker than the general purpose LU
    # factorisation algorithm, as expected


def orthog_proj(Q):
    """
    Given a vector v and an orthonormal set of vectors q_1,...q_n,
    compute the orthogonal projector P that projects vectors onto
    the subspace spanned by those vectors.

    :param Q: an mxn-dimensional numpy array whose columns are the \
    orthonormal vectors

    :return P: an mxm-dimensional numpy array containing the projector
    """

    return Q.dot(Q.conj().T)


def orthog_space(V):
    """
    Given set of vectors u_1,u_2,..., u_n, compute the
    orthogonal complement to the subspace U spanned by the vectors.

    :param V: an mxn-dimensional numpy array whose columns are the \
    vectors u_1,u_2,...,u_n.

    :return Q: an lxm-dimensional numpy array whose columns are an \
    orthonormal basis for the subspace orthogonal to U.
    """

    m, n = V.shape
    Q, R = np.linalg.qr(V, mode="complete") # get complete QR factorisation

    # Q = [Q1, Q2], where Q1 spans Csp(V), we want the complement of this so return Q2

    return Q[:,n:]


def GS_classical(A):
    """
    Given an mxn matrix A, compute the QR factorisation by classical
    Gram-Schmidt algorithm.

    :param A: mxn numpy array

    :return Q: mxn numpy array
    :return R: nxn numpy array
    """

    m, n = A.shape

    Q = np.zeros((m, n), dtype=A.dtype)
    R = np.zeros((n, n), dtype=A.dtype)

    for j in range(n):
        R[:j-1,j] = np.dot(Q[:,:j-1].T.conj(), A[:,j])
        Q[:,j] = A[:, j] - np.dot(Q[:,:j-1], R[:j-1,j])
        R[j,j] = np.linalg.norm(Q[:,j])
        Q[:,j] /= R[j,j]

    return Q, R

def GS_modified(A):
    """
    Given an mxn matrix A, compute the QR factorisation by modified
    Gram-Schmidt algorithm, producing

    :param A: mxn numpy array

    :return Q: mxn numpy array
    :return R: nxn numpy array
    """

    m, n = A.shape
    R = np.zeros((n, n), dtype=A.dtype)
    V = A.copy()

    for j in range(n):
        R[j,j] = np.linalg.norm(V[:,j])
        V[:,j] /= R[j,j]
        R[j,j+1:] = np.dot(V[:,j].conj(), V[:,j+1:])
        V[:,j+1:] -= np.outer(V[:,j], R[j,j+1:])

    return V, R


def GS_modified_get_R(A, k):
    """
    Given an mxn matrix A, with columns of A[:, 0:k] assumed orthonormal,
    return upper triangular nxn matrix R such that
    Ahat = A*R has the properties that
    1) Ahat[:, 0:k] = A[:, 0:k],
    2) A[:, k] is orthogonal to the columns of A[:, 0:k].

    :param A: mxn numpy array
    :param k: integer indicating the column that R should orthogonalise

    :return R: nxn numpy array
    """

    m, n = A.shape
    R = np.eye(n, dtype=A.dtype)

    r_kk = np.linalg.norm(A[:,k])

    R[k,k+1:] = -np.dot(A[:,k].conj() / r_kk, A[:,k+1:])
    R[k] /= r_kk

    return R

def GS_modified_R(A):
    """
    Implement the modified Gram Schmidt algorithm using the lower triangular
    formulation with Rs provided from GS_modified_get_R.
    :param A: mxn numpy array
    :return Q: mxn numpy array
    :return R: nxn numpy array
    """
    m, n = A.shape
    R = np.eye(n, dtype=A.dtype)
    for i in range(n):
        Rk = GS_modified_get_R(A, i)
        A[:,:] = np.dot(A, Rk)
        R[:,:] = np.dot(R, Rk)
    R = np.linalg.inv(R)
    return A, R
