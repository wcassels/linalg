import numpy as np

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
