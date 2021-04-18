import numpy as np

def hessenberg(A):
    """
    For a matrix A, transform to Hessenberg form H by Householder
    similarity transformations, in place.

    :param A: an mxm numpy array
    """

    def sign(x):
        return (1 if x >= 0 else -1)

    m = A.shape[0]

    for k in range(m-2):
        v = A[k+1:,k].copy()
        v[0] += sign(v[0]) * np.linalg.norm(v)
        v /= np.linalg.norm(v)
        A[k+1:,k:] -= 2 * np.outer(v, v.conj().dot(A[k+1:,k:]))
        A[:,k+1:] -= 2 * np.outer(A[:,k+1:].dot(v), v.conj())


def hessenbergQ(A):
    """
    For a matrix A, transform to Hessenberg form H by Householder
    similarity transformations, in place, and return the matrix Q
    for which QHQ^* = A.

    :param A: an mxm numpy array

    :return Q: an mxm numpy array
    """
    def sign(x):
        return (1 if x >= 0 else -1)

    m, n = A.shape

    Q = np.eye(m, dtype=A.dtype)

    for k in range(m-2):
        v = A[k+1:,k].copy()
        v[0] += sign(v[0]) * np.linalg.norm(v)
        v /= np.linalg.norm(v)

        A[k+1:,k:] -= 2 * np.outer(v, v.conj().dot(A[k+1:,k:]))
        A[:,k+1:] -= 2 * np.outer(A[:,k+1:].dot(v), v.conj())

        Q[k+1:] -= 2 * np.outer(v, v.conj().dot(Q[k+1:]))

    return Q.conj().T

def hessenberg_ev(H):
    """
    Given a Hessenberg matrix, return the eigenvalues and eigenvectors.
    :param H: an mxm numpy array
    :return ee: an m dimensional numpy array containing the eigenvalues of H
    :return V: an mxm numpy array whose columns are the eigenvectors of H
    """
    m, n = H.shape
    assert(m==n)
    assert(np.linalg.norm(H[np.tril_indices(m, -2)]) < 1.0e-6)
    _, V = np.linalg.eig(H)
    return V


def ev(A):
    """
    Given a matrix A, return the eigenvalues and eigenvectors. This should
    be done by using your functions to reduce to upper Hessenberg
    form, before calling hessenberg_ev (which you should not edit!).

    :param A: an mxm numpy array

    :return ee: an m dimensional numpy array containing the eigenvalues of A
    :return V: an mxm numpy array whose columns are the eigenvectors of A
    """

    Q = hessenbergQ(A)
    return Q.dot(hessenberg_ev(A))
