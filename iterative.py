import numpy as np
from lup_funcs import LUP_inplace, solve_from_LUP, solve_LUP
from hh_funcs import householder, householer_qr
import time


def pow_it(A, x0, tol, maxit, store_iterations = False):
    """
    For a matrix A, apply the power iteration algorithm with initial
    guess x0, until either
    ||r|| < tol where
    r = Ax - lambda*x,
    or the number of iterations exceeds maxit.
    :param A: an mxm numpy array
    :param x0: the starting vector for the power iteration
    :param tol: a positive float, the tolerance
    :param maxit: integer, max number of iterations
    :param store_iterations: if True, then return the entire sequence \
    of power iterates, instead of just the final iteration. Default is \
    False.
    :return x: an m dimensional numpy array containing the final iterate, or \
    if store_iterations, an mxmaxit dimensional numpy array containing all \
    the iterates.
    :return lambda0: the final eigenvalue.
    """

    v = x0

    if store_iterations:
        V = np.zeros((v.size, maxit+1), dtype=A.dtype)

        V[:,0] = x0

    for i in range(1, maxit+1):
        w = A.dot(v)
        v = w / np.linalg.norm(w)

        if store_iterations:
            V[:,i] = v

        lambda0 = v.conj().dot(A.dot(v))

        if np.linalg.norm(A.dot(v) - lambda0 * v) < tol:
            print(f"breaking at {i}!")
            break

    if store_iterations:
        return V, lambda0
    else:
        return v, lambda0


def inverse_it(A, x0, mu, tol, maxit, store_iterations = False):
    """
    For a Hermitian matrix A, apply the inverse iteration algorithm
    with initial guess x0, using the same termination criteria as
    for pow_it.
    :param A: an mxm numpy array
    :param mu: a floating point number, the shift parameter
    :param x0: the starting vector for the power iteration
    :param tol: a positive float, the tolerance
    :param maxit: integer, max number of iterations
    :param store_iterations: if True, then return the entire sequence \
    of inverse iterates, instead of just the final iteration. Default is \
    False.
    :return x: an m dimensional numpy array containing the final iterate, or \
    if store_iterations, an mxmaxit dimensional numpy array containing \
    all the iterates.
    :return l: a floating point number containing the final eigenvalue \
    estimate, or if store_iterations, an m dimensional numpy array containing \
    all the iterates.
    """

    v = x0

    if store_iterations:
        V = np.zeros((v.size, maxit+1), dtype=A.dtype)
        V[:,0] = v

    # Not LU yet but will be when transformed in-place
    LU = A - mu * np.eye(*A.shape)
    p = LUP_inplace(LU)

    for i in range(1, maxit+1):
        w = solve_from_LUP(LU, v, p)
        v = w / np.linalg.norm(w)

        if store_iterations:
            V[:,i] = v

        lambda0 = v.conj().dot(A.dot(v))

        if np.linalg.norm(A.dot(v) - lambda0 * v) < tol:
            break

    if store_iterations:
        return V, lambda0
    else:
        return v, lambda0


def rq_it(A, x0, tol, maxit, store_iterations = False):
    """
    For a Hermitian matrix A, apply the Rayleigh quotient algorithm
    with initial guess x0, using the same termination criteria as
    for pow_it.
    :param A: an mxm numpy array
    :param x0: the starting vector for the power iteration
    :param tol: a positive float, the tolerance
    :param maxit: integer, max number of iterations
    :param store_iterations: if True, then return the entire sequence \
    of inverse iterates, instead of just the final iteration. Default is \
    False.
    :return x: an m dimensional numpy array containing the final iterate, or \
    if store_iterations, an mxmaxit dimensional numpy array containing \
    all the iterates.
    :return l: a floating point number containing the final eigenvalue \
    estimate, or if store_iterations, an m dimensional numpy array containing \
    all the iterates.
    """

    v = x0
    lambda0 = v.conj().dot(A.dot(v))
    identity = np.eye(*A.shape, dtype=A.dtype)

    if store_iterations:
        V = np.zeros((v.size, maxit+1), dtype=A.dtype)

        V[:,0] = v

    for i in range(1, maxit+1):
        w = solve_LUP(A - lambda0 * identity, v).reshape((v.size,))
        v = w / np.linalg.norm(w)

        if store_iterations:
            V[:,i] = v

        lambda0 = v.conj().dot(A.dot(v))

        if np.linalg.norm(A.dot(v) - lambda0 * v) < tol:
            break

    if store_iterations:
        return V, lambda0
    else:
        return v, lambda0


def pure_QR(A, maxit, tol=1e-8, in_place=True, return_iterations=False, termination=None):
    """
    For matrix A, apply the QR algorithm and return the result.

    :param A: an mxm numpy array
    :param maxit: the maximum number of iterations
    :param tol: termination tolerance
    :param in_place: choice of method
    :param return_iterations: boolean to choose whether to return the number of
                              QR iterations until convergence
    :param termination: option to provide custom termination criteria

    :return Ak: the result
    """

    m = A.shape[0]
    if return_iterations:
        iterations = 0

    # in_place = True avoids creating new NumPy arrays each loop; this however
    # does not seem to noticably improve performance
    if in_place:
        AI = np.c_[A, np.eye(m)]

        for i in range(maxit):
            householder(AI, kmax=m)
            AI[:,:m] = AI[:,:m].dot(AI[:,m:].conj().T)
            AI[:,m:] = np.eye(m)

            if np.linalg.norm(AI[:,:m][np.tril_indices(m, k=-1)])/m**2 < tol:
                break

        return AI[:,:m]

    # Simplest method, which seems to perform just as well
    else:
        for i in range(maxit):
            Q, R = householder_qr(A)
            A = R.dot(Q)
            if return_iterations:
                iterations += 1

            if termination:
                check = termination(A)
            else:
                check = np.linalg.norm(A[np.tril_indices(m, k=-1)]) / m**2 < tol

            if check:
                break

        if return_iterations:
            return A, iterations
        else:
            return A


def time_QR(m, in_place, maxit=10000, tol=1e-6):
    """
    Function to time pure_QR performance using specified method.
    """
    np.random.seed(1337)
    A = np.random.randn(m, m) + 1j*np.random.randn(m, m)

    t1 = time.time()
    pure_QR(A, maxit, tol, in_place=in_place)
    t2 = time.time()
    return t2-t1
