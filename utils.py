import numpy as np


def operator_2_norm(A):
    """
    Given a real mxn matrix A, return the operator 2-norm.

    :param A: an mxn-dimensional numpy array

    :return o2norm: the norm
    """
    # A^TA is Hermitian so can use eigh
    eigs, _ = np.linalg.eigh(A.conj().T.dot(A))

    return np.sqrt(np.max(eigs))


def verify_norm_ineq(num_tests=100, max_m = 20, max_n = 20, tol=1e-6):
    """
    Verify the inequality |Ax| <= |A||x| for various dimensions m, n.

    :param num_tests: the number of (m, n) pairs to test

    :param tol: the tolerance for checking equality of floating point values
    """
    ms = np.random.randint(low=1, high=max_m, size=num_tests)
    ns = np.random.randint(low=1, high=max_n, size=num_tests)

    for m, n in zip(ms, ns):
        A = np.random.randn(m, n) + np.random.randn(m, n) * 1j
        x = np.random.randn(n) + np.random.randn(n) * 1j

        A_norm = operator_2_norm(A)
        Ax_norm = np.linalg.norm(A.dot(x))
        x_norm = np.linalg.norm(x)

        assert (Ax_norm - A_norm * x_norm < tol), f"Test failed for m={m}, n={n}: |Ax|={Ax_norm}, |A||x|={np.real(A_norm * x_norm)}"

    print("All tests passed successfully")


def cond(A):
    """
    Given a real mxn matrix A, return the condition number in the 2-norm.

    :return A: an mxn-dimensional numpy array

    :param ncond: the condition number
    """
    # A^TA is Hermitian so can use eigh
    eigs, _ = np.linalg.eigh(A.conj().T.dot(A))

    return np.sqrt(np.max(eigs)/np.min(eigs))
