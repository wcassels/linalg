# linalg

Various linear algebra procedures implemented in Python. Written as part of the computational linear algebra module at Imperial College London.

Contents:
- elementary_linalg.py : Basic matrix/vector multiplication
- gram_schmidt.py : QR decomposition using classical/modified Gram-Schmidt 
- hh_funcs.py : QR decomposition and least squares solution using Householder reflections
- utils.py : Operator norm, condition number
- lu_funcs.py : In-place LU decomposition, corresponding solution procedures
- lup_funcs.py : As above, now introducing pivoting 
- hess.py : Reduction to Hessenberg using Householder reflections
- iterative.py : Power/inverse/Rayleigh iteration, pure QR algorithm
- arnoldi.py : Arnoldi iteration and GMRES with Givens rotations
