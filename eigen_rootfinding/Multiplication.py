import numpy as np
import mpmath as mp
import itertools
from scipy.linalg import eig, schur
from eigen_rootfinding.polynomial import MultiCheb, MultiPower
from eigen_rootfinding.utils import memoize
from scipy.stats import ortho_group

def indexarray(matrix_terms, which, var):
    """Compute the array mapping monomials under multiplication by x_var

    Parameters
    ----------
    matrix_terms : 2d integer ndarray
        Array containing the monomials in order. matrix_terms[i] is the array
        containing the exponent for each variable in the ith multivariate
        monomial
    which : slice object
        object to index into the matrix_terms for the monomials we want to multiply by var
    var : int
        Variable to multiply by: x_0, ..., x_(dim-1)

    Returns
    -------
    arr : 1d integer ndarray
        Array containing the indices of the lower-degree monomials after multiplication
        by x_var
    """
    mults = matrix_terms[which].copy()
    mults[:, var] += 1
    return np.argmin(np.abs(mults[:, np.newaxis] - matrix_terms[np.newaxis]).sum(axis=-1), axis=1)

def indexarray_cheb(matrix_terms, which, var):
    """Compute the array mapping Chebyshev monomials under multiplication by x_var:

        T_1*T_0 = T_1
        T_1*T_n = .5(T_(n+1)+ T_(n-1))

    Parameters
    ----------
    matrix_terms : 2d integer ndarray
        Array containing the monomials in order. matrix_terms[i] is the array
        containing the degree for each univariate Chebyshev monomial in the ith
        multivariate monomial
    m : int
        Number of monomials of highest degree, i.e. those that do not need to be
        multiplied
    var : int
        Variable to multiply by: x_0, ..., x_(dim-1)

    Returns
    -------
    arr1 : 1d integer ndarray
        Array containing the indices of T_(n+1)
    arr2 : 1d
        Array containing the indices of T_(n-1)
    """
    up = matrix_terms[which].copy()
    up[:, var] += 1
    down = matrix_terms[which].copy()
    down[:, var] -= 1
    down[down[:, var]==-1, var] += 2
    arr1 = np.argmin(np.abs(up[:, np.newaxis] - matrix_terms[np.newaxis]).sum(axis=-1), axis=1)
    arr2 = np.argmin(np.abs(down[:, np.newaxis] - matrix_terms[np.newaxis]).sum(axis=-1), axis=1)
    return arr1, arr2

def ms_matrices(E, Q, matrix_terms, dim):
    """Compute the Möller-Stetter matrices in the monomial basis from a
    reduced Macaulay matrix

    Parameters
    ----------
    E : (m, k) ndarray
        Columns of the reduced Macaulay matrix corresponding to the quotient basis
    Q : (l, n) 2d ndarray
        Matrix whose columns give the quotient basis in terms of the monomial basis
    matrix_terms : 2d ndarray
        Array with ordered monomial basis
    dim : int
        Number of variables

    Returns
    -------
    M : (n, n, dim) ndarray
        Array containing the nxn Möller-Stetter matrices, where the matrix
        corresponding to multiplication by x_i is M[..., i]
    """
    m = E.rows
    M = []
    A = (-E.H).copy()
    A.cols += Q.rows
    A[:,A.cols-Q.rows:] = Q.H
    for i in range(dim):
        arr = indexarray(matrix_terms, m, i)
        A_indexed = mp.matrix([[A[row,colnum] for colnum in arr] for row in range(A.rows)])
        M.append(A_indexed*Q)
    return M

def ms_matrices_cheb(E, Q, matrix_terms, dim):
    """Compute the Möller-Stetter matrices in the Chebyshev basis from a
    reduced Macaulay matrix

    Parameters
    ----------
    E : (m, k) ndarray
        Columns of the reduced Macaulay matrix corresponding to the quotient basis
    Q : (l, n) 2d ndarray
        Matrix whose columns give the quotient basis in terms of the Chebyshev basis
    matrix_terms : 2d ndarray
        Array with ordered Chebyshev basis
    dim : int
        Number of variables

    Returns
    -------
    M : (n, n, dim) ndarray
        Array containing the nxn Möller-Stetter matrices, where the matrix
        corresponding to multiplication by x_i is M[..., i]
    """
    m = E.rows
    M = []
    A = (-E.H).copy()
    A.cols += Q.rows
    A[:,A.cols-Q.rows:] = Q.H
    for i in range(dim):
        arr1, arr2 = indexarray_cheb(matrix_terms, m, i)
        A_up = mp.matrix([A[:,colnum] for colnum in arr1]).T
        A_down = mp.matrix([A[:,colnum] for colnum in arr2]).T
        M.append(.5*(A_up + A_down)*Q)
    return M

def sort_eigs(eigs, diag, arr=False):
    """Sorts the eigs array to match the order on the diagonal
    of the Schur factorization

    Parameters
    ----------
    eigs : 1d ndarray
        Array of unsorted eigenvalues
    diag : 1d complex ndarray
        Array containing the diagonal of the approximate Schur factorization

    Returns
    -------
    w : 1d ndarray
        Eigenvalues from eigs sorted to match the order in diag
    """
    if arr:
        n = len(diag)
        lst = list(range(n))
        arr = []
        for eig in eigs:
            i = lst[np.argmin([mp.fabs(diag[l] - eig) for l in lst])]
            arr.append(i)
            lst.remove(i)
        return np.argsort(arr)
    else:
        n = len(diag)
        lst = list(range(n))
        sorted_eigs = [0]*n
        for eig in eigs:
            dists = [mp.fabs(eig - diag[l]) for l in lst]
            nearest = lst.pop(np.argmin(dists))
            sorted_eigs[nearest] = eig
        return sorted_eigs

@memoize
def get_Q_c(dim):
    """ Generates a once-chosen random orthogonal matrix and a random linear combination
    for use in the simultaneous eigenvalue compution.

    Parameters
    ----------
    dim : int
        Dimension of the system

    Returns
    -------
    Q : (dim, dim) ndarray
        Random orthogonal rotation
    c : (dim, ) ndarray
        Random linear combination
    """
    np.random.seed(103)
    Q = ortho_group.rvs(dim)
    c = np.random.randn(dim)
    return Q, c

def msroots(M):
    """Computes the roots to a system via the eigenvalues of the Möller-Stetter
    matrices. Implicitly performs a random rotation of the coordinate system
    to avoid repeated eigenvalues arising from special structure in the underlying
    polynomial system. Approximates the joint eigenvalue problem using a Schur
    factorization of a linear combination of the matrices.

    Parameters
    ----------
    M : (n, n, dim) ndarray
        Array containing the nxn Möller-Stetter matrices, where the matrix
        corresponding to multiplication by x_i is M[..., i]

    Returns
    -------
    roots : (n, dim) ndarray
        Array containing the approximate roots of the system, where each row
        is a root.
    """
    dim = len(M)

    # perform a random rotation with a random orthogonal Q
    Q, c = get_Q_c(dim)
    Q = mp.matrix(Q)
    c = mp.matrix(c)
    My = [sum([Q[j,i]*M[i] for i in range(dim)]) for j in range(dim)]

    eigs = mp.matrix(dim, M[0].rows)
    # Compute the matrix U that triangularizes a random linear combination
    M = sum([Myj*cj for Myj, cj in zip(My,c)])
    U = mp.schur(M)[0]

    for j in range(0, dim):
        T = (U.H)*(My[j])*U
        w = mp.eig(My[j], right=False)
        sorted_eigs = sort_eigs(w, [T[_,_] for _ in range(T.rows)])
        for k,eig in enumerate(sorted_eigs):
            eigs[j,k] = eig

    # Rotate back before returning, transposing to match expected shape
    return (Q.T*eigs).T
