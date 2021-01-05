import numpy as np
import mpmath as mp
import itertools
from eigen_rootfinding.polynomial import Polynomial, MultiCheb, MultiPower
from eigen_rootfinding.utils import row_swap_matrix, MacaulayError, slice_top, mon_combos, \
                              num_mons_full, memoized_all_permutations, mons_ordered, \
                              all_permutations_cheb, ConditioningError, TooManyRoots, mp_solve_triangular
from matplotlib import pyplot as plt
from warnings import warn

macheps = 2.220446049250313e-16

def plot_scree(s,tol):
    plt.semilogy(s,marker='.')
    plt.plot(np.ones(len(s))*tol)
    plt.show()

def add_polys(degree, poly, poly_coeff_list):
    """Adds polynomials to a Macaulay Matrix.

    This function is called on one polynomial and adds all monomial multiples of
     it to the matrix.

    Parameters
    ----------
    degree : int
        The degree of the Macaulay Matrix
    poly : Polynomial
        One of the polynomials used to make the matrix.
    poly_coeff_list : list
        A list of all the current polynomials in the matrix.
    Returns
    -------
    poly_coeff_list : list
        The original list of polynomials in the matrix with the new monomial
        multiplications of poly added.
    """

    poly_coeff_list.append(poly.coeff)
    deg = degree - poly.degree
    dim = poly.dim

    mons = mon_combos([0]*dim,deg)

    for mon in mons[1:]: #skips the first all 0 mon
        poly_coeff_list.append(poly.mon_mult(mon, returnType = 'Matrix'))
    return poly_coeff_list

def find_degree(poly_list, verbose=False):
    '''Finds the appropriate degree for the Macaulay Matrix.

    Parameters
    --------
    poly_list: list
        The polynomials used to construct the matrix.
    verbose : bool
        If True prints the degree
    Returns
    -----------
    find_degree : int
        The degree of the Macaulay Matrix.

    '''
    if verbose:
        print('Degree of Macaulay Matrix:', sum(poly.degree for poly in poly_list) - len(poly_list) + 1)
    return sum(poly.degree for poly in poly_list) - len(poly_list) + 1

def compute_rank(M):
    S = mp.svd(M, compute_uv=False)
    tol = max(M.rows,M.cols)*S[0]*macheps
    return sum([s>tol for s in S])

def reduce_macaulay_qrt(M, cut, bezout_bound, max_cond=1e6):
    """Reduces the Macaulay matrix using the Transposed QR method.

    Parameters:
    -----------
    matrix : 2d ndarray
        The Macaulay matrix
    cut : int
        Number of columns of max degree
    max_cond : int or float
        Max condition number for the two condition number checks

    Returns:
    --------
    E : 2d ndarray
        The columns of the reduced Macaulay matrix corresponding to the quotient basis
    Q2 : 2d ndarray
        Matrix giving the quotient basis in terms of the monomial basis. Q2[:,i]
        being the coefficients for the ith basis element
    """
    # Check condition number before first QR
    cond_num = np.linalg.cond(M[:,:cut])
    if cond_num > max_cond:
        return None, "Condition number of the Macaulay high-degree columns is {}".format(cond_num)

    M = mp.matrix(M)

    # Check if numerical rank doesn't match bezout bound
    rank = compute_rank(M)
    bezout_rank = M.cols-bezout_bound
    if rank < bezout_rank:
        warn("Rank of Macaulay Matrix does not match the Bezout bound. Expected rank {}, found rank {}. System potentially has infinitely many solutions.".format(bezout_rank,rank))
    elif rank > bezout_rank:
        warn('Rank of Macaulay Matrix does not match the Bezout bound. Expected rank {}, found rank {}.'.format(bezout_rank,rank))

    # QR reduce the highest-degree columns
    Q,M[:,:cut] = mp.qr(M[:,:cut])
    M[:,cut:] = Q.H * M[:,cut:]
    Q = None
    del Q

    # If the matrix is "tall", compute an orthogonal transformation of the remaining
    # columns, generating a new polynomial basis
    if cut < M.rows:
        Q = mp.qr(M[cut:,cut:].T)[0] #mpmath can't do pivoted QR
        M[:cut,cut:] = M[:cut,cut:] * Q # Apply column transform

    # Return the backsolved columns and coefficient matrix for the quotient basis
    return mp_solve_triangular(M[:cut,:cut],M[:cut,bezout_rank:]),Q[:,Q.cols-bezout_bound:]

def reduce_macaulay_svd(M, cut, bezout_bound, max_cond=1e6):
    """Reduces the Macaulay matrix using the Transposed QR method.

    Parameters:
    -----------
    matrix : 2d ndarray
        The Macaulay matrix
    cut : int
        Number of columns of max degree
    max_cond : int or float
        Max condition number for the two condition number checks

    Returns:
    --------
    E : 2d ndarray
        The columns of the reduced Macaulay matrix corresponding to the quotient basis
    Q2 : 2d ndarray
        Matrix giving the quotient basis in terms of the monomial basis. Q2[:,i]
        being the coefficients for the ith basis element
    """
    # Check condition number before first QR
    cond_num = np.linalg.cond(M[:,:cut])
    if cond_num > max_cond:
        return None, "Condition number of the Macaulay high-degree columns is {}".format(cond_num)

    M = mp.matrix(M)

    # Check if numerical rank doesn't match bezout bound
    rank = compute_rank(M)
    bezout_rank = M.cols-bezout_bound
    if rank < bezout_rank:
        warn("Rank of Macaulay Matrix does not match the Bezout bound. Expected rank {}, found rank {}. System potentially has infinitely many solutions.".format(bezout_rank,rank))
    elif rank > bezout_rank:
        warn('Rank of Macaulay Matrix does not match the Bezout bound. Expected rank {}, found rank {}.'.format(bezout_rank,rank))

    # QR reduce the highest-degree columns
    Q,M[:,:cut] = mp.qr(M[:,:cut])
    M[:,cut:] = Q.H * M[:,cut:]
    Q = None
    del Q

    # If the matrix is "tall", compute an orthogonal transformation of the remaining
    # columns, generating a new polynomial basis
    if cut < M.rows:
        Q = mp.svd(M[cut:,cut:],full_matrices=True)[2].H
        M[:cut,cut:] = M[:cut,cut:] * Q # Apply column transform

    # Return the backsolved columns and coefficient matrix for the quotient basis
    return mp_solve_triangular(M[:cut,:cut],M[:cut,bezout_rank:]),Q[:,Q.cols-bezout_bound:]

#can't use QRP because mpmath can't do qr with pivoting
