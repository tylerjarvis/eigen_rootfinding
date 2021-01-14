import numpy as np
from eigen_rootfinding.Macaulay import build_macaulay, find_degree, \
                                       create_matrix
from eigen_rootfinding.Multiplication import indexarray,indexarray_cheb,\
                                             msroots,get_rand_combos_matrix
from eigen_rootfinding.polynomial import is_power
from eigen_rootfinding.utils import row_swap_matrix, slice_top, mon_combos
from scipy import linalg as la
from scipy.special import binom

def svd_nullspace(a,nullity=None):
    """Computes the nullspace of a matrix via the singular value decomposition.

    Parameters:
    -----------
    a : 2d ndarray
        Matrix to compute the nullspace of
    nullity : int
        Nullity of matrix. Results are more reliable if the nullity of a
        is known from the start.

    Returns:
    --------
    N : 2d ndarray
        Matrix whose columns form a basis for the nullspace of a
    """
    U,S,Vh = np.linalg.svd(a)
    if nullity is None:
        #mimics np.linalg.matrix_rank
        tol = S.max()*max(a.shape)*np.finfo(S.dtype).eps
        rank = np.count_nonzero(S>tol)
    else:
        rank = a.shape[1] - nullity
    return Vh[rank:].T.conj()

def nullspace_solve(polys, return_all_roots=True,method='svd',nullmethod='svd',
                    randcombos=False):
    '''
    Finds the roots of the given list of multidimensional polynomials using
    the nullspace of the Macaulay matrix to create Moller-Stetter matrices.

    Parameters
    ----------
    polys : list of polynomial objects
        Polynomials to find the common roots of.
    return_all_roots : bool
        If True returns all the roots, otherwise just the ones in the unit box.
    method : str
        Which method to use to compute the Moller Stetter matrices from the nullspace.
        Options are 'qrp','lq','svd'.
    nullmethod : str
        Which method to use to compute the nullspace of the Macaulay matrix.
        Options are 'svd', 'fast'.
    randcombos : bool
        Whether or not to first take random linear combinations of the Macaulay matrix.
        Not allowed for fast nullspace computations (nullmethod='fast').

    returns
    -------
    roots : numpy array
        The common roots of the polynomials. Each row is a root.
    '''
    #setup
    degs = [poly.degree for poly in polys]
    dim = len(polys)
    bezout_bound = np.prod(degs)
    #compute nullspace
    if nullmethod=='svd':
        #build macaulay marix
        M,matrix_terms,cut = build_macaulay(polys)
        if randcombos:
            C = get_rand_combos_matrix(M.shape[1]-bezout_bound,M.shape[0])
            M = C@M
        nullspace = svd_nullspace(M,bezout_bound).conj().T
    elif nullmethod=='fast':
        #todo change fast_null to make it
        #  return things in the order we want later
        nullspace,matrix_terms,cut = fast_null(polys)
        srt = np.argsort(matrix_terms.sum(axis=1))[::-1]
        nullspace = nullspace[srt]
        matrix_terms = matrix_terms[srt]
        nullspace = nullspace.conj().T
    #create MS matrices
    if method=='svd': MSfunc=get_MS_svd_nullspace
    elif method=='lq': MSfunc=get_MS_lq_nullspace
    elif method=='qrp': MSfunc=get_MS_qrp_nullspace
    MS = MSfunc(nullspace,matrix_terms,cut,bezout_bound,dim,power=is_power(polys))
    #return roots
    roots = msroots(MS)
    if return_all_roots:
        return roots
    else:
        # only return roots in the unit complex hyperbox
        return roots[[np.all(np.abs(root) <= 1) for root in roots]]

def get_MS_qrp_nullspace(nullspace,matrix_terms,cut,bezout_bound,dim,power=True):
    '''
    Constructs the Moller-Stetter matrices from the nullspace of the macaulay matrix
    via a QRP of the low degree terms in the nullspace.

    Parameters
    ----------
    nullspace : 2d ndarray
        Matrix whose rows (transposed) span the nullspace of the Macaulay matrix.
    matrix_terms : 2d integer ndarray
        Array containing the ordered column labels, where the ith row contains the
        exponent/degree of the ith monomial.
    cut : int
        Number of highest degree columns in the nullspace
    bezout_bound : int
        Number of roots of the system
    dim : int
        Dimension of the system.
    power : bool
        Whether the polynomails are expressed in the Chebyshev or Power basis.

    returns
    -------
    MS : (n, n, dim) ndarray
        Array containing the nxn Möller-Stetter matrices, where the matrix
        corresponding to multiplication by x_i is MS[..., i]
    '''
    #QRP on low degree columns
    Q,R,P = la.qr(nullspace[:,cut:],pivoting=True)
    R1 = R[:,:bezout_bound]
    #reorder columns
    matrix_terms[cut:] = matrix_terms[cut:][P]
    nullspace[:,cut:] = nullspace[:,cut:][:,P]
    #reduce columns we need for MS matrices
    cut2 = cut+bezout_bound
    basiscols = slice(cut,cut2)
    if power:
        idx_arrs = [indexarray(matrix_terms,basiscols,i) for i in range(dim)]
    else:
        idx_arrs = [indexarray_cheb(matrix_terms,basiscols,i) for i in range(dim)]
    needcols = np.unique(idx_arrs) #all the cols we need to reduce
    needhighdeg = needcols[needcols<cut] #high deg cols to reduce
    #left multiply by Q.H to get [Q.HN R1 R2]
    nullspace[:,needhighdeg] = Q.T.conj()@nullspace[:,needhighdeg]
    nullspace[:,cut2:] = R[:,bezout_bound:]
    #left multiply by R1inv to get [R1invQ.HN I R1invR2]
    nullspace[:,basiscols] = np.eye(bezout_bound)
    neednonbasis = np.concatenate((needhighdeg,needcols[needcols>=cut2]))
    nullspace[:,neednonbasis] = la.solve_triangular(R1,nullspace[:,neednonbasis])
    #construct MS matrices
    MS = np.empty((bezout_bound, bezout_bound, dim))
    for i,arr in enumerate(idx_arrs):
        if power:
            MS[...,i] = nullspace[:,arr]
        else:
            MS[...,i] = (nullspace[:,arr[0]] + nullspace[:,arr[1]])/2
    return MS

def get_MS_lq_nullspace(nullspace,matrix_terms,cut,bezout_bound,dim,power=True):
    '''
    Constructs the Moller-Stetter matrices from the nullspace of the macaulay matrix
    via a LQ of the low degree terms in the nullspace.

    Parameters
    ----------
    nullspace : 2d ndarray
        Matrix whose rows (transposed) span the nullspace of the Macaulay matrix.
    matrix_terms : 2d integer ndarray
        Array containing the ordered column labels, where the ith row contains the
        exponent/degree of the ith monomial.
    cut : int
        Number of highest degree columns in the nullspace
    bezout_bound : int
        Number of roots of the system
    dim : int
        Dimension of the system.
    power : bool
        Whether the polynomails are expressed in the Chebyshev or Power basis.

    returns
    -------
    MS : (n, n, dim) ndarray
        Array containing the nxn Möller-Stetter matrices, where the matrix
        corresponding to multiplication by x_i is MS[..., i]
    '''
    #QRP on low degree columns transposed
    Q,R,P = la.qr(nullspace[:,cut:].conj().T,pivoting=True)
    L = R[:bezout_bound].conj().T
    Q = Q[:,:bezout_bound]
    #rearrange rows in nullspace
    nullspace = nullspace[P]
    #reduce columns we need for MS matrices
    nullspace[:,:cut] = la.solve_triangular(L,nullspace[:,:cut],lower=True)
    nullspace[:,cut:] = Q.conj().T
    #construct MS matrices
    MS = np.empty((bezout_bound, bezout_bound, dim))
    lowdeg = slice(cut,None)
    for i in range(dim):
        if power:
            arr = indexarray(matrix_terms,lowdeg,i)
            MS[...,i] = nullspace[:,arr]@Q
        else:
            arr = indexarray_cheb(matrix_terms,lowdeg,i)
            MS[...,i] = (nullspace[:,arr[0]] + nullspace[:,arr[1]])@Q/2
    return MS

def get_MS_svd_nullspace(nullspace,matrix_terms,cut,bezout_bound,dim,power=True):
    '''
    Constructs the Moller-Stetter matrices from the nullspace of the macaulay matrix
    via a SVD of the low degree terms in the nullspace.

    Parameters
    ----------
    nullspace : 2d ndarray
        Matrix whose rows (transposed) span the nullspace of the Macaulay matrix.
    matrix_terms : 2d integer ndarray
        Array containing the ordered column labels, where the ith row contains the
        exponent/degree of the ith monomial.
    cut : int
        Number of highest degree columns in the nullspace
    bezout_bound : int
        Number of roots of the system
    dim : int
        Dimension of the system.
    power : bool
        Whether the polynomails are expressed in the Chebyshev or Power basis.

    returns
    -------
    MS : (n, n, dim) ndarray
        Array containing the nxn Möller-Stetter matrices, where the matrix
        corresponding to multiplication by x_i is MS[..., i]
    '''
    #SVD on low degree columns
    U,S,Vh = la.svd(nullspace[:,cut:])
    V = Vh[:bezout_bound].conj().T
    S = S[:bezout_bound]
    #reduce columns we need for MS matrices
    #left multiply by U
    nullspace[:,:cut] = U.conj().T@nullspace[:,:cut]
    nullspace[:,:cut] = (nullspace[:,:cut].T/S).T #transposed for broadcasting
    nullspace[:,cut:] = V.conj().T
    #construct MS matrices
    MS = np.empty((bezout_bound, bezout_bound, dim))
    lowdeg = slice(cut,None)
    for i in range(dim):
        if power:
            arr = indexarray(matrix_terms,lowdeg,i)
            MS[...,i] = nullspace[:,arr]@V
        else:
            arr = indexarray_cheb(matrix_terms,lowdeg,i)
            MS[...,i] = (nullspace[:,arr[0]] + nullspace[:,arr[1]])@V/2
    return MS

def all_shifts(polys, matrix_degree):
    shifts = dict()
    min_degree = np.min([poly.degree for poly in polys])
    for i in range(min_degree,matrix_degree+1):
        shifts[i] = list()
    for poly in polys:
        degree = matrix_degree - poly.degree
        dim = poly.dim
        mons = mon_combos([0]*dim,degree)
        for mon in mons:
            shifts[np.sum(mon)+poly.degree].append(tuple([poly,mon]))
    return shifts

#TODO DOCSTRINGS for this
def new_terms(coeffs, old_term_set):
    new_term_set = set()
    for coeff in coeffs:
        for term in zip(*np.where(coeff!=0)):
            if term not in old_term_set:
                new_term_set.add(tuple(term))
    if len(new_term_set)==0:
        return None
    else:
        return np.vstack(tuple(new_term_set))

def null_reduce(N,shifts,old_matrix_terms,bigShape):
    old_term_set = set()
    for term in old_matrix_terms:
        old_term_set.add(tuple(term))
    coeffs = list()
    for poly,shift in shifts:
        coeffs.append(poly.mon_mult(shift,returnType = 'Matrix'))

    new_matrix_terms = new_terms(coeffs, old_term_set)

    matrix_terms = np.vstack([old_matrix_terms, new_matrix_terms])

    new_matrix_term_indexes = list()
    old_matrix_term_indexes = list()
    for row in new_matrix_terms.T:
        new_matrix_term_indexes.append(row)
    for row in old_matrix_terms.T:
        old_matrix_term_indexes.append(row)

    #Adds the poly_coeffs to flat_polys, using added_zeros to make sure every term is in there.
    added_zeros = np.zeros(bigShape)
    new_flat_polys = list()
    old_flat_polys = list()
    for coeff in coeffs:
        slices = slice_top(coeff.shape)
        added_zeros[tuple(slices)] = coeff
        new_flat_polys.append(added_zeros[tuple(new_matrix_term_indexes)])
        old_flat_polys.append(added_zeros[tuple(old_matrix_term_indexes)])
        added_zeros[tuple(slices)] = np.zeros_like(coeff)

    R1 = np.reshape(old_flat_polys, (len(old_flat_polys),len(old_matrix_terms)))
    R2 = np.reshape(new_flat_polys, (len(new_flat_polys),len(new_matrix_terms)))

    X = np.hstack([R1@N,R2])
    #TODO can we know nullity analytically without computing?
    K = svd_nullspace(X)

    cut = N.shape[1]
    K1 = K[:cut]
    K2 = K[cut:]
    N = np.vstack([N@K1,K2])

    return N, matrix_terms

def fast_null(polys):
    matrix_degree = find_degree(polys)
    dim = polys[0].dim
    bigShape = [matrix_degree+1]*dim

    shifts = all_shifts(polys, matrix_degree)
    degs = list(shifts.keys())

    initial_shifts = shifts[degs[0]]
    initial_coeffs = list()
    for poly,shift in initial_shifts:
        initial_coeffs.append(poly.mon_mult(shift,returnType = 'Matrix'))

    matrix, matrix_terms, cut = create_matrix(initial_coeffs, degs[0], dim)

    #TODO can we know nullity analytically without computing?
    # there's a theorem for that
    N = svd_nullspace(matrix)

    spot = 1
    while spot < len(degs):
        deg = degs[spot]
        new_shifts = shifts[deg]
        N,matrix_terms = null_reduce(N,new_shifts,matrix_terms,bigShape)
        spot += 1
    return N, matrix_terms, int(binom(dim+matrix_degree-1,matrix_degree))
