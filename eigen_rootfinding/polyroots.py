import numpy as np
from eigen_rootfinding import OneDimension as oneD
from eigen_rootfinding.polynomial import is_power
from eigen_rootfinding.Macaulay import macaulay_solve
from eigen_rootfinding.Nullspace import nullspace_solve
from eigen_rootfinding.utils import match_poly_dimensions, ConditioningError

#todo docstrings EVERYWHERE
#todo random linear combinations
#todo test and decide which method to use a default
def solve(polys, verbose=False, return_all_roots=True,
          max_cond_num=1.e6, method='svdmac',randcombos=False):
    polys = match_poly_dimensions(polys)
    # Determine polynomial type and dimension of the system
    poly_type = is_power(polys, return_string = True)
    dim = polys[0].dim

    if dim == 1:
        #default to using rotated companion matrix with eigenvalues
        if method=='svdmac' or method=='multeigvals': #todo update to final default method
            MSmatrix = 0
            eigvals = True
        elif method=='diveigvals':
            MSmatrix = -1
            eigvals = True
        elif method=='multeigvecs':
            MSmatrix = 0
            eigvals = False
        elif method=='diveigvecs':
            MSmatrix = -1
            eigvals = False
        else:
            raise ValueError('invalid method type for univariate solver')
        if len(polys) == 1:
            return oneD.solve(polys[0], MSmatrix=MSmatrix, eigvals=eigvals, verbose=verbose)
        else:
            zeros = np.unique(oneD.solve(polys[0], MSmatrix=MSmatrix, eigvals=eigvals, verbose=verbose))
            #Finds the roots of each succesive polynomial and checks which roots are common.
            for poly in polys[1:]:
                if len(zeros) == 0:
                    break
                zeros2 = np.unique(oneD.solve(poly, MSmatrix=MSmatrix, eigvals=eigvals, verbose=verbose))
                common = list()
                tol = 1.e-10
                for zero in zeros2:
                    spot = np.where(np.abs(zeros-zero)<tol)
                    if len(spot[0]) > 0:
                        common.append(zero)
                zeros = common
            return zeros
    else:
        if method in {'qrpnull','lqnull','svdnull',
                      'qrpfastnull','lqfastnull','svdfastnull'}:
            if method[-8:]=='fastnull':
                if randcombos:
                    #todo verify this is true
                    raise ValueError('Cannot do random linear combinations and fast nullspace together')
                return nullspace_solve(polys, return_all_roots=return_all_roots,
                                   method=method[:-8],nullmethod='fast',randcombos=randcombos)
            else:
                return nullspace_solve(polys, return_all_roots=return_all_roots,
                               method=method[:-4],nullmethod='svd',randcombos=randcombos)
        elif method in {'qrpmac','lqmac','svdmac'}:
            res = macaulay_solve(polys, max_cond_num=max_cond_num, verbose=verbose,
                                 return_all_roots=return_all_roots, method=method[:-3],randcombos=randcombos)
            if res[0] is None:
                raise ConditioningError(res[1])
            else:
                return res
        else:
            raise ValueError('invalid method type for multivariate solver')
