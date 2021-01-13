import numpy as np
from eigen_rootfinding import OneDimension as oneD
from eigen_rootfinding.polynomial import is_power
from eigen_rootfinding.Macaulay import macaulay_solve
from eigen_rootfinding.Nullspace import nullspace_solve
from eigen_rootfinding.utils import match_poly_dimensions, ConditioningError

#todo take out references to MSmatrix andd clean this up
#todo docstrings EVERYWHERE
#todo random linear combinations
def solve(polys, MSmatrix=0, eigvals=True, verbose=False,
          return_all_roots=True, max_cond_num=1.e6,
          macaulay_zero_tol=1.e-12, method='svdmac'):
    polys = match_poly_dimensions(polys)
    # Determine polynomial type and dimension of the system
    poly_type = is_power(polys, return_string = True)
    dim = polys[0].dim

    if dim == 1:
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
        if method in {'qrpnull','lqnull','svdnull','qrpfastnull','lqfastnull','svdfastnull'}:
            if method[-8:]=='fastnull':
                return nullspace_solve(polys, return_all_roots=return_all_roots,
                                   method=method[:-8],nullmethod='fast')
            return nullspace_solve(polys, return_all_roots=return_all_roots,
                               method=method[:-4],nullmethod='svd')
        elif method in {'qrpmac','lqmac','svdmac'}:
            res = macaulay_solve(polys, max_cond_num=max_cond_num, verbose=verbose,
                                 return_all_roots=return_all_roots, method=method[:-3])
            if res[0] is None:
                raise ConditioningError(res[1])
            else:
                return res
        else:
            raise ValueError('invalid method type')
