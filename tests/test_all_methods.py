import sys
import os
import random_tests
import eigen_rootfinding as er
import numpy as np
import pickle
from time import time
from eigen_rootfinding.utils import ConditioningError


# Valid methods to pass into er.solve
methods = {'qrpnull', 'lqnull', 'svdnull', 'qrpfastnull',
           'lqfastnull', 'svdfastnull', 'qrpmac', 'lqmac', 'svdmac'}

# Fast nullspace methods (cannot include )
fast_null = {'qrpfastnull', 'lqfastnull', 'svdfastnull'}

# The results of what we are looking for
results_template = {'abs_residuals': [],  # Size fo the absolute residuals
                    'rel_residuals': [],  # Residuals as computed by Telen (relative)
                    'condeigs': [],  # Condition number of the eigenvalues
                    'timings': []}  # Average time it takes to solve each system

# Build our results dicitonaries
results = {method: dict() for method in methods}
results.update({method + '_fm_normal': dict() for method in methods.difference(fast_null)})
results.update({method + '_fm_ortho': dict() for method in methods.difference(fast_null)})

# Temporary testing for condition number stuff
results = {'svdmac_fm_normal': dict(),
           'svdmac_fm_ortho': dict(),
           'svdmac': dict(),
           'svdnull_fm_normal': dict(),
           'svdnull_fm_ortho': dict(),
           'svdnull': dict()}

dir = 'tests/variation_tests'


def relative_residual(poly, root):
    """Computes the relative residual for a given root of a polynomial.

    Parameters
    ----------
        poly : polynomial object
            The polynomial to evaluate at the root.
        root : ndarray
            The root to calculate the relative residual of.
    
    Returns
    -------
        float
            The relative residual of the root.
    """
    polyEval = abs(poly(root))[0]
    absCoeff = np.abs(poly.coeff)
    absPoly = er.MultiPower(absCoeff) if isinstance(poly, er.MultiPower) else er.MultiCheb(absCoeff)
    denom = absPoly(np.abs(root))[0] + 1
    return polyEval / denom


if __name__ == "__main__":
    dim_degs = {#2: np.arange(2, 51),
                #3: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                3: [8, 9],
                4: [2, 3, 4] #, 5],
                # 5: [2, 3],
                # 6: [2],
                # 7: [2]
                }
    # dims = [2, 3, 4, 5, 6, 7]

    dims = [3, 4]
    
    for method in results.keys():
        os.system(f"mkdir -p {dir}/{method}/")

    if len(sys.argv) > 1:
        # Specify dimension only
        if len(sys.argv) >= 2:
            dims = [int(sys.argv[1])]

        # Specify the specific degree, too
        if len(sys.argv) == 3:
            dim_degs = {dims[0]: [int(sys.argv[2])]}

    # Random power polynomial tests
    for dim in dims:
        for method in results.keys():
            results[method][dim] = dict()
        for deg in dim_degs[dim]:
            tests = random_tests.load_tests(dim, deg, "power", "randn")
            print(f"Starting dimension {dim} degree {deg} polynomials")
            print("==================================================")
            for method in results.keys():
                print(f"Starting with method {method}.")
                # Flush so that the printed statements show up on time
                sys.stdout.flush()
                
                randcombos = '_fm' in method
                normal = '_fm_normal' in method
                method2 = method

                # Remove the extra tail not supported by polyroots
                if randcombos:
                    method2 = method[:method.index('_')]

                results[method][dim][deg] = {'abs_residuals': [],  # Size fo the absolute residuals
                                             'rel_residuals': [],  # Residuals as computed by Telen (relative)
                                             'condeigs': [],  # Condition number of the eigenvalues
                                             'timings': []}  # Average time it takes to solve each system
                for system in tests:
                    abs_residuals = []
                    rel_residuals = []
                    try:
                        start = time()
                        roots, condeigs = er.solve(system, method=method2,
                                                   randcombos=randcombos,
                                                   normal=normal)
                        end = time() - start

                        for poly in system:
                            abs_residuals.append(np.abs(poly(roots)))
                            rel_residuals.append([relative_residual(poly, root) for root in roots])

                        results[method][dim][deg]['timings'].append(end)
                        results[method][dim][deg]['abs_residuals'].append(abs_residuals)
                        results[method][dim][deg]['rel_residuals'].append(rel_residuals)
                        results[method][dim][deg]['condeigs'].append(condeigs)

                    # except ConditioningError:
                    except Exception:  # Any exception, including SVD not converging.
                        # Dim 3, degree 7 system number 31 tends to have conditioning
                        # errors (for testing purposes)
                        results[method][dim][deg]['timings'].append(np.nan)
                        results[method][dim][deg]['abs_residuals'].append([np.nan])
                        results[method][dim][deg]['rel_residuals'].append([np.nan])
                        results[method][dim][deg]['condeigs'].append([np.nan])

                    finally:
                        with open(f'{dir}/{method}/dim_{dim}_deg_{deg}_polys.pkl', 'wb') as ofile:
                            pickle.dump(results[method], ofile)

