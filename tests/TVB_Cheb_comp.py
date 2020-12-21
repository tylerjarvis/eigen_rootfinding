import eigen_rootfinding as eig_rf
import numpy as np
from time import time
from random_tests import load_tests
import pickle
from eigen_rootfinding.utils import ConditioningError

dir = 'tests/random_tests'

if __name__ == "__main__":

    dim_degs = {2: [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30],
                3: [2, 3, 4, 5, 6, 7, 8],
                4: [2, 3, 4, 5, 6],
                5: [2, 3, 4, 5]}
    dims = [2, 3, 4, 5]

    dicts = {'svd': dict(),
             'qrt': dict(),
             'tvb': dict()}

    for dim in dims:
        degs = dim_degs[dim]
        for deg in degs:
            tests = load_tests(dim, deg, "Chebyshev", "randn")
            for method in ['svd', 'qrt', 'tvb']:
                print(f'Starting dimension {dim} degree {deg} polynomials with {method}')
                start = time()
                residuals = []
                try:
                    for system in tests:
                        # Want Residuals, condition numbers, and time
                        roots, cond = eig_rf.solve(system, method=method)
                        for func in system:
                            residuals.append(func(roots))
                        end = time() - start

                        dicts[method][(dim, deg)] = {'timings': end/len(tests),
                                                     'residuals': residuals,
                                                     'condeigs': cond}

                except ConditioningError as e:
                    print(f"A Conditioning Exception occured for degree {deg} dimension {dim} polys for the {method} method.")
                    end = time() - start

                    dicts[method][(dim, deg)] = {'timings': np.nan,
                                                 'residuals': np.nan,
                                                 'condeigs': np.nan}
                except Exception as e:
                    print(f"An Exception occured for degree {deg} dimension {dim} polys for the {method} method.")
                    end = time() - start

                    dicts[method][(dim, deg)] = {'timings': np.nan,
                                                 'residuals': np.nan,
                                                 'condeigs': np.nan}

                with open(f'{dir}/{method}_results.pkl', 'wb') as ofile:
                    pickle.dump(dicts[method], ofile)
