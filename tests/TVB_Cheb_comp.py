import eigen_rootfinding as eig_rf
import numpy as np
from time import time
from random_tests import load_tests
import pickle

dir = 'tests/random_tests'

if __name__ == "__main__":

    dim_degs = {2: [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30],
                3: [2, 3, 4, 5, 6, 7, 8, 9, 10],
                4: [2, 3, 4, 5, 6, 7, 8],
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
                start = time()
                residuals = []
                for system in tests:
                    # Want Residuals, condition numbers, and time
                    roots, cond = eig_rf.solve(system)
                    f, g = system
                    residuals.append(f(roots))
                    residuals.append(g(roots))

                end = time() - start

                dicts[method][(dim, deg)] = {'timings': end/len(tests),
                                             'residuals': residuals,
                                             'condeigs': cond}
                with open(f'{dir}/{method}_results.pkl', 'wb') as ofile:
                    pickle.dump(dicts[method], ofile)
