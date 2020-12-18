"""Run tests of the eigen rootfinder on the Chebfun Test Suite using
different solvers. Compare TVB, SVD, and QRT solving methods.
"""
import numpy as np
import eigen_rootfinding as eig_rf
import pickle
from glob import glob
import sys
import time
from eigen_rootfinding.utils import ConditioningError


# Directory to get the coeffs from
dir = "tests/Chebfun_Poly_Coeff/"
num_loops = 5

# Names of all the Chebfun tests
test_names = ['1_1', '1_2', '1_3', '1_4', '1_5', '2_1', '2_2', '2_3',
              '2_4', '2_5', '3_1', '3_2', '4_1', '4_2', '5', '6_1',
              '6_2', '6_3', '7_1', '7_2', '7_3', '7_4', '8_1', '8_2',
              '9_1', '9_2', '10']


if __name__ == "__main__":
    # Decide what type of polynomials to use
    if len(sys.argv) == 2:
        poly_type = sys.argv[1]
        assert poly_type == 'MultiCheb' or poly_type == 'MultiPower', "Polytype must be 'MultiCheb' or 'MultiPower'"
    else:
        poly_type = 'MultiCheb'

    TVB_results = {'timings': [], 'residuals': [], 'test_num': []}
    SVD_results = {'timings': [], 'residuals': [], 'test_num': []}
    QRT_results = {'timings': [], 'residuals': [], 'test_num': []}

    for name in test_names:
        print(f'Running tests for Test {name}.')

        # Get our functions to solve over
        test_strs = glob(f'{dir}{poly_type}/Test{name}*')

        if len(test_strs) == 0:
            print(f'Test {name} does not have an existing coeff file in {dir}{poly_type}.')
            continue

        if name == '1_2':
            print(f"Skipping Test {name}.")
            continue

        funcs = [np.load(file, allow_pickle=True) for file in test_strs]
        problem = False
        for func in funcs:
            if len(func.shape) == 0:
                print(f'Test {name} has a coeff matrix that is empty.')
                problem = True

        if problem:
            continue

        if poly_type == 'MultiCheb':
            funcs = [eig_rf.MultiCheb(func) for func in funcs]
        else:
            funcs = [eig_rf.MultiPower(func) for func in funcs]



        # a = -np.ones(2)
        # b = np.ones(2)

        # # Cover the tests that have weird bounds
        # if name == '7.3':
        #     a = -1e-9 * a
        #     b = 1e-9 * b
        # elif name == '5':
        #     a *= 2
        #     b *= 2
        # elif name == '2.5':
        #     a *= 4
        #     b *= 4

        # Time solver using TVB
        print('Sovling with TVB.')
        start = time.time()

        try:
            for _ in range(num_loops):
                roots = eig_rf.solve(funcs, method='tvb')
            end = time.time() - start
            roots = eig_rf.solve(funcs, method='tvb')
            TVB_results['timings'].append(end/num_loops)
            TVB_results['residuals'].append([funcs[0](roots), funcs[1](roots)])

        except ConditioningError as CE:
            print(CE.message)
            print("Conditioning Error with the Macaulay Matrix.")
            TVB_results['timings'].append(np.inf)
            TVB_results['residuals'].append(np.inf)

        with open(f'{dir}/TVB_results_{poly_type}.pkl', 'wb') as ofile:
            pickle.dump(TVB_results, ofile)

        # Time solver using SVD
        print('Sovling with SVD.')
        start = time.time()

        try:
            for _ in range(num_loops):
                roots = eig_rf.solve(funcs, method='svd')
            end = time.time() - start
            roots = eig_rf.solve(funcs, method='svd')
            SVD_results['timings'].append(end/num_loops)
            SVD_results['residuals'].append([funcs[0](roots), funcs[1](roots)])

        except ConditioningError as CE:
            print(CE.message)
            print("Conditioning Error with the Macaulay Matrix.")
            SVD_results['timings'].append(np.inf)
            SVD_results['residuals'].append(np.inf)

        with open(f'{dir}/SVD_results_{poly_type}.pkl', 'wb') as ofile:
            pickle.dump(SVD_results, ofile)

        # Time solver using QRT
        print('Sovling with QRT.')
        start = time.time()

        try:
            for _ in range(num_loops):
                roots = eig_rf.solve(funcs, method='qrt')
            end = time.time() - start
            roots = eig_rf.solve(funcs, method='qrt')
            QRT_results['timings'].append(end/num_loops)
            QRT_results['residuals'].append([funcs[0](roots), funcs[1](roots)])

        except ConditioningError as CE:
            print(CE.message)
            print("Conditioning Error with the Macaulay Matrix.")
            QRT_results['timings'].append(np.inf)
            QRT_results['residuals'].append(np.inf)

        # Record the test numbers that have entries in the dictionary.
        TVB_results['test_num'].append(name)
        SVD_results['test_num'].append(name)
        QRT_results['test_num'].append(name)

        with open(f'{dir}/QRT_results_{poly_type}.pkl', 'wb') as ofile:
            pickle.dump(QRT_results, ofile)
