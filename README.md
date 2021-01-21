# Eigen Rootfinding

Eigen_Rootfinding is a Python package for numerical root finding. See  DemoNotebook.ipynb for a JupyterNotebook demonstration of the code's capabilities.  This project was supported in part by the National Science Foundation, grant number DMS-1564502.

<!-- [![Build Status](https://travis-ci.com/tylerjarvis/RootFinding.svg?branch=master)](https://travis-ci.com/tylerjarvis/RootFinding) -->
<!-- [![codecov](https://codecov.io/gh/mtmoncur/tylerjarvis/branch/master/graphs/badge.svg)](https://codecov.io/gh/tylerjarvis/RootFinding) -->
<!-- [![PyPI version](https://badge.fury.io/py/RootFinding.svg)](https://badge.fury.io/py/RootFinding) -->
<!-- [![Code Health](https://landscape.io/github/tylerjarvis/RootFinding/pypackage/landscape.svg)](https://landscape.io/github/tylerjarvis/RootFinding/pypackage) -->

<!-- [![Build Status](https://travis-ci.com/tylerjarvis/RootFinding.svg?branch=master)](https://travis-ci.com/tylerjarvis/RootFinding) -->
<!-- [![codecov](https://codecov.io/gh/mtmoncur/tylerjarvis/branch/master/graphs/badge.svg)](https://codecov.io/gh/tylerjarvis/RootFinding) -->
<!-- [![PyPI version](https://badge.fury.io/py/RootFinding.svg)](https://badge.fury.io/py/RootFinding) -->
<!-- [![Code Health](https://landscape.io/github/tylerjarvis/RootFinding/pypackage/landscape.svg)](https://landscape.io/github/tylerjarvis/RootFinding/pypackage) -->

### Requirements
* Python 3.6 and up

## Installation

`$ git clone https://github.com/tylerjarvis/eigen_rootfinding.git`

<!-- (We are currently working on getting a `pip` or `conda` for download) -->

Rootfinding can now be installed locally by using `pip install -e .` while inside the RootFinding folder.
The package can then by imported using `import eigen_rootfinding`.

## Usage

```python
#imports
import eigen_rootfinding as eig_rf

# Define the polynomial system (this uses an example of random polynomials)
f = eig_rf.polynomial.getPoly(5, 2, power=True)  # Degree 5, dimension 2 power polynomial
g = eig_rf.polynomial.getPoly(7, 2, power=True)  # Degree 7, dimension 2 power polynomial

#solve
eig_rf.solve([f,g])
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Build status

|             | [master](https://github.com/tylerjarvis/RootFinding/tree/master) | [develop](https://github.com/tylerjarvis/RootFinding/tree/develop) |
|-------------|--------|-----|
| Status      |  [![Build Status](https://travis-ci.com/tylerjarvis/RootFinding.svg?branch=master)](https://travis-ci.com/tylerjarvis/RootFinding)      |  [![Build Status](https://travis-ci.com/tylerjarvis/RootFinding.svg?branch=develop)](https://travis-ci.com/tylerjarvis/RootFinding)    |
| Codecov     |  [![Coverage Status](https://codecov.io/gh/mtmoncur/tylerjarvis/branch/master/graphs/badge.svg)](https://codecov.io/gh/tylerjarvis/RootFinding)  |  [![Coverage Status](https://codecov.io/gh/mtmoncur/tylerjarvis/branch/develop/graphs/badge.svg)](https://codecov.io/gh/tylerjarvis/RootFinding)   |

## License
[MIT](https://choosealicense.com/licenses/mit/)
