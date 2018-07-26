============
Installation
============
The package should be compatible with Python 2 and Python 3.
If you just want to use the framework to do you own simulations it is enough to just install the package
::

    python setup.py

If you are interested in changing some of the internals (like the Hamiltonian) or if you want to make further
developments based on the package, it is best to clone the GitHub repository and afterwards install the package via
::

    python setup.py develop

Then all the changes in the internal files directly take place when using the package.

Dependencies
============
Mandatory
---------
- `QuTiP <http://qutip.org/>`_
- `Numpy <http://www.numpy.org/>`_

Optional
--------
The following packages are needed if you want to execute all examples and the tests:

- `Matplotlib <https://matplotlib.org/>`_ (for the plotting in the examples)
- `pytest <https://docs.pytest.org/en/latest/>`_ (for testing)

