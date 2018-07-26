""" HilbertSpace Module

This modules provides the stage on which all simulations take place.
"""
# level scheme
#                         |4>---------------------
#                                      |
#  |3>--------------------             |
#              |        \              |
#              |         \             | g_s
#              |   omega_c\            | eta_s
#          g_p |   crtl_det\           | delta_42
#        eta_p |            \          | signal_det
#     delta_31 |             \         |
#    probe_det |              \        |
#              |          |2>---------------------
#              |
#  |1>--------------------

import numpy as np
from qutip import *


class HilbertSpace(object):
    """This class represents a physical system in which one can do experiments

    This class sets the stage for all the following experiments(simulations) and provides
    all needed operators and constants

    :param N_a: Size of Hilberspace in first cavity mode (3)
    :type N_a: int
    :param N_b: Size of Hilbertspace in second cavity mode (3)
    :type N_b: int
    :param kappa_a: Decay rate of first cavity mode (4.1*2*pi)
    :type kappa_a: float
    :param kappa_b: Decay rate of second cavity mode (2.9*2*pi)
    :type kappa_b: float
    :param gamma_d1: Decay rate of D1 excited state (6.07*2*pi)
    :type gamma_d1: float
    :param gamma_d2: Decay rate of D2 excited state (5.75*2*pi)
    :type gamma_d2: float
    :param gamma_dephasing: Dephasing rate between the two ground states (0.1*2*pi)
    :type gamma_dephasing: float
    """
    possible_params = ('N_a', 'N_b', 'kappa_a', 'kappa_b', 'gamma_d2', 'gamma_d1', 'dephasing', 'gamma31',
                       'gamma32', 'gamma41', 'gamma42')

    def __init__(self, **kwargs):

        for key in kwargs.keys():
            if key not in HilbertSpace.possible_params:
                raise ValueError('%s is not a possible parameter. Possible parameters are %s'
                                 % (key, str(HilbertSpace.possible_params)))
        # define constants
        self.N_a = kwargs.get('N_a', 3)
        self.N_b = kwargs.get('N_b', 3)
        self.N_atom = 4
        self.kappa_a = kwargs.get('kappa_a', 4.1)
        self.kappa_b = kwargs.get('kappa_b', 2.9)
        self.gamma_d2 = kwargs.get('gamma_d2', 6.07)
        self.gamma_d1 = kwargs.get('gamma_d1', 5.75)
        self.gamma_dephasing = kwargs.get('dephasing', 0.128)

        # Clebsch-Gordans are chosen such that they add up to 1
        # for each excited state keeping the real ratios
        self.gamma31 = kwargs.get('gamma31', (6.0 / 7.0) * self.gamma_d1)  # D1F21, mF'=-1 to mF=-2
        self.gamma32 = kwargs.get('gamma32', (1.0 / 7.0) * self.gamma_d1)  # D1F11, mF'=-1 to mF=-1
        self.gamma41 = kwargs.get('gamma41', (0.0 / 1.0) * self.gamma_d2)  # D2F21, mF'=0 to mF=-2
        self.gamma42 = kwargs.get('gamma42', (1.0 / 1.0) * self.gamma_d2)  # D2F11, mF'=0 to mF=-1

        # Define atomic states
        self.s1 = basis(self.N_atom, 0)
        self.s3 = basis(self.N_atom, 1)
        self.s2 = basis(self.N_atom, 2)
        self.s4 = basis(self.N_atom, 3)

        # operators
        # cavity annihilation
        self.a = tensor(destroy(self.N_a), qeye(self.N_b), qeye(4))
        self.b = tensor(qeye(self.N_a), destroy(self.N_b), qeye(4))

        self.n_a = self.a.dag() * self.a
        self.n_b = self.b.dag() * self.b

        # define atomic transition operators and projectors
        self.sigma_13 = tensor(qeye(self.N_a), qeye(self.N_b), self.s1 * self.s3.dag())  # |1><3|
        self.sigma_23 = tensor(qeye(self.N_a), qeye(self.N_b), self.s2 * self.s3.dag())  # |2><3|
        self.sigma_24 = tensor(qeye(self.N_a), qeye(self.N_b), self.s2 * self.s4.dag())  # |2><4|
        self.sigma_14 = tensor(qeye(self.N_a), qeye(self.N_b), self.s1 * self.s4.dag())  # |1><4|
        self.sigma_11 = tensor(qeye(self.N_a), qeye(self.N_b), self.s1 * self.s1.dag())  # |1><1|
        self.sigma_33 = tensor(qeye(self.N_a), qeye(self.N_b), self.s3 * self.s3.dag())  # |3><3|
        self.sigma_22 = tensor(qeye(self.N_a), qeye(self.N_b), self.s2 * self.s2.dag())  # |2><2|
        self.sigma_44 = tensor(qeye(self.N_a), qeye(self.N_b), self.s4 * self.s4.dag())  # |4><4|

        # decay operators
        self.c_ops = list()

        # cavity relaxation
        self.c_ops.append(np.sqrt(self.kappa_a * 2 * np.pi) * self.a)
        self.c_ops.append(np.sqrt(self.kappa_b * 2 * np.pi) * self.b)

        # decay to different levels
        self.c_ops.append(np.sqrt(self.gamma31 * 2 * np.pi) * self.sigma_13)
        self.c_ops.append(np.sqrt(self.gamma32 * 2 * np.pi) * self.sigma_23)
        self.c_ops.append(np.sqrt(self.gamma42 * 2 * np.pi) * self.sigma_24)
        self.c_ops.append(np.sqrt(self.gamma41 * 2 * np.pi) * self.sigma_14)
        if self.gamma_dephasing > 0:
            self.c_ops.append(np.sqrt(self.gamma_dephasing * 2 * np.pi) * self.sigma_22)

    def __repr__(self):
        return 'HilbertSpace(N_a=%s, N_b=%s, kappa_a=%s, kappa_b=%s, gamma_d1=%s, gamma_d2=%s, dephasing=%s)' % (
            self.N_a, self.N_b, self.kappa_a, self.kappa_b, self.gamma_d1, self.gamma_d2, self.gamma_dephasing)

    def __str__(self):
        return 'N_a=%s, N_b=%s, kappa_a=%s, kappa_b=%s, gamma_d1=%s, gamma_d2=%s, dephasing=%s' % (
            self.N_a, self.N_b, self.kappa_a, self.kappa_b, self.gamma_d1, self.gamma_d2, self.gamma_dephasing)
