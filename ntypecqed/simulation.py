from copy import deepcopy
import pickle
import numpy as np
from ntypecqed.hilbertspace import HilbertSpace


class NTypeExperiment(object):
    """This is the basic simulation class and describes one specific set of parameters

    :param system_parameters: The dictionary with all relevant experimental parameters. 
        Needed parameters are: g_p, g_s, eta_p, eta_s, omega_c, delta_31, delta_42, probe_detuning, signal_detuning, 
        control_detuning.
    :type system_parameters: dict
    :param environment: The HilbertSpace in which the simulations take place
    :type environment: ntypecqed.hilbertspace.HilbertSpace
    :param driving: The driving of probe and signal beams, defaults to cavity drive
    :type driving: dict
    """

    necessary_params = ('g_p', 'g_s', 'eta_p', 'eta_s', 'omega_c', 'delta_31', 'delta_42', 'probe_detuning',
                        'signal_detuning', 'control_detuning')

    def __init__(self, system_parameters, environment=HilbertSpace(), driving=None):
        self.driving = deepcopy(driving)
        if driving is None:
            driving = {'probe': 'c', 'signal': 'c'}
        try:
            driving_probe = driving['probe']
            driving_signal = driving['signal']
        except KeyError:
            raise KeyError("Parameter driving needs a dict with the keys 'probe' and 'signal'")

        if driving_probe not in ['c', 'a'] or driving_signal not in ['c', 'a']:
            raise KeyError("Allowed values for driving are 'c' for cavity drive and 'a' for atom drive")

        self.driving_probe = driving_probe
        self.driving_signal = driving_signal
        if not all(param in system_parameters for param in NTypeExperiment.necessary_params):
            raise KeyError("Please provide all of the following parameters in a dict: %s"
                           % str(NTypeExperiment.necessary_params))
        self.system_parameters = deepcopy(system_parameters)
        self.environment = environment

    def __setitem__(self, key, item):
        """Simplifies the setting of parameters for existing instances"""

        if key in self.system_parameters:
            self.system_parameters[key] = item
        else:
            raise KeyError('%s is no simulation parameter' % key)

    def __getitem__(self, key):
        """Get a parameter through a dictionary like syntax"""

        return self.system_parameters[key]

    def __str__(self):
        """Present a human readable representation of the current instance"""

        string = ''
        string += 'Parameters: '
        for param in NTypeExperiment.necessary_params[:-1]:
            string += '%s=%s, ' % (param, self.system_parameters[param])
        last_param = NTypeExperiment.necessary_params[-1]
        string += '%s=%s\n' % (last_param, self.system_parameters[last_param])
        string += 'HilbertSpace: %s' % self.environment
        return string

    @staticmethod
    def load(filename):
        """Loads a saved instance of NTypeExperiment

            :param filename: Filename/Path of the saved instance
            :type filename: str
        """

        with open(filename + '.pkl', 'rb') as fh:
            return pickle.load(fh)

    def save(self, filename):
        """Saves the current instance of NTypeExperiment

            :param filename: Filename/Path of the saved instance
            :type filename: str
        """

        with open(filename + '.pkl', 'wb') as fh:
            pickle.dump(self, fh)

    def copy(self):
        """Returns a copy of this instance of NTypeExperiment for further usages

        :return: A deep copy of the NTypeExperiment
        :rtype: NTypeExperiment
        """
        return NTypeExperiment(deepcopy(self.system_parameters), environment=deepcopy(self.environment),
                               driving=deepcopy(self.driving))

    @property
    def undriven_hamiltonians(self):
        """Property that returns the bare and the interaction Hamiltonian for the current parameters

        :return: bare and interaction Hamiltonian in tuple (h_bare, h_inter, h_control)
        :rtype: tuple(qutip.Qobj)
        """

        g_p, g_s, eta_p, eta_s, omega_c, delta_31, delta_42, probe_detuning, control_detuning, signal_detuning = [
            self.system_parameters[key] * 2 * np.pi for key in
            ['g_p', 'g_s', 'eta_p', 'eta_s', 'omega_c', 'delta_31', 'delta_42', 'probe_detuning',
             'control_detuning', 'signal_detuning']]

        h_atom = (-control_detuning + delta_31 + probe_detuning) * self.environment.sigma_11 + \
                 (-control_detuning) * self.environment.sigma_33 + \
                 (-signal_detuning - delta_42) * self.environment.sigma_44
        h_cavity = - probe_detuning * self.environment.a.dag() * self.environment.a + \
                   - signal_detuning * self.environment.b.dag() * self.environment.b
        h_bare = h_atom + h_cavity
        h_inter = g_p * (self.environment.a * self.environment.sigma_13.dag() +
                         self.environment.a.dag() * self.environment.sigma_13) + \
                  g_s * (self.environment.b * self.environment.sigma_24.dag() +
                         self.environment.b.dag() * self.environment.sigma_24)
        h_control = omega_c * (self.environment.sigma_23 + self.environment.sigma_23.dag())
        return h_bare, h_inter, h_control

    @property
    def full_undriven_hamiltonian(self):
        """Property that returns the full undriven Hamiltonian for the current parameters

        :return: The undriven but otherwise complete system's Hamiltonian including drive
        :rtype: qutip.QObj
        """
        h_bare, h_inter, h_control = self.undriven_hamiltonians
        return h_bare + h_inter + h_control

    @property
    def driven_hamiltonian(self):
        """Property that returns the full system Hamiltonian for the current parameters
        
        :return: The full system's Hamiltonian including drive
        :rtype: qutip.QObj
        """
        g_p, g_s, eta_p, eta_s, omega_c, delta_31, delta_42, probe_detuning, control_detuning, signal_detuning = [
            self.system_parameters[key] * 2 * np.pi for key in
            ['g_p', 'g_s', 'eta_p', 'eta_s', 'omega_c', 'delta_31', 'delta_42', 'probe_detuning',
             'control_detuning', 'signal_detuning']]

        h_bare, h_inter, h_control = self.undriven_hamiltonians

        if self.driving_probe is 'c':
            probe_drive = self.environment.a.dag() + self.environment.a
        else:
            probe_drive = self.environment.sigma_13 + self.environment.sigma_13.dag()
        if self.driving_signal is 'c':
            signal_drive = self.environment.b.dag() + self.environment.b
        else:
            signal_drive = self.environment.sigma_24 + self.environment.sigma_24.dag()

        h_drive = eta_p * probe_drive + eta_s * signal_drive
        return h_bare + h_inter + h_drive + h_control

    @property
    def eigenstates(self):
        """Property that returns the unsorted Eigenenergies and Eigenstates of the undriven system

        :return: The Eigenenergies and Eigenvectors of the undriven system
        :rtype: tuple(eigenvalues, eigenvectors)
        """
        return self.full_undriven_hamiltonian.eigenstates()

    @property
    def sorted_eigenenergies(self):
        """Property that returns the sorted Eigenenergies of the system.

        :return: An 2d array with sorted Eigenenergies of the form sorted_array[n_p, n_s] with signal 
                 and probe photon number
        :rtype: 2d array
        """
        probe_index_step = self.environment.N_atom * self.environment.N_b
        ordered_states = [[[] for x in range(self.environment.N_b)] for y in range(self.environment.N_a)]
        for eig_val, eig_vec in zip(*self.eigenstates):
            indices = np.nonzero(eig_vec.full())[0]
            for index in indices:
                n_p = int(index / probe_index_step)
                if n_p > 0:
                    n_s_ind = index % (n_p * probe_index_step)
                else:
                    n_s_ind = index % probe_index_step
                if n_s_ind % self.environment.N_atom == 0:
                    n_s = int(n_s_ind / self.environment.N_atom)
                    ordered_states[n_p][n_s].append(eig_val / (2 * np.pi))
        return ordered_states
