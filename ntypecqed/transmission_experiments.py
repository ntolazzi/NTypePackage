from __future__ import print_function
from typing import List, Tuple, Dict
from ntypecqed.simulation import NTypeExperiment
from qutip import steadystate, expect, Qobj, mesolve, parallel_map, serial_map
import numpy as np


def ss_freq(freq, experiment, scan_laser):
    tmp_exp = experiment.copy()
    tmp_exp[scan_laser] = freq
    return steadystate(tmp_exp.driven_hamiltonian, tmp_exp.environment.c_ops)


def ss_power(power, experiment, power_scanned_laser):
    tmp_exp = experiment.copy()
    tmp_exp[power_scanned_laser] = power
    return steadystate(tmp_exp.driven_hamiltonian, tmp_exp.environment.c_ops)


def scan_laser_freq(experiment, start_freq, stop_freq, observables=None, scan_laser='probe', steps=100,
                    parallelize=False, progress_bar=True):
    """Scans the frequency of a laser and returns transmission by default or user given observables

    :param parallelize: Use multiple cores to calculate
    :type parallelize: bool
    :param experiment: The experiment on which the scan is performed
    :type experiment: ntypecqed.simulation.NTypeExperiment
    :param start_freq: Start frequency of the scan
    :type start_freq: float
    :param stop_freq: Stop frequency of the scan
    :type stop_freq: float
    :param observables: Observables for which the steadystate is calculated
    :type observables: list(qutip.operator)
    :param scan_laser: Which laser to scan, either *probe*, *signal* or *control*
    :type scan_laser: str
    :param steps: Number of steps
    :type steps: int
    :return: tuple(frequencies, list of lists of the steadystates of the observables)
    """

    freqs = np.linspace(start_freq, stop_freq, steps)
    tmp_experiment = experiment.copy()
    if scan_laser in ['signal', 'control', 'probe']:
        scan_laser += '_detuning'
    else:
        raise (KeyError, "No valid scan laser, must be one of signal, control, probe")

    if observables is None:
        observables = tmp_experiment.environment.n_a, tmp_experiment.environment.n_b
    if parallelize:
        steady_states = parallel_map(ss_freq, freqs, task_args=(experiment, scan_laser), progress_bar=progress_bar)
    else:
        steady_states = serial_map(ss_freq, freqs, task_args=(experiment, scan_laser), progress_bar=progress_bar)
    ob_results = []
    for result in steady_states:
        ob_results.append(tuple(expect(result, ob) for ob in observables))
    return freqs, list(map(list, zip(*ob_results)))


def scan_laser_power(experiment, start_power, stop_power, observables=None, scan_laser='probe', steps=100,
                     parallelize=False, progress_bar=True):
    """Scans the frequency of a laser and returns transmission by default or user given observables

    :param parallelize: Use multiple cores to calculate
    :type parallelize: bool
    :param experiment: The experiment on which the scan is performed
    :type experiment: ntypecqed.simulation.NTypeExperiment
    :param start_power: Start power of power scan
    :type start_power: float
    :param stop_power: Stop power of power scan
    :type stop_power: float
    :param observables: Observables for which the steadystate is calculated
    :type observables: list of qutip.operator
    :param scan_laser: Which laser to scan, either *probe*, *signal* or *control*
    :type scan_laser: str
    :param steps: Number of steps
    :type steps: int
    :return: tuple(powers, list of lists of the steadystates of the observables)
    """

    laser_powers = {'probe': 'eta_p', 'signal': 'eta_s', 'control': 'omega_c'}
    powers = np.linspace(start_power, stop_power, steps)
    tmp_experiment = experiment.copy()

    try:
        power_scanned_laser = laser_powers[scan_laser]
    except KeyError:
        raise (KeyError, "No valid scan laser, must be one of signal, control, probe")

    if observables is None:
        observables = tmp_experiment.environment.n_a, tmp_experiment.environment.n_b
    if parallelize:
        steady_states = parallel_map(ss_power, powers, task_args=(experiment, power_scanned_laser),
                                     progress_bar=progress_bar)
    else:
        steady_states = serial_map(ss_power, powers, task_args=(experiment, power_scanned_laser),
                                   progress_bar=progress_bar)
    ob_results = []
    for result in steady_states:
        ob_results.append(tuple(expect(result, ob) for ob in observables))
    return powers, list(map(list, zip(*ob_results)))


def pulsed_drive_gaussian(experiment: NTypeExperiment, start_time: float, stop_time: float,
                          observables: List[Qobj] = None, steps: int = 100,
                          driving_parameter: Tuple[float] = None) -> List[List[float]]:
    """Performs a time dependent simulation of a driving pulse via Master Equation

    :param experiment: The experiment on which the time dependent solver is applied
    :param start_time: Start time for the simulation
    :param stop_time: Stop time for simulation
    :param observables: List of observables for which the simulation is done
    :param steps: Number of time steps
    :param driving_parameter: Tuple with eta, sigma and t0 of gaussian pulse
    :return: Lists of bla bla
    """

    time_list = np.linspace(start_time, stop_time, steps)
    if observables is None:
        observables = experiment.environment.n_a, experiment.environment.n_b
    H0 = experiment.full_undriven_hamiltonian
    pulse = 'eta*(1/(sigma*sqrt(2*np.pi)))*exp(-0.5*((t-t0)/sigma)**2)'
    # H = [H0, [drive, pulse]]


def solve_me(experiment: NTypeExperiment, starting_state: Qobj, hamiltonian: Qobj,
             time_dependent_parameters: Dict = None, start_time: float = 0.0, stop_time: float = 20.0,
             observables: List[Qobj] = None, steps: int = 1000) -> List[List[float]]:
    time_list = np.linspace(start_time, stop_time, steps)
    if observables is None:
        observables = [experiment.environment.n_a, experiment.environment.n_b]
    res = mesolve(hamiltonian, starting_state, time_list, c_ops=experiment.environment.c_ops,
                  e_ops=observables, args=time_dependent_parameters, progress_bar=True)
    return time_list, res
