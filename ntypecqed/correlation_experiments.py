from __future__ import print_function
from qutip import steadystate, expect, correlation_3op_1t
import numpy as np


def cross_correlation(experiment, start_time, stop_time, steps=500, flip_time_axis=False):
    """Performs a cross correlation between signal and probe light field

    :param experiment: The experiment on which the scan is performed
    :type experiment: ntypecqed.simulation.NTypeExperiment
    :param start_time: Start time of the correlation
    :type start_time: float
    :param stop_time: Stop time of the correlation
    :type stop_time: float
    :param steps: Number of steps
    :type steps: int
    :param flip_time_axis: if True the positive time direction is a signal photon first and a probe photon second
    :type flip_time_axis: bool
    :return: tuple(times, correlation value)
    """

    if start_time > 0 or stop_time < 0 or start_time >= stop_time:
        raise (ValueError, 'Wrong times, conditions are start_time<0, stop_time>0 and start_time < stop_time')

    tau_list_pos = np.linspace(0, abs(stop_time), steps)
    tau_list_neg = np.linspace(0, abs(start_time), steps)
    ss = steadystate(experiment.driven_hamiltonian, experiment.environment.c_ops)
    photon_number_field_1, photon_number_field_2 = expect(experiment.environment.n_a, ss), \
                                                   expect(experiment.environment.n_b, ss)
    corr_data_pos = correlation_3op_1t(experiment.driven_hamiltonian, ss, tau_list_pos, experiment.environment.c_ops,
                                       experiment.environment.b.dag(), experiment.environment.n_a,
                                       experiment.environment.b)
    corr_data_neg = correlation_3op_1t(experiment.driven_hamiltonian, ss, tau_list_neg, experiment.environment.c_ops,
                                       experiment.environment.a.dag(), experiment.environment.n_b,
                                       experiment.environment.a)
    # norm the correlation
    corr_data_pos /= (photon_number_field_1 * photon_number_field_2)
    corr_data_neg /= (photon_number_field_1 * photon_number_field_2)
    # change one of the correlations to negative times
    if flip_time_axis:
        tau_list_pos = -tau_list_pos[::-1]
        corr_data_pos = corr_data_pos[::-1]
        return np.concatenate((tau_list_pos, tau_list_neg)), np.concatenate((corr_data_pos, corr_data_neg))
    else:
        tau_list_neg = -tau_list_neg[::-1]
        corr_data_neg = corr_data_neg[::-1]
        return np.concatenate((tau_list_neg, tau_list_pos)), np.concatenate((corr_data_neg, corr_data_pos))


def self_correlation(experiment, stop_time, steps=500, field='probe'):
    """Returns the self correlation of one of the cavity fields

    :param experiment: The experiment on which the scan is performed
    :type experiment: ntypecqed.simulation.NTypeExperiment
    :param stop_time: Stop time of the correlation
    :type stop_time: float
    :param steps: Number of steps
    :type steps: int
    :param field: The field for which the self correlation is calculated, either 'probe' or 'signal'
    :type field: str
    :return: tuple(times, correlation value)
    """
    if field == 'probe':
        operator = experiment.environment.a
    elif field == 'signal':
        operator = experiment.environment.b
    else:
        raise (ValueError, "No valid field name, valid fields are: 'probe' or 'signal'")

    tau_list = np.linspace(0, stop_time, steps)
    ss = steadystate(experiment.driven_hamiltonian, experiment.environment.c_ops)
    n = expect(operator.dag() * operator, ss)
    corr_data = correlation_3op_1t(experiment.driven_hamiltonian, ss, tau_list, experiment.environment.c_ops,
                                   operator.dag(), operator.dag() * operator, operator)
    corr_data /= (n * n)
    return tau_list, corr_data


def triggered_self_correlation(experiment, stop_time, steps=500, trigger_photon='probe'):
    """Returns the triggered self correlation of one of the cavity fields

    :param experiment: The experiment on which the scan is performed
    :type experiment: ntypecqed.simulation.NTypeExperiment
    :param stop_time: Stop time of the correlation
    :type stop_time: float
    :param steps: Number of steps
    :type steps: int
    :param trigger_photon: The photon which starts the self correlation, either 'probe' or 'signal'
    :type trigger_photon: str
    :return: tuple(times, correlation value)
    """
    if trigger_photon == 'probe':
        operator = experiment.environment.a
        self_op = experiment.environment.b
    elif trigger_photon == 'signal':
        operator = experiment.environment.b
        self_op = experiment.environment.a
    else:
        raise (ValueError, "No valid trigger photon name, valid names are: 'probe' or 'signal'")
    tau_list = np.linspace(0, stop_time, steps)
    ss = steadystate(experiment.driven_hamiltonian, experiment.environment.c_ops)
    n_a, n_b = expect(experiment.environment.n_a, ss), expect(experiment.environment.n_b, ss)
    corr_data = correlation_3op_1t(experiment.driven_hamiltonian, ss, tau_list, experiment.environment.c_ops,
                                   operator.dag()*self_op.dag(), self_op.dag() * self_op, operator * self_op)
    corr_data /= (n_a * n_b * n_b)
    return tau_list, corr_data

def double_coincidences(experiment, normed=True):
    """Returns the value of the cross correlation at time 0

    :param experiment: The experiment on which the scan is performed
    :type experiment: ntypecqed.simulation.NTypeExperiment
    :param normed: Normalize to the power level
    :type normed: bool
    :return: correlation value
    """
    ss = steadystate(experiment.driven_hamiltonian, experiment.environment.c_ops)
    n_a, n_b = expect(experiment.environment.n_a, ss), expect(experiment.environment.n_b, ss)
    expectation_operator = experiment.environment.a.dag()*experiment.environment.b.dag()*experiment.environment.b*experiment.environment.a
    corr_data = expect(expectation_operator, ss)
    if normed:
        corr_data /= (n_a * n_b)
    return corr_data

def triple_coincidences(experiment, trigger_photon='probe', normed=True):
    """Returns the value of the triggered two photon self correlation at time 0

    :param experiment: The experiment on which the scan is performed
    :type experiment: ntypecqed.simulation.NTypeExperiment
    :param trigger_photon: The photon which starts the self correlation, either 'probe' or 'signal'
    :type trigger_photon: str
    :param normed: Normalize to the power level
    :type normed: bool
    :return: correlation value
    """
    if trigger_photon == 'probe':
        trig_op = experiment.environment.a
        self_op = experiment.environment.b
    elif trigger_photon == 'signal':
        trig_op = experiment.environment.b
        self_op = experiment.environment.a
    else:
        raise (ValueError, "No valid trigger photon name, valid names are: 'probe' or 'signal'")
    ss = steadystate(experiment.driven_hamiltonian, experiment.environment.c_ops)
    n_a, n_b = expect(experiment.environment.n_a, ss), expect(experiment.environment.n_b, ss)
    expectation_operator = trig_op.dag()*self_op.dag()*self_op.dag()*self_op*self_op*trig_op
    corr_data = expect(expectation_operator, ss)
    if normed:
        if trigger_photon == 'probe':
            corr_data /= (n_a * n_b * n_b)
        else:
            corr_data /= (n_b * n_a * n_a)
    return corr_data
