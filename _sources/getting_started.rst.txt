===============
Getting started
===============

All starts with a virtual Hilbert space where the simulations take place.
This Hilbert space is represented by the class :class:`HilbertSpace <ntypecqed.hilbertspace.HilbertSpace>` and can be used (in its simplest form)
as follows::

    from ntypecqed.hilbertspace import HilbertSpace

    hs = HilbertSpace()

Many different keyword arguments that shape the Hilbert space are accepted by the constructor of this class.
Please refer to the :class:`class documentation <ntypecqed.hilbertspace.HilbertSpace>` for more details.


.. warning:: Please note that all values are internally converted to angular frequencies by multiplication by :math:`2\pi` for convenience reasons. Please keep this in mind.


This Hilbert space is now the foundation for the simulations. The concrete implementation of a virtual experiment
is done via the :class:`NTypeExperiment <ntypecqed.simulation.NTypeExperiment>` class. Given the experimental parameters this class acts as a provider for the
different Hamiltonians (bare or driven) of the system. This is instantiated with a dict of parameters for the system::

    from ntypecqed.simulation import NTypeExperiment

    system_parameters = dict()
    system_parameters["g_p"] = 11
    system_parameters["g_s"] = 9.5
    system_parameters["eta_p"] = 0.2
    system_parameters["eta_s"] = 0.0
    system_parameters["omega_c"] = 3.0
    system_parameters["delta_31"] = 0.0
    system_parameters["delta_42"] = 0.0
    system_parameters["probe_detuning"] = 0.0
    system_parameters["control_detuning"] = 0.0
    system_parameters["signal_detuning"] = 0.0

    example_experiment = NTypeExperiment(system_parameters, environment=hs)

Now the variable `example_experiment` holds the full information for the Hamiltonians and all parameters and
is based on the provided HilbertSpace.

The Hamiltionan can i.e. be accessed by::

    print(example_experiment.driven_hamiltonian)

Parameters can be changed by::

    example_experiment['probe_detuning'] = 2.5

To simulate physical observables there are different functions, which are separated in two different modules.
One module is called `ntypecqed.transmission_experiments` and account for all experiments where the measurement quantity
is the intensity when transmitting through the system. The second kind of experiments is `ntypecqed.correlation_experiments` where diffenrent
correlation function of the photon statistics of transmitted beams are evaluated.

These can be used as follows::

    freqs_transmission, result_transmission1 = scan_laser_freq(example_experiment, -25, 25)
    freqs_transmission, result_transmission2 = scan_laser_power(example_experiment, 0, 3)
    times_correlations, result_correlation1 = cross_correlation(example_experiment, -2, 2)
    times_correlations, result_correlation2 = self_correlation(example_experiment, 0, 2)

Now these variables contain the result of the simulations and can be processed further.
A list of all available experiments can be found under :ref:`functions_label`.