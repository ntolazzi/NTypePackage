from ntypecqed.simulation import NTypeExperiment
from ntypecqed.correlation_experiments import cross_correlation, self_correlation
from numpy.testing import assert_allclose


def test_cross_correlation():
    system_parameters = dict()
    system_parameters["g_p"] = 11
    system_parameters["g_s"] = 9.5
    system_parameters["eta_p"] = 0.4
    system_parameters["eta_s"] = 0.2
    system_parameters["omega_c"] = 6.0
    system_parameters["delta_31"] = 0.0
    system_parameters["delta_42"] = 0.0
    system_parameters["probe_detuning"] = 0.0
    system_parameters["control_detuning"] = 0.0
    system_parameters["signal_detuning"] = 0.0

    example_experiment = NTypeExperiment(system_parameters)
    freqs, result = cross_correlation(example_experiment, -2, 2, steps=20)

    expected_freqs = [-2., -1.89473684, -1.78947368, -1.68421053, -1.57894737,
                      -1.47368421, -1.36842105, -1.26315789, -1.15789474, -1.05263158,
                      -0.94736842, -0.84210526, -0.73684211, -0.63157895, -0.52631579,
                      -0.42105263, -0.31578947, -0.21052632, -0.10526316, -0.,
                      0., 0.10526316, 0.21052632, 0.31578947, 0.42105263,
                      0.52631579, 0.63157895, 0.73684211, 0.84210526, 0.94736842,
                      1.05263158, 1.15789474, 1.26315789, 1.36842105, 1.47368421,
                      1.57894737, 1.68421053, 1.78947368, 1.89473684, 2.]
    expected_correlations = [0.99993557 + 0.j, 0.99991693 + 0.j, 0.99986206 + 0.j, 0.99979364 + 0.j,
                             0.99969029 + 0.j, 0.99956057 + 0.j, 0.99936930 + 0.j, 0.99908206 + 0.j,
                             0.99866195 + 0.j, 0.99806245 + 0.j, 0.99716982 + 0.j, 0.99578161 + 0.j,
                             0.99344483 + 0.j, 0.98909764 + 0.j, 0.97993880 + 0.j, 0.95864573 + 0.j,
                             0.90597578 + 0.j, 0.77511853 + 0.j, 0.47708228 + 0.j, 0.03676378 + 0.j,
                             0.03676378 + 0.j, 0.12388339 + 0.j, 0.31794529 + 0.j, 0.48181659 + 0.j,
                             0.61795296 + 0.j, 0.72543200 + 0.j, 0.80526852 + 0.j, 0.86324383 + 0.j,
                             0.90466378 + 0.j, 0.93387119 + 0.j, 0.95432092 + 0.j, 0.96853616 + 0.j,
                             0.97838206 + 0.j, 0.98515430 + 0.j, 0.98982813 + 0.j, 0.99304255 + 0.j,
                             0.99523819 + 0.j, 0.99674103 + 0.j, 0.99776668 + 0.j, 0.99846928 + 0.j]
    assert_allclose(freqs, expected_freqs, rtol=1e-4)
    assert_allclose(result, expected_correlations, rtol=1e-4)


def test_self_correlation():
    system_parameters = dict()
    system_parameters["g_p"] = 11
    system_parameters["g_s"] = 9.5
    system_parameters["eta_p"] = 0.2
    system_parameters["eta_s"] = 0.2
    system_parameters["omega_c"] = 3.0
    system_parameters["delta_31"] = 0.0
    system_parameters["delta_42"] = 0.0
    system_parameters["probe_detuning"] = 12.5
    system_parameters["control_detuning"] = 0.0
    system_parameters["signal_detuning"] = 0.0

    example_experiment = NTypeExperiment(system_parameters)
    freqs, result_probe = self_correlation(example_experiment, 2, steps=20, field='probe')
    _, result_signal = self_correlation(example_experiment, 2, steps=20, field='signal')

    expected_freqs = [0., 0.10526316, 0.21052632, 0.31578947, 0.42105263,
                      0.52631579, 0.63157895, 0.73684211, 0.84210526, 0.94736842,
                      1.05263158, 1.15789474, 1.26315789, 1.36842105, 1.47368421,
                      1.57894737, 1.68421053, 1.78947368, 1.89473684, 2.]

    expected_correlations_probe = [0.60925398 + 0.00000000e+00j, 0.64446492 - 7.32366060e-11j,
                                   0.89245198 + 6.95411475e-10j, 0.98525430 - 1.00115380e-08j,
                                   0.90535706 - 1.65945730e-08j, 0.93164471 - 4.61875257e-09j,
                                   0.99810073 + 3.53295437e-09j, 0.95555689 - 9.99006400e-09j,
                                   0.94996060 + 2.33921725e-08j, 0.99799319 - 3.51579729e-08j,
                                   0.98283165 + 2.18928764e-08j, 0.96661069 - 1.88689782e-08j,
                                   0.99568820 + 5.94834881e-08j, 0.99553840 - 3.78230333e-08j,
                                   0.97972187 + 4.07606921e-08j, 0.99428544 - 2.09818798e-08j,
                                   1.00030556 + 5.05881961e-08j, 0.98891312 - 6.18797872e-08j,
                                   0.99427521 + 4.38604691e-08j, 1.00131922 - 4.36222466e-08j]
    expected_correlations_signal = [0.97626066 + 0.00000000e+00j, 0.98635155 - 2.99312684e-13j,
                                    0.99551888 - 2.20551006e-11j, 0.99951349 - 1.75969337e-11j,
                                    1.00089791 - 2.67736178e-11j, 1.00126515 + 2.33598677e-11j,
                                    1.00126139 - 2.06365657e-12j, 1.00111497 - 2.41132936e-11j,
                                    1.00095432 - 2.40317999e-11j, 1.00081263 + 1.64006459e-11j,
                                    1.00067502 + 2.22999599e-11j, 1.00055966 + 1.70055689e-11j,
                                    1.00047136 + 2.23411286e-12j, 1.00039036 + 2.20624508e-11j,
                                    1.00032148 + 1.14016600e-11j, 1.00027061 + 1.08369187e-11j,
                                    1.00022486 + 1.55231786e-11j, 1.00018440 + 1.86524357e-11j,
                                    1.00015506 + 2.64082369e-11j, 1.00012950 + 2.52681107e-11j]
    assert_allclose(freqs, expected_freqs, rtol=1e-4)
    assert_allclose(result_probe, expected_correlations_probe, rtol=1e-4)
    assert_allclose(result_signal, expected_correlations_signal, rtol=1e-4)

