from ntypecqed.simulation import NTypeExperiment
from ntypecqed.transmission_experiments import scan_laser_freq, scan_laser_power
from numpy.testing import assert_allclose


def test_scan_laser_freq():
    system_parameters = dict()
    system_parameters["g_p"] = 11
    system_parameters["g_s"] = 9.5
    system_parameters["eta_p"] = 0.2
    system_parameters["eta_s"] = 0.2
    system_parameters["omega_c"] = 3.0
    system_parameters["delta_31"] = 1.0
    system_parameters["delta_42"] = 2.0
    system_parameters["probe_detuning"] = -2.0
    system_parameters["control_detuning"] = -1.0
    system_parameters["signal_detuning"] = -3.0

    example_experiment = NTypeExperiment(system_parameters)
    freqs, result = scan_laser_freq(example_experiment, -25, 25, steps=20)

    expected_freqs = [-25., -22.36842105, -19.73684211, -17.10526316,
                      -14.47368421, -11.84210526, -9.21052632, -6.57894737,
                      -3.94736842, -1.31578947, 1.31578947, 3.94736842,
                      6.57894737, 9.21052632, 11.84210526, 14.47368421,
                      17.10526316, 19.73684211, 22.36842105, 25.]
    expected_transmission_1 = [0.00010248229399448757, 0.00014622767785040143, 0.00022946952020501065,
                               0.00042027424708042084,
                               0.0009694206048346799, 0.001671198963851705, 0.0005689565508516637,
                               0.00016223840627686126, 9.882318583722266e-05,
                               0.0009976702110569557, 8.828489111491531e-05, 0.0001318924350809569,
                               0.0003660036556990458, 0.0012753426057769677,
                               0.0018694864104147684, 0.0007662960649636876, 0.00036022717024007325,
                               0.00020861625017047552,
                               0.00013740198437673192, 9.817832772616037e-05]

    expected_transmission_2 = [0.003602561292616098, 0.00360243591649936, 0.0036021492270550542, 0.0036013514455110296,
                               0.003598506463891415, 0.0035924961836857503, 0.0035956921054256267,
                               0.0035973376396477574, 0.0035951927177289107, 0.0035723249584054075,
                               0.0036006447838750227, 0.0036004783479484013, 0.00359900622526823, 0.0035925018652043176,
                               0.0035917652648666206, 0.003599687748520656, 0.003601689447923688, 0.0036022684414063685,
                               0.0036024874687641964, 0.0036025869200948238]
    assert_allclose(freqs, expected_freqs)
    assert_allclose(result[0], expected_transmission_1, rtol=1e-4)
    assert_allclose(result[1], expected_transmission_2, rtol=1e-4)


def test_scan_laser_power():
    system_parameters = dict()
    system_parameters["g_p"] = 11
    system_parameters["g_s"] = 9.5
    system_parameters["eta_p"] = 0.5
    system_parameters["eta_s"] = 0.2
    system_parameters["omega_c"] = 8.0
    system_parameters["delta_31"] = 0.0
    system_parameters["delta_42"] = 0.0
    system_parameters["probe_detuning"] = 0.0
    system_parameters["control_detuning"] = 0.0
    system_parameters["signal_detuning"] = 0.0

    example_experiment = NTypeExperiment(system_parameters)
    powers, result = scan_laser_power(example_experiment, 0, 3, scan_laser='signal', steps=20)

    expected_powers = [0., 0.15789474, 0.31578947, 0.47368421, 0.63157895,
                       0.78947368, 0.94736842, 1.10526316, 1.26315789, 1.42105263,
                       1.57894737, 1.73684211, 1.89473684, 2.05263158, 2.21052632,
                       2.36842105, 2.52631579, 2.68421053, 2.84210526, 3.]
    expected_transmission_1 = [0.05490630287651778, 0.05354336881269737, 0.04971747290493745, 0.044137795937320375,
                               0.03774641396979398, 0.03144515865595226, 0.02587215865666741, 0.021319151925811252,
                               0.017792769783022234, 0.015141337017971738, 0.013166018225259205, 0.011684122941350336,
                               0.010551873386843581, 0.009665052599496883, 0.008951478693910015, 0.00836237701746908,
                               0.007865209501778058, 0.007438406690033419, 0.007067709076996032, 0.006743685116536006]
    expected_transmission_2 = [2.2789332846718487e-18, 0.009643640848001086, 0.039081537309859166, 0.08896283404836447,
                               0.15810068552042708, 0.24201285698479788, 0.3334174370486246, 0.4244595773366843,
                               0.5089874116963405, 0.5835452963480448, 0.6470846182161194, 0.7001168362645763,
                               0.7439080140815942, 0.7799394781246453, 0.8096218069483392, 0.8341771597523147,
                               0.8546130902660339, 0.8717381531347602, 0.8861922805652211, 0.8984790501018871]
    assert_allclose(powers, expected_powers)
    assert_allclose(result[0], expected_transmission_1, rtol=1e-4)
    assert_allclose(result[1], expected_transmission_2, rtol=1e-4)

