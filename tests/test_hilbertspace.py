import pytest
from ntypecqed.hilbertspace import HilbertSpace


def test_hilbertspace():
    hs_1 = HilbertSpace()
    hs_2 = HilbertSpace(kappa_a=5.0)
    assert hs_1.kappa_a == 4.1
    assert hs_2.kappa_a == 5.0
    assert len(hs_1.c_ops) == 7
    assert eval(repr(hs_2)).__dict__ == hs_2.__dict__
    with pytest.raises(ValueError) as error:
        HilbertSpace(wrong_param=1.0)
