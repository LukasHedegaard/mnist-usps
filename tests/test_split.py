# from src import office31
from mnistusps import mnistusps


def test_resize():
    train, val, test = mnistusps(
        "mnist",
        "usps",
        num_source_per_class=20,
        num_target_per_class=3,
        image_resize=(16, 16),
        group_in_out=False,
    )
    assert train.shape == ((16, 16), (16, 16), (), ())
    assert val.shape == ((16, 16), (16, 16), (), ())
    assert test.shape == ((16, 16), (16, 16), (), ())
