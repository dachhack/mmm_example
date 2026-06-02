"""Properties of the core MMM transforms (adstock + Hill saturation)."""
import numpy as np

from draftzone_mmm.transforms import (
    delayed_adstock,
    geometric_adstock,
    hill_saturation,
)


def test_adstock_theta0_is_identity():
    x = np.array([0, 0, 100, 0, 0, 0.0])
    assert np.allclose(geometric_adstock(x, 0.0), x)


def test_adstock_geometric_decay():
    x = np.array([0, 0, 100, 0, 0, 0, 0, 0.0])
    out = geometric_adstock(x, 0.5)
    assert np.isclose(out[2], 100) and np.isclose(out[3], 50) and np.isclose(out[4], 25)


def test_adstock_normalize_preserves_scale():
    big = np.ones(200) * 10
    norm = geometric_adstock(big, 0.7, normalize=True).mean()
    assert abs(norm - 10) < 0.5


def test_truncated_normalized_weights_current_week_most():
    # a single spike should appear at its own week with the largest weight
    x = np.zeros(20)
    x[10] = 100.0
    out = geometric_adstock(x, 0.6, normalize=True, L=12)
    assert out.argmax() == 10


def test_hill_bounds_and_halfsat():
    xs = np.linspace(0, 1000, 500)
    h = hill_saturation(xs, half_sat=200, slope=2.0)
    assert h.min() >= 0 and h.max() < 1
    assert np.all(np.diff(h) >= -1e-9)
    assert abs(hill_saturation(np.array([200.0]), 200, 2.0)[0] - 0.5) < 1e-6


def test_delayed_adstock_peaks_after_burst():
    burst = np.zeros(20)
    burst[5] = 100
    d = delayed_adstock(burst, theta=0.6, peak=2, L=12)
    assert d.argmax() >= 6
