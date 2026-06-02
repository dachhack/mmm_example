"""
Stage 3: MMM transforms — adstock (carryover) and Hill saturation (diminishing returns).
Reusable, vectorized, and tested. These are the heart of the model: they put media
variables into the shape in which advertising actually acts on conversions.
"""
import numpy as np

# ----------------------------------------------------------------------
# ADSTOCK / CARRYOVER
# ----------------------------------------------------------------------
def geometric_adstock(x, theta, normalize=False, L=None):
    """
    Geometric adstock: effect[t] = x[t] + theta * effect[t-1].

    theta in [0,1): fraction of last period's effective exposure carried forward.
      theta=0 -> no carryover (effect = x). theta->1 -> very long carryover.

    normalize: if True, divide by 1/(1-theta) so the transform preserves the
      *scale* of x (total mass conserved). Useful so theta doesn't silently
      inflate the magnitude — it only redistributes timing. We'll discuss why
      this matters for interpreting beta.

    L: optional finite max lag (truncated adstock). If None, infinite recursive.
    """
    x = np.asarray(x, dtype=float)
    if L is not None:
        # finite-window weighted sum (geometric weights up to lag L)
        weights = theta ** np.arange(L)
        if normalize:
            weights = weights / weights.sum()
        out = np.zeros_like(x)
        for t in range(len(x)):
            lo = max(0, t - L + 1)
            seg = x[lo:t+1][::-1]
            out[t] = np.dot(seg, weights[:len(seg)])
        return out
    # infinite recursive form
    out = np.zeros_like(x)
    out[0] = x[0]
    for t in range(1, len(x)):
        out[t] = x[t] + theta * out[t-1]
    if normalize:
        out = out * (1 - theta)   # rescale so mean(out) ~ mean(x)
    return out


def delayed_adstock(x, theta, peak=0, L=12):
    """
    Delayed-geometric adstock: weight peaks at lag `peak`, not lag 0.
    Models channels (e.g. TV) whose effect is strongest a week or two AFTER airing.
    weight[l] = theta ** ((l - peak)**2)  (a peaked, decaying kernel)
    """
    x = np.asarray(x, dtype=float)
    lags = np.arange(L)
    weights = theta ** ((lags - peak) ** 2)
    weights = weights / weights.sum()
    out = np.zeros_like(x)
    for t in range(len(x)):
        lo = max(0, t - L + 1)
        seg = x[lo:t+1][::-1]
        out[t] = np.dot(seg, weights[:len(seg)])
    return out


# ----------------------------------------------------------------------
# SATURATION / DIMINISHING RETURNS
# ----------------------------------------------------------------------
def hill_saturation(x, half_sat, slope):
    """
    Hill function: response = x^slope / (x^slope + half_sat^slope), in [0,1).
    half_sat: x value giving half-maximal response (the 'bend' point).
    slope: steepness. >1 => S-curve with a threshold; <=1 => immediate diminishing returns.
    Returns the FRACTION of the channel ceiling reached. Multiply by beta for conversions.
    """
    x = np.maximum(np.asarray(x, dtype=float), 0)
    return x**slope / (x**slope + half_sat**slope + 1e-12)


def logistic_saturation(x, lam):
    """
    Alternative one-parameter saturation used by Meta Robyn-style models:
    response = (1 - exp(-lam * x)) ... simple concave, no S-shape, no threshold.
    Fewer parameters (easier to fit) but cannot represent a true S-curve.
    """
    x = np.maximum(np.asarray(x, dtype=float), 0)
    return 1.0 - np.exp(-lam * x)


# ----------------------------------------------------------------------
# COMPOSED MEDIA TRANSFORM (the standard pipeline order)
# ----------------------------------------------------------------------
def media_transform(x, theta, half_sat, slope, normalize_adstock=False):
    """Standard order: raw -> adstock -> Hill saturation -> (x beta done by caller)."""
    ad = geometric_adstock(x, theta, normalize=normalize_adstock)
    return hill_saturation(ad, half_sat, slope)


# ----------------------------------------------------------------------
# SELF-TESTS
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np
    print("Running transform self-tests...")

    # adstock: theta=0 is identity
    x = np.array([0,0,100,0,0,0,0,0.])
    assert np.allclose(geometric_adstock(x, 0.0), x), "theta=0 must be identity"

    # adstock: single burst decays geometrically
    out = geometric_adstock(x, 0.5)
    assert np.isclose(out[2], 100) and np.isclose(out[3], 50) and np.isclose(out[4], 25), "geometric decay wrong"

    # adstock: total mass with normalize ~ preserved
    big = np.ones(200)*10
    raw_sum = geometric_adstock(big, 0.7).mean()
    norm_sum = geometric_adstock(big, 0.7, normalize=True).mean()
    assert abs(norm_sum - 10) < 0.5, f"normalized adstock should preserve scale, got {norm_sum}"

    # hill: monotonic increasing, bounded 0..1, equals 0.5 at half_sat
    xs = np.linspace(0, 1000, 500)
    h = hill_saturation(xs, half_sat=200, slope=2.0)
    assert h.min() >= 0 and h.max() < 1, "hill must be in [0,1)"
    assert np.all(np.diff(h) >= -1e-9), "hill must be monotincreasing"
    assert abs(hill_saturation(np.array([200.]), 200, 2.0)[0] - 0.5) < 1e-6, "hill(half_sat)=0.5"

    # delayed adstock: peak shifts the mass
    burst = np.zeros(20)
    burst[5] = 100
    d = delayed_adstock(burst, theta=0.6, peak=2, L=12)
    assert d.argmax() >= 6, "delayed adstock should peak AFTER the burst"

    print("All transform self-tests passed.")
