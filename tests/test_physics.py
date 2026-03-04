"""
Unit tests for physics module: Gx, Gy, Butterworth filtering.
"""

import numpy as np
import pytest

from src.physics import (
    GRAVITY_MPS2,
    lateral_g_from_trajectory,
    longitudinal_g,
)


def test_longitudinal_g_linear_speed():
    """For v(t) = k*t (linear acceleration), Gx should be constant = k/g."""
    t = np.linspace(0, 10, 500)
    k = 5.0  # m/s^2 equivalent from linear v in m/s
    v_ms = k * t
    v_kmh = v_ms * 3.6
    gx = longitudinal_g(v_kmh, t, filter_hz=0)  # no filter to check raw
    expected = k / GRAVITY_MPS2
    assert np.allclose(gx[1:-1], expected, atol=0.5)  # gradient at edges can differ
    assert np.all(np.isfinite(gx))


def test_longitudinal_g_constant_speed():
    """For constant speed, Gx should be ~0 (after smoothing)."""
    t = np.linspace(0, 10, 500)
    v_kmh = np.full_like(t, 100.0)
    gx = longitudinal_g(v_kmh, t, filter_hz=3.0)
    assert np.allclose(gx, 0.0, atol=0.1)
    assert np.all(np.isfinite(gx))


def test_longitudinal_g_returns_same_length():
    """Output length must match input."""
    t = np.linspace(0, 5, 100)
    v = np.sin(t) * 50 + 100
    gx = longitudinal_g(v, t, filter_hz=2.0)
    assert len(gx) == len(t)


def test_lateral_g_from_trajectory_circular():
    """For circular path at constant speed, |a_lat| = v^2/r, so Gy = v^2/(r*g)."""
    n = 500
    t = np.linspace(0, 10, n)
    omega = 2 * np.pi / 10
    r = 50.0
    x = r * np.cos(omega * t)
    y = r * np.sin(omega * t)
    v = omega * r  # m/s
    expected_gy = (v * v / r) / GRAVITY_MPS2
    gy = lateral_g_from_trajectory(x, y, t, filter_hz=0)
    assert np.all(np.isfinite(gy))
    assert np.allclose(gy[10:-10], expected_gy, rtol=0.3)


def test_lateral_g_from_trajectory_returns_same_length():
    """Output length must match input."""
    t = np.linspace(0, 5, 80)
    x = np.cumsum(np.ones_like(t) * 0.1)
    y = np.sin(t) * 20
    gy = lateral_g_from_trajectory(x, y, t, filter_hz=2.0)
    assert len(gy) == len(t)
