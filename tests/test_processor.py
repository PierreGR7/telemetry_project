"""
Unit tests for processor: resync on distance, time delta.
"""

import numpy as np
import pandas as pd
import pytest

from src.processor import (
    compute_time_delta,
    resync_and_derive,
    resync_telemetry,
)


@pytest.fixture
def synthetic_lap_a():
    """Synthetic lap A: distance 0..1000 m, time 0..60 s."""
    n = 200
    distance = np.linspace(0, 1000, n)
    time = np.linspace(0, 60, n)
    speed = np.full(n, 100.0 / 3.6)  # ~100 km/h constant
    return pd.DataFrame({
        "Time": time,
        "Driver": "A",
        "Distance": distance,
        "Speed": speed * 3.6,
        "RPM": np.full(n, 8000),
        "Gear": np.full(n, 4),
        "Throttle": np.full(n, 80),
        "Brake": np.zeros(n),
        "X": distance * 0.5,
        "Y": np.sin(distance / 100) * 10,
        "Z": np.zeros(n),
    })


@pytest.fixture
def synthetic_lap_b():
    """Synthetic lap B: same distance range, slightly different time (slower)."""
    n = 250
    distance = np.linspace(0, 1000, n)
    time = np.linspace(0, 65, n)  # 65 s vs 60 s
    speed = 1000.0 / 65 * 3.6  # km/h
    return pd.DataFrame({
        "Time": time,
        "Driver": "B",
        "Distance": distance,
        "Speed": np.full(n, speed),
        "RPM": np.full(n, 7500),
        "Gear": np.full(n, 4),
        "Throttle": np.full(n, 70),
        "Brake": np.zeros(n),
        "X": distance * 0.5,
        "Y": np.sin(distance / 100) * 10,
        "Z": np.zeros(n),
    })


def test_resync_telemetry_returns_common_length(synthetic_lap_a, synthetic_lap_b):
    """Resynced DataFrames and distance axis have consistent length."""
    dist, df_a, df_b = resync_telemetry(synthetic_lap_a, synthetic_lap_b, distance_step=10.0)
    assert len(dist) == len(df_a) == len(df_b)
    assert np.allclose(df_a["Distance"].values, dist)
    assert np.allclose(df_b["Distance"].values, dist)


def test_compute_time_delta_baseline_a(synthetic_lap_a, synthetic_lap_b):
    """Delta = time_B - time_A; after resync, B is slower so delta > 0 on average."""
    dist, df_a, df_b = resync_telemetry(synthetic_lap_a, synthetic_lap_b, distance_step=20.0)
    delta = compute_time_delta(dist, df_a["Time"].values, df_b["Time"].values)
    assert len(delta) == len(dist)
    assert np.nanmean(delta) > 0


def test_resync_and_derive_returns_delta_t(synthetic_lap_a, synthetic_lap_b):
    """resync_and_derive returns ResyncResult with delta_t."""
    result = resync_and_derive(synthetic_lap_a, synthetic_lap_b, distance_step=10.0, add_g_forces=False)
    assert hasattr(result, "distance")
    assert hasattr(result, "df_a")
    assert hasattr(result, "df_b")
    assert hasattr(result, "delta_t")
    assert len(result.delta_t) == len(result.distance)


def test_resync_and_derive_adds_gx_gy(synthetic_lap_a, synthetic_lap_b):
    """With add_g_forces=True, df_a and df_b have Gx and Gy columns."""
    result = resync_and_derive(synthetic_lap_a, synthetic_lap_b, distance_step=10.0, add_g_forces=True)
    assert "Gx" in result.df_a.columns and "Gy" in result.df_a.columns
    assert "Gx" in result.df_b.columns and "Gy" in result.df_b.columns
    assert np.all(np.isfinite(result.df_a["Gx"].values))
    assert np.all(np.isfinite(result.df_a["Gy"].values))
