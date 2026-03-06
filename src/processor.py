"""
Spatial resynchronization and time-delta computation.

Resamples telemetry onto a common distance axis (e.g. 1 m step) using
scipy.interpolate.interp1d, then computes time delta between Driver A (reference)
and Driver B. Optionally derives Gx, Gy on the resynced data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from src import physics

# Base columns to interpolate (numeric only)
BASE_CHANNELS: List[str] = [
    "Time",
    "Distance",
    "Speed",
    "RPM",
    "Gear",
    "Throttle",
    "Brake",
    "X",
    "Y",
    "Z",
]

# Optional telemetry channels that may exist depending on year/session
OPTIONAL_CHANNELS: List[str] = [
    "DRS",
    "ERSDeployMode",
    "ERSDeploy",
    "ERSPower",
    "ERS",
]


def _resample_on_distance(
    distance: np.ndarray,
    values: np.ndarray,
    new_distance: np.ndarray,
    kind: str = "linear",
) -> np.ndarray:
    """
    Resample a 1D signal from distance to a new distance grid.

    Args:
        distance: Original distance axis (m).
        values: Signal values.
        new_distance: Target distance axis (m).
        kind: Interpolation kind for interp1d.

    Returns:
        Interpolated values on new_distance.
    """
    if len(distance) < 2 or len(values) < 2:
        return np.full_like(new_distance, np.nan)
    f = interp1d(
        distance,
        values,
        kind=kind,
        bounds_error=False,
        fill_value=(values[0], values[-1]),
    )
    return f(new_distance).astype(np.float64)


def compute_time_delta(
    distance: np.ndarray,
    time_a: np.ndarray,
    time_b: np.ndarray,
) -> np.ndarray:
    """
    Compute time difference between Driver B and Driver A at each distance.

    Reference is Driver A: delta_t = time_B - time_A, so baseline is 0 for A.

    Args:
        distance: Common distance axis (m).
        time_a: Time of Driver A at each distance (s).
        time_b: Time of Driver B at each distance (s).

    Returns:
        Delta time in seconds (positive = B slower than A at that point).
    """
    return time_b - time_a


def resync_telemetry(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    distance_step: float = 1.0,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """
    Resample both drivers' telemetry onto a common distance grid.

    Uses the minimum of both laps' max distance so we compare like-for-like.
    All numeric channels in BASE_CHANNELS are interpolated.

    Args:
        df_a: Base schema DataFrame for Driver A (must contain Distance and others).
        df_b: Base schema DataFrame for Driver B.
        distance_step: Step in meters for the common grid.

    Returns:
        Tuple (distance_axis, df_a_resynced, df_b_resynced). DataFrames have
        the same columns as input (Driver preserved as constant) and index
        aligned to distance_axis.
    """
    d_a = df_a["Distance"].values
    d_b = df_b["Distance"].values
    d_max = min(d_a.max(), d_b.max())
    d_min = max(d_a.min(), d_b.min())
    if d_max <= d_min:
        d_max = max(d_a.max(), d_b.max())
        d_min = 0.0
    distance_axis = np.arange(d_min, d_max, step=distance_step, dtype=np.float64)
    if len(distance_axis) == 0:
        distance_axis = np.array([0.0])

    driver_a = df_a["Driver"].iloc[0] if "Driver" in df_a.columns else "A"
    driver_b = df_b["Driver"].iloc[0] if "Driver" in df_b.columns else "B"

    out_a = {"Distance": distance_axis, "Driver": driver_a}
    out_b = {"Distance": distance_axis, "Driver": driver_b}

    for col in (BASE_CHANNELS + OPTIONAL_CHANNELS):
        if col in ("Distance", "Driver"):
            continue
        if col not in df_a.columns or col not in df_b.columns:
            continue
        out_a[col] = _resample_on_distance(
            d_a, df_a[col].values.astype(float), distance_axis
        )
        out_b[col] = _resample_on_distance(
            d_b, df_b[col].values.astype(float), distance_axis
        )

    return distance_axis, pd.DataFrame(out_a), pd.DataFrame(out_b)


@dataclass
class ResyncResult:
    """Result of resync and derived signals."""

    distance: np.ndarray
    df_a: pd.DataFrame
    df_b: pd.DataFrame
    delta_t: np.ndarray


def resync_and_derive(
    lap_a: pd.DataFrame,
    lap_b: pd.DataFrame,
    distance_step: float = 1.0,
    add_g_forces: bool = True,
    filter_hz: float = 3.0,
) -> ResyncResult:
    """
    Resynchronize two laps on distance, compute time delta, and optionally Gx/Gy.

    High-level pipeline: resync -> time delta -> (optional) Gx/Gy on resynced
    time/speed/position.

    Args:
        lap_a: Base schema DataFrame for Driver A.
        lap_b: Base schema DataFrame for Driver B.
        distance_step: Step in meters for the common distance grid.
        add_g_forces: If True, compute Gx and Gy and add to resynced DataFrames.
        filter_hz: Cutoff frequency for G-force smoothing.

    Returns:
        ResyncResult with distance, df_a, df_b, and delta_t. df_a and df_b
        include Gx, Gy columns when add_g_forces is True.
    """
    distance_axis, df_a, df_b = resync_telemetry(lap_a, lap_b, distance_step)
    time_a = df_a["Time"].values
    time_b = df_b["Time"].values
    delta_t = compute_time_delta(distance_axis, time_a, time_b)

    if add_g_forces:
        gx_a = physics.longitudinal_g(
            df_a["Speed"].values, df_a["Time"].values, filter_hz=filter_hz
        )
        gx_b = physics.longitudinal_g(
            df_b["Speed"].values, df_b["Time"].values, filter_hz=filter_hz
        )
        gy_a = physics.lateral_g_from_trajectory(
            df_a["X"].values,
            df_a["Y"].values,
            df_a["Time"].values,
            filter_hz=filter_hz,
        )
        gy_b = physics.lateral_g_from_trajectory(
            df_b["X"].values,
            df_b["Y"].values,
            df_b["Time"].values,
            filter_hz=filter_hz,
        )
        df_a = df_a.assign(Gx=gx_a, Gy=gy_a)
        df_b = df_b.assign(Gx=gx_b, Gy=gy_b)

    return ResyncResult(distance=distance_axis, df_a=df_a, df_b=df_b, delta_t=delta_t)


def detect_brake_onsets(
    distance: np.ndarray,
    brake: np.ndarray,
    *,
    threshold: float = 0.5,
    debounce_m: float = 20.0,
) -> np.ndarray:
    """
    Detect braking onset points along distance.

    Args:
        distance: Distance axis (m).
        brake: Brake signal (bool-ish or 0/1 or %). Any value > threshold is "braking".
        threshold: Threshold above which braking is considered active.
        debounce_m: Minimum distance (m) between two detected onsets.

    Returns:
        1D array of distances (m) where braking starts.
    """
    if len(distance) == 0:
        return np.array([], dtype=np.float64)
    b = np.asarray(brake, dtype=np.float64)
    active = b > threshold
    rising = np.flatnonzero(np.logical_and(active[1:], np.logical_not(active[:-1]))) + 1
    if rising.size == 0:
        return np.array([], dtype=np.float64)

    pts = distance[rising].astype(np.float64)
    # Debounce: keep first onset, then only those far enough
    kept = [pts[0]]
    for x in pts[1:]:
        if (x - kept[-1]) >= debounce_m:
            kept.append(float(x))
    return np.array(kept, dtype=np.float64)
