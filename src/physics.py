"""
Pure physics and signal processing: G-force (longitudinal and lateral) from telemetry.

Uses NumPy/SciPy only; no slow loops. Longitudinal G from dv/dt with low-pass
filtering; lateral G from trajectory (X, Y) and speed when not provided by API.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as scipy_signal

GRAVITY_MPS2: float = 9.81


def longitudinal_g(
    speed_kmh: np.ndarray,
    time_s: np.ndarray,
    filter_hz: float = 3.0,
    filter_order: int = 3,
) -> np.ndarray:
    """
    Compute longitudinal acceleration in Gs from speed and time.

    Physical logic: G_x = (1/g) * dv/dt. The derivative is smoothed with a
    zero-phase Butterworth low-pass filter to reduce sensor noise.

    Args:
        speed_kmh: Speed in km/h (vectorized, same length as time_s).
        time_s: Time in seconds.
        filter_hz: Cutoff frequency in Hz for the low-pass filter.
        filter_order: Order of the Butterworth filter.

    Returns:
        Longitudinal acceleration in Gs (positive = acceleration, negative = braking).
    """
    v_ms = speed_kmh * 1000.0 / 3600.0
    dt = np.gradient(time_s)
    dt = np.where(dt <= 0, np.nanmedian(dt[dt > 0]) if np.any(dt > 0) else 1e-3, dt)
    dv_dt = np.gradient(v_ms, time_s)
    gx_raw = dv_dt / GRAVITY_MPS2
    if filter_hz <= 0 or len(time_s) < 10:
        return gx_raw
    fs = 1.0 / np.median(np.diff(time_s))
    nyq = fs * 0.5
    low = min(filter_hz / nyq, 0.99)
    b, a = scipy_signal.butter(filter_order, low, btype="low")
    gx_smooth = scipy_signal.filtfilt(b, a, gx_raw, axis=0)
    return gx_smooth.astype(np.float64)


def lateral_g_from_trajectory(
    x_m: np.ndarray,
    y_m: np.ndarray,
    time_s: np.ndarray,
    filter_hz: float = 3.0,
    filter_order: int = 3,
) -> np.ndarray:
    """
    Compute lateral acceleration in Gs from trajectory (X, Y) and time.

    Physical logic: lateral acceleration a_y = v^2 / R in the plane of the track
    (XY). We obtain acceleration from second derivative of position, then project
    onto the lateral direction (perpendicular to velocity). Velocity is (dx/dt, dy/dt);
    we use curvature * v^2 = |a_perp| with a = (d²x/dt², d²y/dt²), and a_perp is
    the component of a perpendicular to v. Then G_y = a_perp_magnitude / g.
    Hypothesis: XY plane is the track plane; Z is ignored for lateral G.

    Args:
        x_m: X position in meters.
        y_m: Y position in meters.
        time_s: Time in seconds (same length as x_m, y_m).
        filter_hz: Cutoff frequency in Hz for low-pass filter on derivatives.
        filter_order: Order of the Butterworth filter.

    Returns:
        Lateral acceleration in Gs (absolute value of lateral component).
    """
    vx = np.gradient(x_m, time_s)
    vy = np.gradient(y_m, time_s)
    ax = np.gradient(vx, time_s)
    ay = np.gradient(vy, time_s)
    v_norm_sq = np.maximum(vx * vx + vy * vy, 1e-6)
    a_perp = np.abs(ax * (-vy) + ay * vx) / np.sqrt(v_norm_sq)
    gy_raw = a_perp / GRAVITY_MPS2
    if filter_hz <= 0 or len(time_s) < 10:
        return gy_raw
    fs = 1.0 / np.median(np.diff(time_s))
    nyq = fs * 0.5
    low = min(filter_hz / nyq, 0.99)
    b, a = scipy_signal.butter(filter_order, low, btype="low")
    gy_smooth = scipy_signal.filtfilt(b, a, gy_raw, axis=0)
    return gy_smooth.astype(np.float64)


def lateral_g(
    speed_kmh: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    time_s: np.ndarray,
    filter_hz: float = 3.0,
    filter_order: int = 3,
) -> np.ndarray:
    """
    Compute lateral acceleration in Gs from trajectory.

    Delegates to lateral_g_from_trajectory. Exposed for a unified API when
    no precomputed lateral channel is available.

    Args:
        speed_kmh: Speed in km/h (unused in trajectory-based method; for API consistency).
        x_m: X position in meters.
        y_m: Y position in meters.
        time_s: Time in seconds.
        filter_hz: Cutoff frequency in Hz.
        filter_order: Butterworth order.

    Returns:
        Lateral acceleration in Gs.
    """
    return lateral_g_from_trajectory(x_m, y_m, time_s, filter_hz, filter_order)
