"""
Modular Plotly graph components for telemetry comparison.

Each function returns a go.Figure for use in the Streamlit dashboard.
Shared X-axis is Lap Distance (m). No Streamlit calls here for testability.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


def plot_time_delta(
    distance: np.ndarray,
    delta_t: np.ndarray,
    title: str = "Time Delta (s)",
) -> go.Figure:
    """
    Plot time delta vs lap distance. Baseline 0 = Driver A (reference).

    Args:
        distance: Lap distance in meters.
        delta_t: Time difference in seconds (B - A).
        title: Plot title.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=distance,
            y=delta_t,
            mode="lines",
            name="Δt (B - A)",
            line=dict(color="royalblue", width=2),
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title=title,
        xaxis_title="Lap Distance (m)",
        yaxis_title="Δ t (s)",
        template="plotly_white",
        height=280,
        margin=dict(t=40, b=40, l=50, r=30),
    )
    return fig


def plot_speed_comparison(
    distance: np.ndarray,
    speed_a: np.ndarray,
    speed_b: np.ndarray,
    driver_a: str,
    driver_b: str,
    title: str = "Speed",
    brake_points_a: tuple[np.ndarray, np.ndarray] | None = None,
    brake_points_b: tuple[np.ndarray, np.ndarray] | None = None,
) -> go.Figure:
    """
    Plot speed of both drivers vs lap distance.

    Args:
        distance: Lap distance (m).
        speed_a: Speed of Driver A (km/h).
        speed_b: Speed of Driver B (km/h).
        driver_a: Label for Driver A.
        driver_b: Label for Driver B.
        title: Plot title.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=distance,
            y=speed_a,
            mode="lines",
            name=driver_a,
            line=dict(color="darkblue", width=1.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=distance,
            y=speed_b,
            mode="lines",
            name=driver_b,
            line=dict(color="crimson", width=1.5),
        )
    )
    if brake_points_a is not None:
        x_pts, y_pts = brake_points_a
        fig.add_trace(
            go.Scatter(
                x=x_pts,
                y=y_pts,
                mode="markers",
                name=f"{driver_a} braking onset",
                marker=dict(symbol="x", size=7, color="darkblue"),
            )
        )
    if brake_points_b is not None:
        x_pts, y_pts = brake_points_b
        fig.add_trace(
            go.Scatter(
                x=x_pts,
                y=y_pts,
                mode="markers",
                name=f"{driver_b} braking onset",
                marker=dict(symbol="x", size=7, color="crimson"),
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Lap Distance (m)",
        yaxis_title="Speed (km/h)",
        template="plotly_white",
        height=280,
        margin=dict(t=40, b=40, l=50, r=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_overlays(
    distance: np.ndarray,
    driver_a: str,
    driver_b: str,
    *,
    drs_a: np.ndarray | None = None,
    drs_b: np.ndarray | None = None,
    ers_a: np.ndarray | None = None,
    ers_b: np.ndarray | None = None,
    title: str = "Overlays (DRS / ERS)",
) -> go.Figure:
    """
    Plot optional overlay channels (DRS, ERS) vs distance.
    """
    fig = go.Figure()

    def _add_step(x, y, name, color):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=name,
                line=dict(color=color, width=1.5, shape="hv"),
            )
        )

    has_any = False
    if drs_a is not None and drs_b is not None:
        has_any = True
        _add_step(distance, drs_a, f"{driver_a} DRS", "darkblue")
        _add_step(distance, drs_b, f"{driver_b} DRS", "crimson")

    if ers_a is not None and ers_b is not None:
        has_any = True
        _add_step(distance, ers_a, f"{driver_a} ERS", "teal")
        _add_step(distance, ers_b, f"{driver_b} ERS", "orange")

    if not has_any:
        # Keep an empty-but-valid figure
        fig.add_trace(go.Scatter(x=[0], y=[0], mode="lines", name="N/A", line=dict(color="gray")))

    fig.update_layout(
        title=title,
        xaxis_title="Lap Distance (m)",
        yaxis_title="Value",
        template="plotly_white",
        height=220,
        margin=dict(t=40, b=40, l=50, r=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_throttle_brake(
    distance: np.ndarray,
    throttle_a: np.ndarray,
    brake_a: np.ndarray,
    throttle_b: np.ndarray,
    brake_b: np.ndarray,
    driver_a: str,
    driver_b: str,
    title: str = "Throttle & Brake",
) -> go.Figure:
    """
    Plot throttle (0-100%) and brake for both drivers vs distance.

    Args:
        distance: Lap distance (m).
        throttle_a, brake_a: Throttle and brake for Driver A.
        throttle_b, brake_b: Throttle and brake for Driver B.
        driver_a, driver_b: Driver labels.
        title: Plot title.

    Returns:
        Plotly Figure with 4 traces (throttle A/B, brake A/B).
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=distance,
            y=throttle_a,
            mode="lines",
            name=f"{driver_a} Throttle",
            line=dict(color="green", width=1.2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=distance,
            y=throttle_b,
            mode="lines",
            name=f"{driver_b} Throttle",
            line=dict(color="lime", width=1.2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=distance,
            y=brake_a.astype(float) * 100,
            mode="lines",
            name=f"{driver_a} Brake",
            line=dict(color="darkred", width=1.2, dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=distance,
            y=brake_b.astype(float) * 100,
            mode="lines",
            name=f"{driver_b} Brake",
            line=dict(color="orange", width=1.2, dash="dot"),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Lap Distance (m)",
        yaxis_title="Throttle % / Brake (scaled)",
        template="plotly_white",
        height=280,
        margin=dict(t=40, b=40, l=50, r=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(range=[-5, 105]),
    )
    return fig


def plot_gear(
    distance: np.ndarray,
    gear_a: np.ndarray,
    gear_b: np.ndarray,
    driver_a: str,
    driver_b: str,
    title: str = "Gear",
) -> go.Figure:
    """
    Plot gear for both drivers vs lap distance.

    Args:
        distance: Lap distance (m).
        gear_a, gear_b: Gear values (integer or float).
        driver_a, driver_b: Driver labels.
        title: Plot title.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=distance,
            y=gear_a,
            mode="lines",
            name=driver_a,
            line=dict(color="darkblue", width=1.5, shape="hv"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=distance,
            y=gear_b,
            mode="lines",
            name=driver_b,
            line=dict(color="crimson", width=1.5, shape="hv"),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Lap Distance (m)",
        yaxis_title="Gear",
        template="plotly_white",
        height=280,
        margin=dict(t=40, b=40, l=50, r=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(dtick=1),
    )
    return fig


def plot_kamm_circle(
    gx: np.ndarray,
    gy: np.ndarray,
    driver_label: str,
    color: str = "blue",
    opacity: float = 0.6,
) -> go.Figure:
    """
    Scatter Gx (Y-axis) vs Gy (X-axis) for one driver (friction circle / Kamm circle).

    Args:
        gx: Longitudinal acceleration (G).
        gy: Lateral acceleration (G).
        driver_label: Driver name for legend.
        color: Trace color.
        opacity: Point opacity.

    Returns:
        Plotly Figure (single trace).
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=gy,
            y=gx,
            mode="markers",
            name=driver_label,
            marker=dict(size=3, color=color, opacity=opacity),
        )
    )
    fig.update_layout(
        title="Kamm Circle (Friction Circle)",
        xaxis_title="Lateral G (G_y)",
        yaxis_title="Longitudinal G (G_x)",
        template="plotly_white",
        height=400,
        margin=dict(t=40, b=40, l=50, r=30),
        showlegend=True,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    return fig


def plot_kamm_circle_combined(
    gx_a: np.ndarray,
    gy_a: np.ndarray,
    gx_b: np.ndarray,
    gy_b: np.ndarray,
    driver_a: str,
    driver_b: str,
) -> go.Figure:
    """
    Combined Kamm circle: both drivers on one scatter (Gx vs Gy).

    Args:
        gx_a, gy_a: Gx and Gy for Driver A.
        gx_b, gy_b: Gx and Gy for Driver B.
        driver_a, driver_b: Driver labels.

    Returns:
        Plotly Figure with two scatter traces.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=gy_a,
            y=gx_a,
            mode="markers",
            name=driver_a,
            marker=dict(size=3, color="darkblue", opacity=0.6),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=gy_b,
            y=gx_b,
            mode="markers",
            name=driver_b,
            marker=dict(size=3, color="crimson", opacity=0.6),
        )
    )
    fig.update_layout(
        title="Kamm Circle (Friction Circle)",
        xaxis_title="Lateral G (G_y)",
        yaxis_title="Longitudinal G (G_x)",
        template="plotly_white",
        height=400,
        margin=dict(t=40, b=40, l=50, r=30),
        showlegend=True,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    return fig
