"""
Data ingestion module: FastF1 API connection, caching, and telemetry extraction.

Downloads and cleans raw telemetry using the fastf1 library. Output schema is
normalized for downstream processing (processor, physics).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import pandas as pd

# Lazy import fastf1 to allow cache setup before first use
_fastf1 = None


def _get_fastf1():  # type: () -> object
    """Return fastf1 module, importing on first use."""
    global _fastf1
    if _fastf1 is None:
        import fastf1  # noqa: F401
        _fastf1 = fastf1
    return _fastf1


def setup_cache(cache_path: str | Path | None = None) -> str:
    """
    Enable FastF1 cache to avoid API rate limits and speed up development.

    Args:
        cache_path: Directory for cache. If None, uses project_root/data.

    Returns:
        The absolute path where cache is enabled.
    """
    fastf1 = _get_fastf1()
    if cache_path is None:
        project_root = Path(__file__).resolve().parent.parent
        cache_path = project_root / "data"
    path = str(Path(cache_path).resolve())
    from fastf1.api import Cache
    Cache.enable_cache(path)
    return path


def _telemetry_to_base_df(telemetry_df: pd.DataFrame, driver: str) -> pd.DataFrame:
    """
    Convert FastF1 telemetry DataFrame to the base schema expected by the PRD.

    Maps nGear -> Gear, converts Time to seconds, X/Y/Z from 1/10 m to m.
    Physical logic: Distance is already in meters from add_distance();
    positions X,Y,Z are in tenths of meters in the API.

    Args:
        telemetry_df: Raw telemetry (must contain Distance from add_distance()).
        driver: Driver abbreviation (e.g. 'VER').

    Returns:
        DataFrame with columns ['Time', 'Driver', 'Distance', 'Speed', 'RPM',
        'Gear', 'Throttle', 'Brake', 'X', 'Y', 'Z'].
    """
    df = telemetry_df.copy()
    # Time: convert timedelta to float seconds
    if "Time" in df.columns and hasattr(df["Time"].iloc[0], "total_seconds"):
        time_sec = df["Time"].dt.total_seconds().values
    else:
        time_sec = pd.to_numeric(df["Time"], errors="coerce").values
    # Gear: FastF1 uses nGear
    gear = df["nGear"].values if "nGear" in df.columns else df.get("Gear", pd.Series(dtype=float)).values
    # Positions: FastF1 uses 1/10 m
    x = df["X"].values * 0.1 if "X" in df.columns else df["X"].values
    y = df["Y"].values * 0.1 if "Y" in df.columns else df["Y"].values
    z = df["Z"].values * 0.1 if "Z" in df.columns else df["Z"].values
    out = pd.DataFrame(
        {
            "Time": time_sec,
            "Driver": driver,
            "Distance": df["Distance"].values,
            "Speed": df["Speed"].values,
            "RPM": df["RPM"].values,
            "Gear": gear,
            "Throttle": df["Throttle"].values,
            "Brake": df["Brake"].values,
            "X": x,
            "Y": y,
            "Z": z,
        }
    )
    return out


def load_driver_fastest_lap_telemetry(
    year: int,
    grand_prix: str,
    session: str,
    driver_1: str,
    driver_2: str,
    cache_path: str | Path | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and clean telemetry for the fastest lap of two drivers in a given session.

    Uses FastF1 caching. Extracts telemetry and lap data, filters to fastest lap
    per driver, and returns two DataFrames with the base schema.

    Args:
        year: Season year (e.g. 2023).
        grand_prix: Event name (e.g. 'Monaco', 'French Grand Prix').
        session: Session type (e.g. 'Q', 'R', 'FP1').
        driver_1: Driver A abbreviation (e.g. 'VER').
        driver_2: Driver B abbreviation (e.g. 'HAM').
        cache_path: Optional cache directory. If None, project data/ is used.

    Returns:
        Tuple of (df_driver_1, df_driver_2), each with columns ['Time', 'Driver',
        'Distance', 'Speed', 'RPM', 'Gear', 'Throttle', 'Brake', 'X', 'Y', 'Z'].

    Raises:
        ValueError: If session cannot be loaded, or a driver has no valid fastest lap.
    """
    setup_cache(cache_path)
    fastf1 = _get_fastf1()

    try:
        session_obj = fastf1.get_session(year, grand_prix, session)
    except Exception as e:
        raise ValueError(f"Session not found or invalid: {year} {grand_prix} {session}") from e

    try:
        session_obj.load()
    except Exception as e:
        raise ValueError(
            f"Failed to load session data (API or network error): {year} {grand_prix} {session}"
        ) from e

    def get_fastest_lap_telemetry(driver_abbrev: str) -> pd.DataFrame:
        try:
            driver_laps = session_obj.laps.pick_driver(driver_abbrev)
        except Exception as e:
            raise ValueError(f"Driver not found in session: {driver_abbrev}") from e
        if driver_laps is None or len(driver_laps) == 0:
            raise ValueError(f"No laps found for driver: {driver_abbrev} (e.g. DNS)")
        try:
            fastest = driver_laps.pick_fastest()
        except Exception as e:
            raise ValueError(f"No valid fastest lap for driver: {driver_abbrev}") from e
        if fastest is None:
            raise ValueError(f"No valid fastest lap for driver: {driver_abbrev}")
        try:
            tel = fastest.get_telemetry()
        except Exception as e:
            raise ValueError(f"Failed to get telemetry for driver: {driver_abbrev}") from e
        tel = tel.add_distance()
        return _telemetry_to_base_df(tel, driver_abbrev)

    df1 = get_fastest_lap_telemetry(driver_1)
    df2 = get_fastest_lap_telemetry(driver_2)
    return df1, df2
