"""
Unit tests for data_loader: schema normalization, cache setup, error handling.

Uses mocks for FastF1 to avoid network/API calls.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src import data_loader


def test_setup_cache_returns_path():
    """setup_cache returns the absolute path used for cache."""
    path = data_loader.setup_cache()
    assert path
    assert Path(path).is_absolute()


def test_telemetry_to_base_df_schema():
    """_telemetry_to_base_df produces required columns and types."""
    n = 50
    time_td = pd.to_timedelta(np.linspace(0, 60, n), unit="s")
    df = pd.DataFrame({
        "Time": time_td,
        "Distance": np.linspace(0, 1000, n),
        "Speed": np.full(n, 200.0),
        "RPM": np.full(n, 10000),
        "nGear": np.full(n, 5),
        "Throttle": np.full(n, 100.0),
        "Brake": np.zeros(n),
        "X": np.linspace(0, 500, n) * 10,
        "Y": np.zeros(n) * 10,
        "Z": np.zeros(n) * 10,
    })
    out = data_loader._telemetry_to_base_df(df, "VER")
    required = ["Time", "Driver", "Distance", "Speed", "RPM", "Gear", "Throttle", "Brake", "X", "Y", "Z"]
    for col in required:
        assert col in out.columns, f"Missing column {col}"
    assert out["Driver"].iloc[0] == "VER"
    assert out["Time"].dtype.kind == "f"
    assert out["X"].iloc[0] == 0.0
    assert out["X"].iloc[-1] == 500.0


@patch("src.data_loader._get_fastf1")
def test_load_driver_fastest_lap_telemetry_raises_on_missing_session(mock_get_fastf1):
    """load_driver_fastest_lap_telemetry raises ValueError when session not found."""
    mock_ff1 = MagicMock()
    mock_ff1.get_session.side_effect = Exception("Not found")
    mock_ff1.api.Cache.enable_cache = MagicMock()
    mock_get_fastf1.return_value = mock_ff1
    with pytest.raises(ValueError, match="Session not found|Failed to load"):
        data_loader.load_driver_fastest_lap_telemetry(2023, "Invalid GP", "Q", "VER", "HAM")
