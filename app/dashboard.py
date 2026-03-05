"""
Streamlit dashboard entry point for V-A-G telemetry comparison.

Sidebar: Year, Race, Session, Driver A, Driver B.
Main: Synchronized subplots (Time Delta, Speed, Throttle/Brake, Gear) and Kamm Circle.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on path when running: streamlit run app/dashboard.py
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import streamlit as st

import fastf1

from app import components
from src import data_loader
from src import processor

from fastf1.api import Cache

_cache_dir = _project_root / "data"
_cache_dir.mkdir(parents=True, exist_ok=True)
Cache.enable_cache(str(_cache_dir))

st.set_page_config(page_title="V-A-G Telemetry Comparison", layout="wide")

# Sidebar
st.sidebar.title("Session & Drivers")
year = st.sidebar.number_input("Year", min_value=2018, max_value=2025, value=2023, step=1)

try:
    schedule = fastf1.get_event_schedule(int(year))
    # Use Location for shorter names; fallback to EventName
    if schedule is not None and len(schedule) > 0:
        race_options = schedule["Location"].fillna(schedule["EventName"]).tolist()
        event_names = schedule["EventName"].tolist()
    else:
        race_options = ["Monaco", "Monza", "Silverstone"]
        event_names = race_options
except Exception:
    race_options = ["Monaco", "Monza", "Silverstone"]
    event_names = race_options

race_choice = st.sidebar.selectbox("Race", options=race_options, index=min(0, len(race_options) - 1))
session_choice = st.sidebar.selectbox(
    "Session",
    options=["R", "Q", "FP1", "FP2", "FP3"],
    format_func=lambda x: {"R": "Race", "Q": "Qualifying", "FP1": "FP1", "FP2": "FP2", "FP3": "FP3"}.get(x, x),
)
# Map back to event name if we used Location
try:
    idx = race_options.index(race_choice) if race_choice in race_options else 0
    grand_prix = event_names[idx] if idx < len(event_names) else race_choice
except Exception:
    grand_prix = race_choice

# Load session to get drivers (on demand)
drivers_loaded = False
driver_options = []
if "session_loaded" not in st.session_state:
    st.session_state.session_loaded = None
    st.session_state.driver_list = []

def load_session_drivers():
    try:
        session_obj = fastf1.get_session(year, grand_prix, session_choice)
        session_obj.load()
        drivers = session_obj.laps["Driver"].dropna().unique().tolist()
        if not drivers and hasattr(session_obj, "results") and session_obj.results is not None:
            drivers = session_obj.results["Abbreviation"].dropna().astype(str).tolist()
        st.session_state.driver_list = sorted(drivers) if drivers else []
        st.session_state.session_loaded = session_obj
        return True
    except Exception as e:
        st.sidebar.warning(f"Could not load session: {e}")
        st.session_state.driver_list = []
        return False

if st.sidebar.button("Load session (refresh drivers)"):
    load_session_drivers()

if st.session_state.driver_list:
    driver_options = st.session_state.driver_list
else:
    # Defaults so UI always has choices; will error on Compare if not loaded
    driver_options = ["VER", "HAM", "PER", "LEC", "SAI", "NOR", "ALO"]

driver_a = st.sidebar.selectbox("Driver A (reference)", options=driver_options, index=0)
driver_b = st.sidebar.selectbox("Driver B", options=driver_options, index=min(1, len(driver_options) - 1))

if not st.session_state.driver_list and st.sidebar.button("Auto-load session"):
    load_session_drivers()
    st.rerun()

# Main: Compare button and charts
st.title("Comparative Telemetry Analysis (V-A-G)")

if st.button("Load & Compare"):
    with st.spinner("Loading telemetry and computing..."):
        try:
            df_a, df_b = data_loader.load_driver_fastest_lap_telemetry(
                year=int(year),
                grand_prix=grand_prix,
                session=session_choice,
                driver_1=driver_a,
                driver_2=driver_b,
                cache_path=str(_project_root / "data"),
            )
        except ValueError as e:
            st.error(str(e))
            st.stop()
        except Exception as e:
            st.error(f"Failed to load data: {e}")
            st.stop()

        result = processor.resync_and_derive(df_a, df_b, distance_step=1.0, add_g_forces=True)
        d = result.distance
        da, db = result.df_a, result.df_b
        dt = result.delta_t

        # Subplots: Delta, Speed, Throttle/Brake, Gear
        fig_delta = components.plot_time_delta(d, dt)
        fig_speed = components.plot_speed_comparison(
            d, da["Speed"].values, db["Speed"].values, driver_a, driver_b
        )
        fig_tb = components.plot_throttle_brake(
            d,
            da["Throttle"].values,
            da["Brake"].values,
            db["Throttle"].values,
            db["Brake"].values,
            driver_a,
            driver_b,
        )
        fig_gear = components.plot_gear(
            d, da["Gear"].values, db["Gear"].values, driver_a, driver_b
        )

        st.subheader("Synchronized comparison (Lap Distance)")
        st.plotly_chart(fig_delta, use_container_width=True)
        st.plotly_chart(fig_speed, use_container_width=True)
        st.plotly_chart(fig_tb, use_container_width=True)
        st.plotly_chart(fig_gear, use_container_width=True)

        if "Gx" in da.columns and "Gy" in da.columns:
            st.subheader("Kamm Circle (Friction Circle)")
            fig_kamm = components.plot_kamm_circle_combined(
                da["Gx"].values,
                da["Gy"].values,
                db["Gx"].values,
                db["Gy"].values,
                driver_a,
                driver_b,
            )
            st.plotly_chart(fig_kamm, use_container_width=True)
        else:
            st.info("Gx/Gy not available for Kamm circle.")

else:
    st.info("Select session and drivers in the sidebar, then click **Load & Compare**.")
