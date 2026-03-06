"""
Streamlit dashboard entry point for V-A-G telemetry comparison.

Sidebar: Year, Race, Session, Driver A, Driver B.
Main: Synchronized subplots (Time Delta, Speed, Throttle/Brake, Gear) and Kamm Circle.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

# Ensure project root is on path when running: streamlit run app/dashboard.py
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import streamlit as st

import fastf1
import numpy as np
import pandas as pd

from app import components
from src import data_loader
from src import processor

from fastf1.api import Cache

_cache_dir = _project_root / "data"
_cache_dir.mkdir(parents=True, exist_ok=True)
Cache.enable_cache(str(_cache_dir))

st.set_page_config(page_title="V-A-G Telemetry Comparison", layout="wide")


@st.cache_data(show_spinner=False, ttl=60 * 60)
def _get_schedule(year: int) -> pd.DataFrame | None:
    try:
        return fastf1.get_event_schedule(int(year))
    except Exception:
        return None


def _build_event_options(schedule: pd.DataFrame | None) -> tuple[list[str], dict[str, str]]:
    if schedule is None or len(schedule) == 0:
        opts = ["Monaco", "Monza", "Silverstone"]
        return opts, {x: x for x in opts}

    display_to_event: dict[str, str] = {}
    options: list[str] = []
    for _, row in schedule.iterrows():
        event_name = str(row.get("EventName", "")).strip() or str(row.get("Location", "")).strip()
        location = str(row.get("Location", "")).strip()
        rnd = row.get("RoundNumber", None)
        if pd.notna(rnd):
            display = f"R{int(rnd):02d} — {location or event_name}"
        else:
            display = location or event_name
        if not event_name:
            continue
        options.append(display)
        display_to_event[display] = event_name
    if not options:
        opts = ["Monaco", "Monza", "Silverstone"]
        return opts, {x: x for x in opts}
    return options, display_to_event


_SESSION_NAME_TO_CODE = {
    "practice 1": "FP1",
    "practice 2": "FP2",
    "practice 3": "FP3",
    "fp1": "FP1",
    "fp2": "FP2",
    "fp3": "FP3",
    "qualifying": "Q",
    "race": "R",
    "sprint": "S",
    "sprint qualifying": "SQ",
    "sprint shootout": "SQ",
    "shootout": "SQ",
}


def _available_session_codes(schedule: pd.DataFrame | None, event_name: str) -> list[str]:
    default = ["FP1", "FP2", "FP3", "Q", "SQ", "S", "R"]
    if schedule is None or len(schedule) == 0:
        return default

    try:
        rows = schedule[schedule["EventName"] == event_name]
        if rows is None or len(rows) == 0:
            return default
        row = rows.iloc[0]
    except Exception:
        return default

    # Schedules usually have Session1..Session5 as labels.
    codes: list[str] = []
    for i in range(1, 6):
        key = f"Session{i}"
        if key not in row.index:
            continue
        name = row.get(key, None)
        if name is None or (isinstance(name, float) and np.isnan(name)):
            continue
        code = _SESSION_NAME_TO_CODE.get(str(name).strip().lower())
        if code and code not in codes:
            codes.append(code)
    return codes or default


def _format_session(code: str) -> str:
    return {
        "R": "Race",
        "Q": "Qualifying",
        "FP1": "FP1",
        "FP2": "FP2",
        "FP3": "FP3",
        "SQ": "Sprint Shootout",
        "S": "Sprint",
    }.get(code, code)


# Sidebar
st.sidebar.title("Session & Drivers")
year = st.sidebar.number_input("Year", min_value=2018, max_value=2025, value=2023, step=1)

schedule = _get_schedule(int(year))
event_display_options, display_to_event = _build_event_options(schedule)
event_display = st.sidebar.selectbox("Course", options=event_display_options, index=0)
grand_prix = display_to_event.get(event_display, event_display)

session_codes = _available_session_codes(schedule, grand_prix)
session_choice = st.sidebar.selectbox("Session", options=session_codes, format_func=_format_session)

if "session_key" not in st.session_state:
    st.session_state.session_key = None
    st.session_state.session_obj = None
    st.session_state.driver_list = []
    st.session_state.laps_by_driver = {}

session_key = (int(year), str(grand_prix), str(session_choice))
if st.session_state.session_key != session_key:
    st.session_state.session_key = session_key
    st.session_state.session_obj = None
    st.session_state.driver_list = []
    st.session_state.laps_by_driver = {}
    try:
        with st.sidebar.spinner("Chargement session..."):
            sess = data_loader.load_session(int(year), grand_prix, session_choice, cache_path=str(_cache_dir))
        st.session_state.session_obj = sess
        drivers = sess.laps["Driver"].dropna().unique().tolist()
        if not drivers and hasattr(sess, "results") and sess.results is not None:
            drivers = sess.results["Abbreviation"].dropna().astype(str).tolist()
        st.session_state.driver_list = sorted(map(str, drivers)) if drivers else []
    except Exception as e:
        st.sidebar.warning(f"Impossible de charger la session: {e}")

driver_options = st.session_state.driver_list or ["VER", "HAM", "LEC", "NOR", "ALO"]
driver_a = st.sidebar.selectbox("Pilote A (référence)", options=driver_options, index=0)
driver_b = st.sidebar.selectbox("Pilote B", options=driver_options, index=min(1, len(driver_options) - 1))

if driver_a == driver_b:
    st.sidebar.error("Choisis deux pilotes différents.")


def _lap_choices_for_driver(driver: str) -> list[tuple[str, str | int]]:
    sess = st.session_state.session_obj
    if sess is None or not hasattr(sess, "laps"):
        return [("Fastest", "fastest")]
    try:
        laps = sess.laps.pick_driver(driver)
        # Keep only laps with LapTime and LapNumber
        laps = laps.dropna(subset=["LapNumber"])
        lap_numbers = sorted(set(int(x) for x in laps["LapNumber"].dropna().tolist()))
        out: list[tuple[str, str | int]] = [("Fastest", "fastest")]
        for n in lap_numbers:
            out.append((f"Lap {n}", n))
        return out
    except Exception:
        return [("Fastest", "fastest")]


lap_a_options = _lap_choices_for_driver(driver_a)
lap_b_options = _lap_choices_for_driver(driver_b)

lap_a = st.sidebar.selectbox(
    "Tour A",
    options=[v for _, v in lap_a_options],
    format_func=lambda v: {val: label for label, val in lap_a_options}.get(v, str(v)),
    index=0,
)
lap_b = st.sidebar.selectbox(
    "Tour B",
    options=[v for _, v in lap_b_options],
    format_func=lambda v: {val: label for label, val in lap_b_options}.get(v, str(v)),
    index=0,
)

# Main: Compare button and charts
st.title("Comparative Telemetry Analysis (V-A-G)")

if st.button("Load & Compare"):
    if driver_a == driver_b:
        st.error("Sélection invalide: Pilote A et Pilote B sont identiques.")
        st.stop()
    if st.session_state.session_obj is None:
        st.error("Session non chargée. Vérifie l’année / course / session.")
        st.stop()
    with st.spinner("Loading telemetry and computing..."):
        try:
            df_a, lap_meta_a = data_loader.load_driver_lap_telemetry(
                st.session_state.session_obj, driver_a, lap=lap_a
            )
            df_b, lap_meta_b = data_loader.load_driver_lap_telemetry(
                st.session_state.session_obj, driver_b, lap=lap_b
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

        # --- Metrics / enrichment
        col1, col2, col3, col4 = st.columns(4)
        delta_finish = float(dt[-1]) if len(dt) else float("nan")
        lap_dist = float(d[-1] - d[0]) if len(d) > 1 else float("nan")
        delta_integral = float(np.trapz(dt, d)) if len(d) > 1 else float("nan")  # s·m
        delta_avg = (delta_integral / lap_dist) if lap_dist and lap_dist > 0 else float("nan")

        with col1:
            st.metric("Δt fin de tour (B - A)", f"{delta_finish:+.3f} s")
        with col2:
            st.metric("Δt moyen (sur distance)", f"{delta_avg:+.3f} s")
        with col3:
            st.metric("Vitesse max A / B", f"{np.nanmax(da['Speed']):.0f} / {np.nanmax(db['Speed']):.0f} km/h")
        with col4:
            st.metric("Vitesse min A / B", f"{np.nanmin(da['Speed']):.0f} / {np.nanmin(db['Speed']):.0f} km/h")

        with st.expander("Détails tours (secteurs)"):
            def _fmt_td(x):
                if x is None or (isinstance(x, float) and np.isnan(x)) or (hasattr(pd, "isna") and pd.isna(x)):
                    return None
                try:
                    # Timedelta-like
                    if hasattr(x, "total_seconds"):
                        s = x.total_seconds()
                        m = int(s // 60)
                        r = s - 60 * m
                        return f"{m}:{r:06.3f}"
                except Exception:
                    pass
                return str(x)

            st.write(
                pd.DataFrame(
                    [
                        {
                            "Driver": driver_a,
                            "LapNumber": lap_meta_a.get("LapNumber"),
                            "LapTime": _fmt_td(lap_meta_a.get("LapTime")),
                            "S1": _fmt_td(lap_meta_a.get("Sector1Time")),
                            "S2": _fmt_td(lap_meta_a.get("Sector2Time")),
                            "S3": _fmt_td(lap_meta_a.get("Sector3Time")),
                        },
                        {
                            "Driver": driver_b,
                            "LapNumber": lap_meta_b.get("LapNumber"),
                            "LapTime": _fmt_td(lap_meta_b.get("LapTime")),
                            "S1": _fmt_td(lap_meta_b.get("Sector1Time")),
                            "S2": _fmt_td(lap_meta_b.get("Sector2Time")),
                            "S3": _fmt_td(lap_meta_b.get("Sector3Time")),
                        },
                    ]
                )
            )

        # Braking points (distance)
        brake_pts_a = processor.detect_brake_onsets(d, da["Brake"].values, threshold=0.5, debounce_m=30.0)
        brake_pts_b = processor.detect_brake_onsets(d, db["Brake"].values, threshold=0.5, debounce_m=30.0)

        def _points_to_xy(points: np.ndarray, speed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            if points.size == 0:
                return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
            idx = np.clip(np.searchsorted(d, points), 0, len(d) - 1)
            return points, speed[idx]

        bp_a_xy = _points_to_xy(brake_pts_a, da["Speed"].values)
        bp_b_xy = _points_to_xy(brake_pts_b, db["Speed"].values)

        # Subplots: Delta, Speed, Throttle/Brake, Gear
        fig_delta = components.plot_time_delta(d, dt)
        fig_speed = components.plot_speed_comparison(
            d,
            da["Speed"].values,
            db["Speed"].values,
            driver_a,
            driver_b,
            brake_points_a=bp_a_xy,
            brake_points_b=bp_b_xy,
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

        # Overlays (DRS / ERS if available)
        drs_a = da["DRS"].values if "DRS" in da.columns and "DRS" in db.columns else None
        drs_b = db["DRS"].values if "DRS" in da.columns and "DRS" in db.columns else None
        ers_col = None
        for c in ("ERSDeployMode", "ERSDeploy", "ERSPower", "ERS"):
            if c in da.columns and c in db.columns:
                ers_col = c
                break
        ers_a = da[ers_col].values if ers_col else None
        ers_b = db[ers_col].values if ers_col else None

        if drs_a is not None or ers_col is not None:
            st.subheader("Overlays (DRS / ERS)")
            fig_ov = components.plot_overlays(
                d,
                driver_a,
                driver_b,
                drs_a=drs_a,
                drs_b=drs_b,
                ers_a=ers_a,
                ers_b=ers_b,
                title=f"Overlays (DRS / {ers_col or 'ERS'})",
            )
            st.plotly_chart(fig_ov, use_container_width=True)

        with st.expander("Points de freinage (distance)"):
            max_len = int(max(len(brake_pts_a), len(brake_pts_b)))
            df_brk = pd.DataFrame(
                {
                    f"{driver_a} (m)": list(brake_pts_a) + [np.nan] * (max_len - len(brake_pts_a)),
                    f"{driver_b} (m)": list(brake_pts_b) + [np.nan] * (max_len - len(brake_pts_b)),
                }
            )
            st.dataframe(df_brk, use_container_width=True, height=250)

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

        # --- Exports
        st.subheader("Exports")
        export_cols = ["Time", "Speed", "RPM", "Gear", "Throttle", "Brake", "X", "Y", "Z", "Gx", "Gy", "DRS", "ERSDeployMode", "ERSDeploy", "ERSPower", "ERS"]
        out = pd.DataFrame({"Distance_m": d, "Delta_t_s": dt})
        for col in export_cols:
            if col in da.columns:
                out[f"{driver_a}_{col}"] = da[col].values
            if col in db.columns:
                out[f"{driver_b}_{col}"] = db[col].values
        csv_buf = io.StringIO()
        out.to_csv(csv_buf, index=False)
        st.download_button(
            "Télécharger CSV (signaux resynchronisés)",
            data=csv_buf.getvalue().encode("utf-8"),
            file_name=f"telemetry_resynced_{year}_{grand_prix}_{session_choice}_{driver_a}_vs_{driver_b}.csv",
            mime="text/csv",
        )

        def _fig_to_png_bytes(fig):
            try:
                return fig.to_image(format="png", scale=2)
            except Exception:
                return None

        figs = {
            "delta_t.png": fig_delta,
            "speed.png": fig_speed,
            "throttle_brake.png": fig_tb,
            "gear.png": fig_gear,
        }
        if drs_a is not None or ers_col is not None:
            figs["overlays.png"] = fig_ov
        if "Gx" in da.columns and "Gy" in da.columns:
            figs["kamm_circle.png"] = fig_kamm

        st.caption("Pour l’export PNG, `kaleido` doit être installé côté serveur (Streamlit Cloud).")
        any_png = False
        for filename, fig in figs.items():
            png = _fig_to_png_bytes(fig)
            if png is None:
                continue
            any_png = True
            st.download_button(
                f"Télécharger {filename}",
                data=png,
                file_name=filename,
                mime="image/png",
            )
        if not any_png:
            st.warning("Export PNG indisponible (probablement `kaleido` manquant).")

else:
    st.info("Sélectionne année/course/session/pilotes dans la barre latérale, puis clique sur **Load & Compare**.")
