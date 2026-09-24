
import streamlit as st
import fastf1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from aero_analysis import (g_forces, corner_table, downforce_fit, grid_aero_map,
                           drag_fit_auto, aero_summary)

st.set_page_config(page_title="F1 Telemetry Comparison", layout="wide")

# FastF1 keeps downloaded sessions in its own cache folder (outside this project).

st.title("🏎️ F1 Driver Telemetry Comparison")
st.caption("FastF1 API · sector/corner analysis · G-force estimation · DRS effect · Butterworth filtering · downforce estimate from grip vs speed")

#  Sidebar controls 
with st.sidebar:
    st.header("Session")
    year = st.selectbox("Year", [2025, 2024, 2023, 2022, 2021, 2020, 2019, 2018], index=1)
    gp = st.text_input("Grand Prix", value="Monza")
    session_type = st.selectbox("Session", ["Q", "R", "FP1", "FP2", "FP3"], index=0)
    driver_1 = st.text_input("Driver 1 code", value="VER").upper()
    driver_2 = st.text_input("Driver 2 code", value="LEC").upper()
    n_corners = st.slider("Top corners to report", 3, 10, 5)
    st.header("Signal processing")
    filter_cutoff = st.slider("Low-pass filter cutoff (Hz)", 0.5, 5.0, 2.0, step=0.5,
                               help="Butterworth low-pass applied to raw ~10Hz speed signal to remove sensor noise before computing derivatives (acceleration).")
    st.header("Aero")
    car_mass = st.number_input("Car + driver mass (kg)", 700.0, 900.0, 800.0, step=5.0,
                               help="Minimum is 798 kg for 2022-24 and 800 kg for 2025, including the driver; "
                                    "earlier cars were lighter. ClA and CdA scale with this number; L/D doesn't.")
    show_grid = st.checkbox("Whole-grid aero map (slower)", value=False)
    run = st.button("Analyze", type="primary")

#  Cached data loading 
@st.cache_data(show_spinner=False)
def load_session_data(year, gp, session_type, driver_1, driver_2):
    session = fastf1.get_session(year, gp, session_type)
    session.load(telemetry=True, laps=True, weather=True)

    lap1 = session.laps.pick_drivers(driver_1).pick_fastest()
    lap2 = session.laps.pick_drivers(driver_2).pick_fastest()
    if lap1 is None or lap2 is None:
        raise ValueError("No valid fastest lap for one of the drivers in this session.")
    tel1 = lap1.get_telemetry()
    tel2 = lap2.get_telemetry()

    # Air density from the session weather: rho = p / (R T).
    try:
        w = session.weather_data
        rho = float(w["Pressure"].median()) * 100 / (287.05 * (float(w["AirTemp"].median()) + 273.15))
        if not 1.0 < rho < 1.35:
            rho = 1.20
    except Exception:
        rho = 1.20
    return lap1, lap2, tel1, tel2, rho


@st.cache_data(show_spinner=False)
def load_grid_map(year, gp, session_type):
    session = fastf1.get_session(year, gp, session_type)
    session.load(telemetry=True, laps=True, weather=False)
    grid, fast = grid_aero_map(session)
    return grid, (len(fast) if fast is not None else 0)

def build_comparison(tel1, tel2, d1, d2):
    common_distance = np.linspace(
        max(tel1['Distance'].min(), tel2['Distance'].min()),
        min(tel1['Distance'].max(), tel2['Distance'].max()),
        2000
    )
    def interp(tel, col):
        return np.interp(common_distance, tel['Distance'], tel[col])

    df = pd.DataFrame({
        'Distance': common_distance,
        f'Speed_{d1}': interp(tel1, 'Speed'),
        f'Speed_{d2}': interp(tel2, 'Speed'),
        f'Throttle_{d1}': interp(tel1, 'Throttle'),
        f'Brake_{d1}': interp(tel1, 'Brake'),
    })
    df['SpeedDelta'] = df[f'Speed_{d1}'] - df[f'Speed_{d2}']
    return df

def corner_min_speed_comparison(tel1, tel2, d1, d2, n_report=5):
    inv_speed = -tel1['Speed'].values
    peaks, _ = find_peaks(inv_speed, distance=50, prominence=10)
    corner_distances = tel1['Distance'].values[peaks]

    results = []
    for cd in corner_distances:
        idx1 = (tel1['Distance'] - cd).abs().idxmin()
        idx2 = (tel2['Distance'] - cd).abs().idxmin()
        s1 = tel1.loc[idx1, 'Speed']
        s2 = tel2.loc[idx2, 'Speed']
        results.append({'Distance_m': round(cd, 1), f'{d1}_MinSpeed_kmh': s1,
                         f'{d2}_MinSpeed_kmh': s2, 'Delta': round(s1 - s2, 1)})
    return pd.DataFrame(results).sort_values('Delta', key=abs, ascending=False).head(n_report)

#  Signal filtering and G-forces live in aero_analysis.py:
#  resample to an even 10 Hz grid -> Butterworth low-pass -> differentiate.
#  (FastF1 X/Y are in 1/10 m; they are converted to metres there.)

#  DRS analysis 

def drs_analysis(tel1, tel2, d1, d2):
    """
    FastF1's DRS channel: odd values (10,12,14) generally = DRS open/active,
    even values (8) = available but closed, 0/1 = unavailable. We treat any
    value >= 10 as 'active' — this is the standard FastF1 convention.
    Compares average speed gain while DRS is active vs the run-up before it,
    and reports total distance with DRS active as a proxy for zone exposure.
    """
    def summarize(tel, driver):
        drs_active = tel['DRS'].fillna(0) >= 10
        pct_active = 100 * drs_active.mean()
        avg_speed_drs = tel.loc[drs_active, 'Speed'].mean() if drs_active.any() else np.nan
        avg_speed_no_drs = tel.loc[~drs_active, 'Speed'].mean()
        return {
            'Driver': driver,
            'DRS active (% of lap)': round(pct_active, 1),
            'Avg speed w/ DRS (km/h)': round(avg_speed_drs, 1) if pd.notna(avg_speed_drs) else np.nan,
            'Avg speed w/o DRS (km/h)': round(avg_speed_no_drs, 1),
        }

    if 'DRS' not in tel1.columns or 'DRS' not in tel2.columns:
        return None

    return pd.DataFrame([summarize(tel1, d1), summarize(tel2, d2)])

#  Main 
if run:
    with st.spinner(f"Loading {year} {gp} {session_type}..."):
        try:
            lap1, lap2, tel1, tel2, rho = load_session_data(year, gp, session_type, driver_1, driver_2)
        except Exception as e:
            st.error(f"Couldn't load session: {e}")
            st.stop()

    col1, col2 = st.columns(2)
    def fmt_lap(td):
        secs = td.total_seconds()
        return f"{int(secs // 60)}:{secs % 60:06.3f}"
    col1.metric(f"{driver_1} fastest lap", fmt_lap(lap1['LapTime']))
    col2.metric(f"{driver_2} fastest lap", fmt_lap(lap2['LapTime']))

    df = build_comparison(tel1, tel2, driver_1, driver_2)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(df['Distance'], df[f'Speed_{driver_1}'], label=driver_1)
    axes[0].plot(df['Distance'], df[f'Speed_{driver_2}'], label=driver_2)
    axes[0].set_ylabel('Speed (km/h)')
    axes[0].legend()
    axes[1].plot(df['Distance'], df['SpeedDelta'], color='purple')
    axes[1].axhline(0, color='black', linewidth=0.5)
    axes[1].set_ylabel(f'Δ Speed ({driver_1}-{driver_2})')
    axes[1].set_xlabel('Distance (m)')
    st.pyplot(fig)
    plt.close(fig)

    st.subheader(f"Top {n_corners} corner speed deltas")
    corners_df = corner_min_speed_comparison(tel1, tel2, driver_1, driver_2, n_corners)
    st.dataframe(corners_df, width="stretch")

    st.subheader("Sector times")
    def sector_s(lap):
        return [round(pd.Timedelta(lap[f'Sector{i}Time']).total_seconds(), 3) for i in (1, 2, 3)]
    sectors = pd.DataFrame({
        'Sector': ['S1', 'S2', 'S3'],
        f'{driver_1} (s)': sector_s(lap1),
        f'{driver_2} (s)': sector_s(lap2),
    })
    sectors['Delta (s)'] = (sectors[f'{driver_1} (s)'] - sectors[f'{driver_2} (s)']).round(3)
    st.dataframe(sectors, width="stretch")

    #  G-forces 
    st.subheader("Estimated G-forces")
    st.caption(
        f"Telemetry resampled to an even 10 Hz grid, then filtered with a 3rd-order zero-phase "
        f"Butterworth low-pass (speed {filter_cutoff} Hz, position 1 Hz) before differentiating. "
        f"Lateral g = v² × path curvature, with X/Y converted from 1/10 m to m."
    )
    tel1_g = g_forces(tel1, cutoff_hz=filter_cutoff)
    tel2_g = g_forces(tel2, cutoff_hz=filter_cutoff)

    fig_g, axes_g = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes_g[0].plot(tel1_g['Distance'], tel1_g['LongG'], label=f'{driver_1} Long. G', alpha=0.8)
    axes_g[0].plot(tel2_g['Distance'], tel2_g['LongG'], label=f'{driver_2} Long. G', alpha=0.8)
    axes_g[0].axhline(0, color='black', linewidth=0.5)
    axes_g[0].set_ylabel('Longitudinal G')
    axes_g[0].legend()
    axes_g[0].set_title('Braking (negative) / Acceleration (positive)')

    axes_g[1].plot(tel1_g['Distance'], tel1_g['LatG'], label=f'{driver_1} Lat. G', alpha=0.8)
    axes_g[1].plot(tel2_g['Distance'], tel2_g['LatG'], label=f'{driver_2} Lat. G', alpha=0.8)
    axes_g[1].set_ylabel('Lateral G (magnitude)')
    axes_g[1].set_xlabel('Distance (m)')
    axes_g[1].legend()
    st.pyplot(fig_g)
    plt.close(fig_g)

    col_g1, col_g2 = st.columns(2)
    col_g1.metric(f"{driver_1} peak braking G", f"{tel1_g['LongG'].min():.2f} G")
    col_g1.metric(f"{driver_1} peak lateral G", f"{tel1_g['LatG'].max():.2f} G")
    col_g2.metric(f"{driver_2} peak braking G", f"{tel2_g['LongG'].min():.2f} G")
    col_g2.metric(f"{driver_2} peak lateral G", f"{tel2_g['LatG'].max():.2f} G")

    #  DRS 
    st.subheader("DRS (drag reduction system) effect")
    drs_df = drs_analysis(tel1, tel2, driver_1, driver_2)
    if drs_df is not None:
        st.caption("DRS 'active' = FastF1 DRS channel value ≥ 10 (open), per FastF1 convention.")
        st.dataframe(drs_df, width="stretch")
    else:
        st.info("DRS channel not available for this session (e.g. practice sessions before DRS is enabled, or non-DRS-eligible tracks).")

    #  Aero: grip vs speed -> downforce 
    st.subheader("Aero: downforce, drag and L/D")
    st.caption(
        "At the grip limit, m·a_y = μ(m·g + ½ρv²·ClA), so lateral g = μ + (μρ·ClA / 2mg)·v². "
        "Fitting apex lateral g against v² over every corner gives tyre friction μ (intercept) and "
        "downforce area ClA (slope). Apex radius comes from a circle fitted to the filtered racing line. "
        "Drag: at full throttle, m·v·a = P − ½ρ·CdA·v³ (plus rolling and slope terms), fitted on the straights. "
        f"Air density from session weather: {rho:.3f} kg/m³."
    )
    fits = {}
    fig_a, ax_a = plt.subplots(figsize=(10, 4.5))
    for tel_g, drv, colour in [(tel1_g, driver_1, "#2a78d6"), (tel2_g, driver_2, "#eb6834")]:
        corners = corner_table(tel_g)
        fit = downforce_fit(corners, mass_kg=car_mass, rho=rho)
        fits[drv] = (corners, fit)
        if fit is None:
            continue
        used = fit["used"]
        vv = np.linspace(0, max(used["ApexSpeed_kmh"].max() * 1.08, 300), 200)
        ax_a.plot(vv, fit["mu"] + fit["slope"] * (vv / 3.6) ** 2, color=colour,
                  label=f"{drv} fit: μ={fit['mu']:.2f}, ClA={fit['ClA']:.1f}±{fit['ClA_err']:.1f} m²")
        ax_a.plot(used["ApexSpeed_kmh"], used["LatG"], "o", color=colour, alpha=0.8)
    ax_a.set_xlabel("Apex speed (km/h)")
    ax_a.set_ylabel("Lateral acceleration (g)")
    ax_a.set_xlim(0, None)
    ax_a.set_ylim(0, None)
    ax_a.legend()
    st.pyplot(fig_a)
    plt.close(fig_a)

    cols = st.columns(2)
    for col, drv in zip(cols, [driver_1, driver_2]):
        corners, fit = fits[drv]
        if fit is None:
            col.warning(f"{drv}: fewer than 4 grip-limited corners found, so no fit.")
            continue
        col.metric(f"{drv} tyre friction μ", f"{fit['mu']:.2f} ± {fit['mu_err']:.2f}")
        col.metric(f"{drv} downforce area ClA", f"{fit['ClA']:.2f} ± {fit['ClA_err']:.2f} m²")
        col.metric(f"{drv} downforce / weight at 250 km/h", f"{fit['L_over_W_250']:.2f}")
        tel_g = tel1_g if drv == driver_1 else tel2_g
        drag, note = drag_fit_auto(tel_g, mass_kg=car_mass, rho=rho)
        if drag is not None:
            col.metric(f"{drv} drag area CdA (DRS closed)", f"{drag['CdA']:.2f} ± {drag['CdA_err']:.2f} m²")
            summ = aero_summary(fit, drag)
            if summ:
                col.metric(f"{drv} aero efficiency L/D", f"{summ['L_over_D']:.2f} ± {summ['L_over_D_err']:.2f}")
            extra = (f"; DRS saves {drag['dCdA_DRS']:.2f} m² of CdA"
                     if np.isfinite(drag["dCdA_DRS"]) else "")
            col.caption(f"Wheel power {drag['power_kW']:.0f} kW{extra}. {note[:1].upper() + note[1:]}.")
            if not drag["physical"]:
                col.warning("Drag fit outside the sensible range (CdA 0.6–2.0 m², power 400–900 kW).")
        col.caption(f"R² = {fit['r2']:.3f} over {fit['n']} corners")
        if not fit["physical"]:
            col.warning("Outside the physically sensible range (μ 1–2.5, ClA 2–7 m²): treat as unreliable.")
        col.dataframe(corners, width="stretch")

    #  Aero: whole-grid map 
    if show_grid:
        st.subheader("Aero: straight-line speed vs high-speed-corner speed (whole grid)")
        with st.spinner("Loading every driver's fastest lap..."):
            grid, n_fast = load_grid_map(year, gp, session_type)
        if grid is None or grid["FastCornerSpeed_kmh"].notna().sum() < 5:
            st.info("Not enough valid laps for a grid map in this session.")
        else:
            grid = grid.dropna(subset=["FastCornerSpeed_kmh", "SpeedTrap_kmh"])
            fig_m, ax_m = plt.subplots(figsize=(10, 5))
            ax_m.plot(grid["FastCornerSpeed_kmh"], grid["SpeedTrap_kmh"], "o", color="#2a78d6")
            for _, r in grid.iterrows():
                ax_m.annotate(r["Driver"], (r["FastCornerSpeed_kmh"], r["SpeedTrap_kmh"]),
                              xytext=(4, 3), textcoords="offset points", fontsize=8)
            ax_m.axvline(grid["FastCornerSpeed_kmh"].median(), color="grey", ls=":")
            ax_m.axhline(grid["SpeedTrap_kmh"].median(), color="grey", ls=":")
            ax_m.set_xlabel(f"Mean minimum speed, {n_fast} fastest corners (km/h)")
            ax_m.set_ylabel("Speed trap (km/h)")
            st.pyplot(fig_m)
            plt.close(fig_m)
            st.caption("Top-left: low-drag set-up. Bottom-right: high-downforce set-up. "
                       "Speed trap also depends on power unit, tow and battery deployment.")
            st.dataframe(grid.sort_values("LapTime_s").round({"LapTime_s": 3, "SpeedTrap_kmh": 1, "FastCornerSpeed_kmh": 1}), width="stretch")
else:
    st.info("Set your session and drivers in the sidebar, then click **Analyze**.")
