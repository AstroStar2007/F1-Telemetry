
import streamlit as st
import fastf1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks, butter, filtfilt

st.set_page_config(page_title="F1 Telemetry Comparison", layout="wide")

CACHE_DIR = "f1_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

st.title("🏎️ F1 Driver Telemetry Comparison")
st.caption("FastF1 API · 10Hz telemetry · sector/corner analysis · G-force estimation · DRS effect · Butterworth filtering")

#  Sidebar controls 
with st.sidebar:
    st.header("Session")
    year = st.selectbox("Year", [2024, 2023, 2022, 2021], index=0)
    gp = st.text_input("Grand Prix", value="Monza")
    session_type = st.selectbox("Session", ["Q", "R", "FP1", "FP2", "FP3"], index=0)
    driver_1 = st.text_input("Driver 1 code", value="VER").upper()
    driver_2 = st.text_input("Driver 2 code", value="LEC").upper()
    n_corners = st.slider("Top corners to report", 3, 10, 5)
    st.header("Signal processing")
    filter_cutoff = st.slider("Low-pass filter cutoff (Hz)", 0.5, 5.0, 2.0, step=0.5,
                               help="Butterworth low-pass applied to raw ~10Hz speed signal to remove sensor noise before computing derivatives (acceleration).")
    run = st.button("Analyze", type="primary")

#  Cached data loading 
@st.cache_data(show_spinner=False)
def load_session_data(year, gp, session_type, driver_1, driver_2):
    session = fastf1.get_session(year, gp, session_type)
    session.load(telemetry=True, laps=True, weather=False)

    lap1 = session.laps.pick_drivers(driver_1).pick_fastest()
    lap2 = session.laps.pick_drivers(driver_2).pick_fastest()
    tel1 = lap1.get_telemetry()
    tel2 = lap2.get_telemetry()
    return lap1, lap2, tel1, tel2

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

#  Signal filtering 

def apply_lowpass_filter(signal, sample_rate_hz, cutoff_hz):
    """
    Zero-phase Butterworth low-pass filter (order 3), applied forward+backward
    via filtfilt to avoid phase lag — standard practice for post-processing
    noisy sensor/telemetry channels before differentiating.
    """
    nyquist = sample_rate_hz / 2
    normal_cutoff = min(cutoff_hz / nyquist, 0.99)
    b, a = butter(3, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)

def estimate_sample_rate(tel):
    """Estimate telemetry sample rate from the Time channel."""
    dt = tel['Time'].diff().dt.total_seconds().dropna()
    dt = dt[dt > 0]
    return 1.0 / dt.median() if len(dt) else 10.0

#  G-force estimation 

def compute_g_forces(tel, cutoff_hz):
    """
    Longitudinal g: from filtered speed vs time (dv/dt).
    Lateral g: from track curvature (X,Y position) and speed — v^2 * curvature.
    This mirrors how you'd estimate accelerations from raw IMU/position data
    in a vehicle dynamics or flight-test context: filter first, then differentiate.
    """
    tel = tel.copy().reset_index(drop=True)
    fs = estimate_sample_rate(tel)

    speed_ms = tel['Speed'].values / 3.6  # km/h -> m/s
    speed_filt = apply_lowpass_filter(speed_ms, fs, cutoff_hz)

    t = tel['Time'].dt.total_seconds().values
    dt = np.gradient(t)
    dt[dt == 0] = np.nan
    long_accel = np.gradient(speed_filt) / dt
    long_g = long_accel / 9.81

    x = apply_lowpass_filter(tel['X'].values, fs, cutoff_hz)
    y = apply_lowpass_filter(tel['Y'].values, fs, cutoff_hz)
    dx = np.gradient(x, dt)
    dy = np.gradient(y, dt)
    ddx = np.gradient(dx, dt)
    ddy = np.gradient(dy, dt)
    denom = (dx**2 + dy**2)**1.5
    denom[denom == 0] = np.nan
    curvature = np.abs(dx * ddy - dy * ddx) / denom  # 1/m

    lat_accel = speed_filt**2 * curvature
    lat_g = lat_accel / 9.81

    tel['SpeedFiltered'] = speed_filt * 3.6  # back to km/h for plotting
    tel['LongG'] = pd.Series(long_g).clip(-6, 6).values
    tel['LatG'] = pd.Series(lat_g).clip(0, 8).values
    return tel

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
            lap1, lap2, tel1, tel2 = load_session_data(year, gp, session_type, driver_1, driver_2)
        except Exception as e:
            st.error(f"Couldn't load session: {e}")
            st.stop()

    col1, col2 = st.columns(2)
    col1.metric(f"{driver_1} fastest lap", str(lap1['LapTime']))
    col2.metric(f"{driver_2} fastest lap", str(lap2['LapTime']))

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

    st.subheader(f"Top {n_corners} corner speed deltas")
    corners_df = corner_min_speed_comparison(tel1, tel2, driver_1, driver_2, n_corners)
    st.dataframe(corners_df, use_container_width=True)

    st.subheader("Sector times")
    sectors = pd.DataFrame({
        'Sector': ['S1', 'S2', 'S3'],
        driver_1: [lap1['Sector1Time'], lap1['Sector2Time'], lap1['Sector3Time']],
        driver_2: [lap2['Sector1Time'], lap2['Sector2Time'], lap2['Sector3Time']],
    })
    st.dataframe(sectors, use_container_width=True)

    #  G-forces 
    st.subheader("Estimated G-forces")
    st.caption(
        f"Speed and position filtered with a 3rd-order zero-phase Butterworth low-pass "
        f"({filter_cutoff} Hz cutoff) before differentiating, to suppress sensor noise "
        f"amplification in the acceleration estimate."
    )
    tel1_g = compute_g_forces(tel1, filter_cutoff)
    tel2_g = compute_g_forces(tel2, filter_cutoff)

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
        st.dataframe(drs_df, use_container_width=True)
    else:
        st.info("DRS channel not available for this session (e.g. practice sessions before DRS is enabled, or non-DRS-eligible tracks).")
else:
    st.info("Set your session and drivers in the sidebar, then click **Analyze**.")
