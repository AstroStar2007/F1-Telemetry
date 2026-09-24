"""
aero_analysis.py - vehicle-dynamics and aero estimates from FastF1 telemetry.

1. uniform_lap():      resample a lap onto an even time grid (Butterworth
                       filters assume even sampling; FastF1's merged
                       telemetry is uneven) and convert X/Y from 1/10 m to m.
2. g_forces():         longitudinal and lateral acceleration from filtered
                       speed and position.
3. corner_table():     every corner apex: speed, peak lateral g, throttle.
4. downforce_fit():    grip-vs-speed fit that estimates tyre friction (mu)
                       and downforce area (ClA) from one lap.
5. drag_fit():         power-vs-drag fit on full-throttle straights that
                       estimates drag area (CdA), wheel power and the drag
                       saved by DRS. drag_fit_auto() also checks for
                       hybrid-battery clipping at top speed.
6. aero_summary():     downforce + drag together: L/D = ClA / CdA.
7. grid_aero_map():    straight-line speed vs high-speed-corner speed for
                       every driver (low-drag vs high-downforce set-ups).

The physics behind downforce_fit, for a car cornering at its grip limit:

    m * a_y = mu * (m * g + L),   with  L = 0.5 * rho * v^2 * ClA

    =>  a_y / g = mu + (mu * rho * ClA / (2 * m * g)) * v^2

So lateral g plotted against v^2 is a straight line: the intercept is the
tyre friction coefficient mu, and the slope gives ClA. Slow corners pin
down mu (almost no downforce), fast corners pin down ClA.

The physics behind drag_fit, at full throttle on a straight:

    m * a = P / v - 0.5 * rho * CdA * v^2 - Crr * m * g - m * g * sin(theta)

Multiply by v and move the known terms left:

    m * v * (a + Crr * g + g * sin(theta)) = P - 0.5 * rho * CdA * v^3

A straight line in v^3: the intercept is wheel power P and the slope gives
CdA. With DRS open the slope is shallower; an extra term measures that.
Because ClA and CdA both scale with the assumed mass and air density,
L/D = ClA / CdA does not depend on either assumption.
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

G = 9.81  # m/s^2


# ---------------------------------------------------------------------------
# 1. Resampling and filtering
# ---------------------------------------------------------------------------

def lowpass(signal, fs_hz, cutoff_hz, order=3):
    """Zero-phase Butterworth low-pass (forward + backward, so no lag)."""
    wn = min(cutoff_hz / (fs_hz / 2.0), 0.99)
    b, a = butter(order, wn, btype="low")
    return filtfilt(b, a, signal)


def uniform_lap(tel, fs_hz=10.0):
    """Resample one lap of FastF1 telemetry onto an even time grid.

    Returns a DataFrame with t [s], Distance [m], Speed [km/h], X, Y, Z [m],
    Throttle [%], DRS, nGear, Brake (when present).
    """
    t = tel["Time"].dt.total_seconds().to_numpy()
    keep = np.concatenate([[True], np.diff(t) > 1e-6])   # drop repeated stamps
    t = t[keep]
    grid = np.arange(t[0], t[-1], 1.0 / fs_hz)

    out = {"t": grid - grid[0]}
    for col, scale in [("Distance", 1.0), ("Speed", 1.0), ("X", 0.1), ("Y", 0.1),
                       ("Z", 0.1), ("Throttle", 1.0)]:
        if col in tel.columns:
            # X/Y/Z: 1/10 m -> m. Missing samples are skipped, not interpolated through.
            vals = pd.to_numeric(tel[col], errors="coerce").to_numpy(dtype=float)[keep] * scale
            ok = np.isfinite(vals)
            if ok.sum() >= 2:
                out[col] = np.interp(grid, t[ok], vals[ok])
    idx = np.clip(np.searchsorted(t, grid), 0, len(t) - 1)
    for col in ("DRS", "nGear"):                      # step channels: nearest sample
        if col in tel.columns:
            vals = pd.to_numeric(tel[col], errors="coerce").fillna(0).to_numpy()[keep]
            out[col] = vals[idx]
    if "Brake" in tel.columns:                        # missing brake sample = not braking
        vals = tel["Brake"].to_numpy()[keep]
        out["Brake"] = np.where(pd.isna(vals), False, vals).astype(bool)[idx]
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# 2. Accelerations
# ---------------------------------------------------------------------------

def g_forces(tel, cutoff_hz=2.0, pos_cutoff_hz=1.0, fs_hz=10.0):
    """Longitudinal and lateral g for one lap.

    Longitudinal: d(speed)/dt.
    Lateral: v^2 * curvature, with curvature from the filtered X/Y path:
        kappa = |x' y'' - y' x''| / (x'^2 + y'^2)^1.5   (derivatives in time)
    Everything is filtered on an even grid BEFORE differentiating, because
    differentiation amplifies high-frequency noise. Position is differentiated
    twice, so it gets a lower cutoff (pos_cutoff_hz) than speed (cutoff_hz).
    """
    u = uniform_lap(tel, fs_hz)
    dt = 1.0 / fs_hz

    v = lowpass(u["Speed"].to_numpy() / 3.6, fs_hz, cutoff_hz)       # m/s
    x = lowpass(u["X"].to_numpy(), fs_hz, pos_cutoff_hz)
    y = lowpass(u["Y"].to_numpy(), fs_hz, pos_cutoff_hz)

    dx, dy = np.gradient(x, dt), np.gradient(y, dt)
    ddx, ddy = np.gradient(dx, dt), np.gradient(dy, dt)
    denom = np.maximum((dx**2 + dy**2) ** 1.5, 1e-6)
    kappa = np.abs(dx * ddy - dy * ddx) / denom                      # 1/m

    u["XF"], u["YF"] = x, y
    if "Z" in u:
        # Elevation changes slowly; filter hard so grade = dZ/ds is usable.
        u["ZF"] = lowpass(u["Z"].to_numpy(), fs_hz, 0.2)
    u["SpeedFiltered"] = v * 3.6
    u["LongG"] = np.gradient(v, dt) / G
    u["LatG"] = v**2 * kappa / G
    u["Curvature"] = kappa
    return u


# ---------------------------------------------------------------------------
# 3. Corners
# ---------------------------------------------------------------------------

def fit_circle(x, y):
    """Least-squares (Kasa) circle through points; returns radius in m."""
    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x**2 + y**2)
    (D, E, F), *_ = np.linalg.lstsq(A, b, rcond=None)
    return float(np.sqrt(max(D**2 / 4 + E**2 / 4 - F, 0.0)))


def corner_table(g_lap, prominence_kmh=10.0, min_gap_s=1.5, window_s=0.75):
    """One row per corner apex (local minimum of speed).

    Corner radius comes from a circle fitted to the filtered racing line over
    +/- window_s around the apex. Fitting many points averages out position
    noise, which a pointwise second derivative would amplify. Lateral g at
    the apex is then v^2 / (R g).
    """
    fs = 1.0 / (g_lap["t"].iloc[1] - g_lap["t"].iloc[0])
    speed = g_lap["SpeedFiltered"].to_numpy()
    x, y = g_lap["XF"].to_numpy(), g_lap["YF"].to_numpy()
    peaks, _ = find_peaks(-speed, prominence=prominence_kmh, distance=int(min_gap_s * fs))
    w = max(int(window_s * fs), 3)
    rows = []
    for p in peaks:
        lo, hi = max(p - w, 0), min(p + w + 1, len(speed))
        radius = fit_circle(x[lo:hi], y[lo:hi])
        v = speed[p] / 3.6
        rows.append({
            "Distance_m": round(float(g_lap["Distance"].iloc[p]), 1),
            "ApexSpeed_kmh": round(float(speed[p]), 1),
            "Radius_m": round(radius, 1),
            "LatG": round(v**2 / radius / G, 2) if radius > 0 else np.nan,
            # Lowest throttle around the apex: 100 means the driver never
            # lifted, so the car was not necessarily at its grip limit.
            "MinThrottle_pct": round(float(g_lap["Throttle"].iloc[lo:hi].min()), 0)
            if "Throttle" in g_lap else np.nan,
        })
    return pd.DataFrame(rows, columns=["Distance_m", "ApexSpeed_kmh", "Radius_m", "LatG",
                                       "MinThrottle_pct"])


# ---------------------------------------------------------------------------
# 4. Grip and downforce from one lap
# ---------------------------------------------------------------------------

def downforce_fit(corners, mass_kg=800.0, rho=1.20, flat_out_pct=99.0):
    """Fit lateral g = mu + slope * v^2 across grip-limited corners.

    Corners where the driver never lifted (minimum throttle >= flat_out_pct
    around the apex) are excluded: the car was not necessarily at its grip
    limit there.
    Returns a dict with mu, ClA [m^2] (each with a 1-sigma fit error),
    downforce-to-weight at 250 km/h, R^2 and the corners used.
    """
    c = corners.dropna(subset=["LatG"]).copy()
    if "MinThrottle_pct" in c:
        c = c[~(c["MinThrottle_pct"] >= flat_out_pct)]
    c = c[c["LatG"] > 0.5]                      # ignore near-straight kinks
    if len(c) < 4:
        return None
    v2 = (c["ApexSpeed_kmh"].to_numpy() / 3.6) ** 2
    gy = c["LatG"].to_numpy()
    (slope, mu), cov = np.polyfit(v2, gy, 1, cov=True)
    pred = mu + slope * v2
    r2 = 1 - np.sum((gy - pred) ** 2) / np.sum((gy - gy.mean()) ** 2)
    cla = 2 * mass_kg * G * slope / (rho * mu) if mu > 0 else np.nan
    # Error propagation for ClA = const * slope / mu (1 sigma).
    rel = np.sqrt(cov[0, 0] / slope**2 + cov[1, 1] / mu**2 - 2 * cov[0, 1] / (slope * mu))
    v250 = 250 / 3.6
    return {"mu": mu, "mu_err": float(np.sqrt(cov[1, 1])),
            "ClA": cla, "ClA_err": float(abs(cla) * rel),
            "slope": slope,
            "L_over_W_250": 0.5 * rho * v250**2 * cla / (mass_kg * G),
            "r2": r2, "n": len(c), "used": c,
            "physical": bool(1.0 < mu < 2.5 and 2.0 < cla < 7.0)}


# ---------------------------------------------------------------------------
# 5. Drag and power from full-throttle straights
# ---------------------------------------------------------------------------

def drag_fit(g_lap, mass_kg=800.0, rho=1.20, crr=0.012, v_min_kmh=220.0,
             v_max_kmh=None, lat_g_max=1.0, settle_s=0.5, use_grade=True):
    """Fit m v (a + Crr g + g sin(theta)) = P - 0.5 rho v^3 (CdA - dCdA * DRS).

    Uses only samples at full throttle, not braking, on near-straight track
    (lateral g < lat_g_max), above v_min_kmh (below that F1 cars are
    traction-limited, not power-limited), after the throttle has been fully
    open for settle_s. v_max_kmh can cut off the end of long straights where
    the hybrid battery may stop deploying ("clipping"), which would otherwise
    read as extra drag.

    Returns a dict with power_kW, CdA, dCdA_DRS (each with 1-sigma errors),
    n samples, R^2 and the mask of samples used; or None if too few samples.
    """
    fs = 1.0 / (g_lap["t"].iloc[1] - g_lap["t"].iloc[0])
    v = g_lap["SpeedFiltered"].to_numpy() / 3.6
    a = g_lap["LongG"].to_numpy() * G

    full = g_lap["Throttle"].to_numpy() >= 99 if "Throttle" in g_lap else np.ones(len(v), bool)
    if "Brake" in g_lap:
        full &= ~g_lap["Brake"].astype(bool).to_numpy()
    # Time since the throttle reached 100 %: skip the first settle_s of each run.
    run = np.zeros(len(full))
    for i in range(1, len(full)):
        run[i] = run[i - 1] + 1 if full[i] else 0
    mask = full & (run >= settle_s * fs)
    mask &= (v * 3.6 >= v_min_kmh) & (g_lap["LatG"].to_numpy() < lat_g_max)
    if v_max_kmh is not None:
        mask &= v * 3.6 <= v_max_kmh

    sin_theta = np.zeros(len(v))
    grade_used = False
    if use_grade and "ZF" in g_lap:
        dist = g_lap["Distance"].to_numpy()
        ok = np.diff(dist, prepend=dist[0] - 1) > 0.05
        if ok.sum() > 10:
            sin_theta = np.clip(np.gradient(g_lap["ZF"].to_numpy(), dist + 1e-9 * np.arange(len(dist))),
                                -0.12, 0.12)
            grade_used = True
    mask &= np.isfinite(a) & np.isfinite(sin_theta)
    if mask.sum() < 30:
        return None

    vm, am, sm = v[mask], a[mask], sin_theta[mask]
    y = mass_kg * vm * (am + crr * G + G * sm)                     # watts
    cols = [np.ones_like(vm), -0.5 * rho * vm**3]
    drs_open = None
    if "DRS" in g_lap:
        drs_open = (g_lap["DRS"].to_numpy() >= 10)[mask]
        if drs_open.sum() >= 20 and (~drs_open).sum() >= 20:
            cols.append(0.5 * rho * vm**3 * drs_open)
        else:
            drs_open = None
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(len(y) - X.shape[1], 1)
    cov = (resid @ resid / dof) * np.linalg.inv(X.T @ X)
    err = np.sqrt(np.diag(cov))
    r2 = 1 - resid @ resid / np.sum((y - y.mean()) ** 2)

    out = {"power_kW": beta[0] / 1e3, "power_err": err[0] / 1e3,
           "CdA": beta[1], "CdA_err": err[1],
           "dCdA_DRS": beta[2] if drs_open is not None else np.nan,
           "dCdA_err": err[2] if drs_open is not None else np.nan,
           "n": int(mask.sum()), "r2": r2, "grade_used": grade_used,
           "mask": mask, "v": vm, "y": y, "drs": drs_open}
    out["physical"] = bool(0.6 < out["CdA"] < 2.0 and 400 < out["power_kW"] < 900)
    return out


def drag_fit_auto(g_lap, cap_kmh=290.0, tolerance=0.12, **kw):
    """drag_fit with a check for battery clipping.

    Fits twice: all speeds, and only below cap_kmh. If the hybrid battery
    stops deploying near the end of straights, power drops at top speed and
    the all-speed fit reads it as extra drag, so its CdA comes out higher.
    If the all-speed CdA exceeds the capped one by more than `tolerance`
    (and more than the combined fit error), the capped fit is used.
    Returns (fit, note).
    """
    full = drag_fit(g_lap, **kw)
    capped = drag_fit(g_lap, v_max_kmh=cap_kmh, **kw)
    if full is None:
        return capped, "all-speed fit had too few samples; used the capped fit"
    if capped is None:
        return full, "not enough samples below the cap to check for clipping"
    gap = full["CdA"] - capped["CdA"]
    if gap > tolerance * capped["CdA"] and gap > np.hypot(full["CdA_err"], capped["CdA_err"]):
        return capped, (f"possible battery clipping: all-speed CdA {full['CdA']:.2f} vs "
                        f"{capped['CdA']:.2f} below {cap_kmh:.0f} km/h; used the capped fit")
    return full, (f"no clipping detected (CdA {full['CdA']:.2f} all speeds vs "
                  f"{capped['CdA']:.2f} below {cap_kmh:.0f} km/h)")


def aero_summary(down, drag):
    """Combine downforce_fit and drag_fit into L/D = ClA / CdA (DRS closed).
    Mass and air density cancel in the ratio, so it is the most robust number."""
    if down is None or drag is None or drag["CdA"] <= 0:
        return None
    ld = down["ClA"] / drag["CdA"]
    rel = np.hypot(down["ClA_err"] / down["ClA"], drag["CdA_err"] / drag["CdA"])
    return {"L_over_D": ld, "L_over_D_err": abs(ld) * rel}


# ---------------------------------------------------------------------------
# 7. Whole-grid aero map
# ---------------------------------------------------------------------------

def grid_aero_map(session, fast_corner_kmh=180.0, match_m=60.0):
    """Speed trap vs mean minimum speed in the fast corners, per driver.

    Fast corners are found on the fastest lap (apex >= fast_corner_kmh; if fewer
    than two qualify, the three fastest apexes are used). Each driver's
    minimum speed within +/- match_m of those apexes is averaged.
    """
    laps = session.laps
    fastest = []
    for drv in laps["Driver"].dropna().unique():
        lap = laps.pick_drivers(drv).pick_fastest()
        if lap is not None and pd.notna(lap["LapTime"]):
            fastest.append(lap)
    if not fastest:
        return None, None
    fastest.sort(key=lambda lp: lp["LapTime"])

    ref = uniform_lap(fastest[0].get_telemetry())
    ref["SpeedFiltered"] = ref["Speed"]
    peaks, _ = find_peaks(-ref["Speed"].to_numpy(), prominence=10, distance=15)
    apexes = ref.iloc[peaks][["Distance", "Speed"]]
    fast = apexes[apexes["Speed"] >= fast_corner_kmh]
    if len(fast) < 2:
        fast = apexes.nlargest(3, "Speed")

    rows = []
    for lap in fastest:
        tel = lap.get_telemetry()
        d = pd.to_numeric(tel["Distance"], errors="coerce").to_numpy(dtype=float)
        s = pd.to_numeric(tel["Speed"], errors="coerce").to_numpy(dtype=float)
        mins = []
        for a in fast["Distance"]:
            seg = s[(d > a - match_m) & (d < a + match_m)]
            seg = seg[np.isfinite(seg)]
            if len(seg):
                mins.append(seg.min())
        rows.append({
            "Driver": lap["Driver"], "Team": lap["Team"],
            "LapTime_s": lap["LapTime"].total_seconds(),
            "SpeedTrap_kmh": float(lap["SpeedST"]) if pd.notna(lap["SpeedST"]) else float(np.nanmax(s)),
            "FastCornerSpeed_kmh": float(np.mean(mins)) if mins else np.nan,
            "Compound": lap["Compound"],
        })
    return pd.DataFrame(rows), fast
