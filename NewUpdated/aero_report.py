"""
aero_report.py - portfolio figures from one qualifying session.

    python aero_report.py --year 2024 --gp Silverstone --session Q --driver NOR

Makes, in figures/ (next to this file):
  aero_grip_vs_speed.png   lateral g vs apex speed + fitted grip/downforce model
  aero_lateral_g.png       lateral g around the lap, corners labelled
  aero_grid_map.png        speed trap vs fast-corner speed, whole grid
  aero_drag_fit.png        power balance on straights vs speed: drag and DRS
and prints the numbers (also saved to figures/aero_results.csv).

FastF1 downloads each session once and keeps it in its own cache folder
(outside this project), so later runs are fast.
"""

import argparse
import os

import fastf1
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import aero_analysis as A

HERE = os.path.dirname(os.path.abspath(__file__))

BLUE, ORANGE, GREY, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#52514e", "#0b0b0b", "#8a8985", "#e6e5e1"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "axes.edgecolor": MUTED, "xtick.color": GREY, "ytick.color": GREY,
    "legend.frameon": False, "lines.linewidth": 1.8,
})


def air_density(session):
    """rho = p / (R T) from the session's median air temperature and pressure."""
    try:
        w = session.weather_data
        p_pa = float(w["Pressure"].median()) * 100.0          # mbar -> Pa
        t_k = float(w["AirTemp"].median()) + 273.15
        rho = p_pa / (287.05 * t_k)
        if 1.0 < rho < 1.35:
            return rho, float(w["AirTemp"].median()), float(w["Pressure"].median())
    except Exception:
        pass
    return 1.20, np.nan, np.nan


def label_corners(session, distances):
    """Turn numbers for apex distances (nearest official corner within 150 m)."""
    try:
        ci = session.get_circuit_info().corners
        out = []
        for d in distances:
            i = (ci["Distance"] - d).abs().idxmin()
            if abs(ci.loc[i, "Distance"] - d) < 150:
                letter = ci.loc[i, "Letter"]
                letter = letter if isinstance(letter, str) else ""
                out.append(f"T{int(ci.loc[i, 'Number'])}{letter}")
            else:
                out.append("")
        return out
    except Exception:
        return [""] * len(distances)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--gp", default="Silverstone")
    ap.add_argument("--session", default="Q")
    ap.add_argument("--driver", default=None, help="3-letter code; default = fastest lap of the session")
    ap.add_argument("--mass", type=float, default=800.0, help="car + driver, kg")
    ap.add_argument("--out", default=os.path.join(HERE, "figures"), help="output folder")
    args = ap.parse_args(argv)

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    session = fastf1.get_session(args.year, args.gp, args.session)
    session.load(laps=True, telemetry=True, weather=True)
    event = f"{args.year} {session.event['EventName']} {args.session}"

    laps = session.laps
    lap = (laps.pick_drivers(args.driver.upper()) if args.driver else laps).pick_fastest()
    if lap is None or pd.isna(lap["LapTime"]):
        raise SystemExit("No valid fastest lap found for that driver/session.")
    drv = lap["Driver"]
    if str(lap["Compound"]).upper() in ("INTERMEDIATE", "WET"):
        print("WARNING: this lap was on wet/intermediate tyres - grip numbers won't be comparable.")

    rho, t_air, p_air = air_density(session)
    g = A.g_forces(lap.get_telemetry(), cutoff_hz=2.0, pos_cutoff_hz=1.0)
    corners = A.corner_table(g)
    corners.insert(0, "Turn", label_corners(session, corners["Distance_m"]))
    fit = A.downforce_fit(corners, mass_kg=args.mass, rho=rho)

    print(f"\n{event} - {drv} fastest lap {lap['LapTime'].total_seconds():.3f} s")
    print(f"Air density {rho:.3f} kg/m^3 (air {t_air:.1f} C, {p_air:.0f} mbar)")
    print(corners.to_string(index=False))
    if fit is None:
        raise SystemExit("Fewer than 4 grip-limited corners found - try another circuit.")
    print(f"\nTyre friction  mu  = {fit['mu']:.2f} +/- {fit['mu_err']:.2f}")
    print(f"Downforce area ClA = {fit['ClA']:.2f} +/- {fit['ClA_err']:.2f} m^2")
    print(f"Downforce / weight at 250 km/h = {fit['L_over_W_250']:.2f}")
    print(f"Fit R^2 = {fit['r2']:.3f} over {fit['n']} corners")
    if not fit["physical"]:
        print("WARNING: result outside the physically sensible range "
              "(mu 1-2.5, ClA 2-7 m^2). Don't publish it - try another session.")

    drag, drag_note = A.drag_fit_auto(g, mass_kg=args.mass, rho=rho)
    summ = A.aero_summary(fit, drag)
    if drag is None:
        print("\nDrag fit skipped: not enough full-throttle straight-line data.")
    else:
        print(f"\nDrag (full-throttle straights, {drag['n']} samples; {drag_note})")
        print(f"Wheel power    P   = {drag['power_kW']:.0f} +/- {drag['power_err']:.0f} kW")
        print(f"Drag area      CdA = {drag['CdA']:.2f} +/- {drag['CdA_err']:.2f} m^2 (DRS closed)")
        if np.isfinite(drag["dCdA_DRS"]):
            pct = 100 * drag["dCdA_DRS"] / drag["CdA"]
            print(f"DRS drag saving    = {drag['dCdA_DRS']:.2f} +/- {drag['dCdA_err']:.2f} m^2 ({pct:.0f}% of CdA)")
        print(f"Fit R^2 = {drag['r2']:.3f}; grade correction {'on' if drag['grade_used'] else 'off'}")
        if summ:
            print(f"\nAERO EFFICIENCY  L/D = ClA / CdA = {summ['L_over_D']:.2f} +/- {summ['L_over_D_err']:.2f}")
        if not drag["physical"]:
            print("WARNING: drag result outside the sensible range (CdA 0.6-2.0 m^2, "
                  "power 400-900 kW). Don't publish it.")

    # ---- Figure 1: grip vs speed ------------------------------------------
    used = fit["used"]
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    vv = np.linspace(0, max(used["ApexSpeed_kmh"].max() * 1.08, 300), 200)
    ax.plot(vv, fit["mu"] + fit["slope"] * (vv / 3.6) ** 2, color=BLUE,
            label=f"Fit: μ = {fit['mu']:.2f}, ClA = {fit['ClA']:.1f} ± {fit['ClA_err']:.1f} m²")
    ax.axhline(fit["mu"], color=GREY, ls="--", lw=1.2, label="Grip without downforce (μ)")
    ax.plot(used["ApexSpeed_kmh"], used["LatG"], "o", color=ORANGE, ms=7,
            label=f"{drv} corner apexes")
    for _, r in used.iterrows():
        if r["Turn"]:
            ax.annotate(r["Turn"], (r["ApexSpeed_kmh"], r["LatG"]), xytext=(5, 4),
                        textcoords="offset points", fontsize=8, color=INK)
    ax.set_xlim(0, vv.max())
    ax.set_ylim(0, None)
    ax.set_xlabel("Apex speed (km/h)")
    ax.set_ylabel("Lateral acceleration (g)")
    ax.set_title(f"{event}: grip rises with speed² = downforce")
    ax.legend(loc="upper left", fontsize=8.5)
    fig.savefig(os.path.join(out_dir, "aero_grip_vs_speed.png"))
    plt.close(fig)

    # ---- Figure 2: lateral g around the lap --------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    ax.plot(g["Distance"], g["LatG"], color=BLUE, lw=1.4, label=f"{drv} lateral g")
    ax.plot(corners["Distance_m"], corners["LatG"], "o", color=ORANGE, ms=6,
            label="Apex (circle-fit radius)")
    for _, r in corners.iterrows():
        if r["Turn"]:
            ax.annotate(r["Turn"], (r["Distance_m"], r["LatG"]), xytext=(0, 6),
                        textcoords="offset points", ha="center", fontsize=8, color=INK)
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Lateral acceleration (g)")
    ax.set_title(f"{event}: lateral g from filtered position data")
    ax.legend(loc="upper right", fontsize=8.5)
    fig.savefig(os.path.join(out_dir, "aero_lateral_g.png"))
    plt.close(fig)

    # ---- Figure 3: whole-grid aero map --------------------------------------
    grid, fast = A.grid_aero_map(session)
    if grid is not None and grid["FastCornerSpeed_kmh"].notna().sum() >= 5:
        grid = grid.dropna(subset=["FastCornerSpeed_kmh", "SpeedTrap_kmh"])
        fig, ax = plt.subplots(figsize=(6.8, 4.6))
        ax.plot(grid["FastCornerSpeed_kmh"], grid["SpeedTrap_kmh"], "o", color=BLUE, ms=7)
        for _, r in grid.iterrows():
            ax.annotate(r["Driver"], (r["FastCornerSpeed_kmh"], r["SpeedTrap_kmh"]),
                        xytext=(4, 3), textcoords="offset points", fontsize=8, color=INK)
        ax.margins(x=0.12, y=0.12)
        mx, my = grid["FastCornerSpeed_kmh"].median(), grid["SpeedTrap_kmh"].median()
        ax.axvline(mx, color=MUTED, ls=":", lw=1)
        ax.axhline(my, color=MUTED, ls=":", lw=1)
        xl, yl = ax.get_xlim(), ax.get_ylim()
        kw = dict(fontsize=8, color=GREY, style="italic")
        ax.text(xl[0], yl[1], " Low drag", va="top", **kw)
        ax.text(xl[1], yl[1], "Fast everywhere ", va="top", ha="right", **kw)
        ax.text(xl[1], yl[0], "High downforce ", va="bottom", ha="right", **kw)
        ax.set_xlabel(f"Mean minimum speed, {len(fast)} fastest corners (km/h)")
        ax.set_ylabel("Speed trap (km/h)")
        ax.set_title(f"{event}: straight-line vs high-speed-corner speed")
        fig.savefig(os.path.join(out_dir, "aero_grid_map.png"))
        plt.close(fig)
        grid.to_csv(os.path.join(out_dir, "aero_grid_map.csv"), index=False)
    else:
        print("Grid map skipped: not enough drivers with valid laps.")

    # ---- Figure 4: power balance on the straights (drag) --------------------
    if drag is not None:
        v_kmh = drag["v"] * 3.6
        p_kw = drag["y"] / 1e3
        drs = drag["drs"] if drag["drs"] is not None else np.zeros(len(v_kmh), bool)
        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        ax.plot(v_kmh[~drs], p_kw[~drs], "o", ms=3, color=BLUE, alpha=0.5, label="DRS closed")
        if drs.any():
            ax.plot(v_kmh[drs], p_kw[drs], "o", ms=3, color=ORANGE, alpha=0.5, label="DRS open")
        vv = np.linspace(v_kmh[~drs].min(), v_kmh[~drs].max(), 100) / 3.6   # closed range only
        ax.plot(vv * 3.6, (drag["power_kW"] * 1e3 - 0.5 * rho * drag["CdA"] * vv**3) / 1e3,
                color=BLUE, lw=2, label=f"Fit: P = {drag['power_kW']:.0f} kW, CdA = {drag['CdA']:.2f} m²")
        if np.isfinite(drag["dCdA_DRS"]) and drs.any():
            cd_open = drag["CdA"] - drag["dCdA_DRS"]
            vo = np.linspace(v_kmh[drs].min(), v_kmh[drs].max(), 100) / 3.6
            ax.plot(vo * 3.6, (drag["power_kW"] * 1e3 - 0.5 * rho * cd_open * vo**3) / 1e3,
                    color=ORANGE, lw=2, label=f"DRS open: CdA = {cd_open:.2f} m²")
        lo, hi = np.percentile(p_kw, [2, 99.5])
        ax.set_ylim(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo))   # hide the odd noisy point
        ax.set_xlabel("Speed (km/h)")
        ax.set_ylabel("Power into acceleration + climbing (kW)")
        ax.set_title(f"{event}: full-throttle power balance (drag grows with v³)")
        ax.legend(loc="lower left", fontsize=8.5)
        fig.savefig(os.path.join(out_dir, "aero_drag_fit.png"))
        plt.close(fig)
        pd.DataFrame({"v_kmh": np.round(v_kmh, 2), "power_kW": np.round(p_kw, 2),
                      "drs_open": drs}).to_csv(os.path.join(out_dir, "aero_drag_samples.csv"), index=False)

    row = {
        "event": event, "driver": drv, "rho": round(rho, 4), "mass_kg": args.mass,
        "mu": round(fit["mu"], 3), "mu_err": round(fit["mu_err"], 3),
        "ClA_m2": round(fit["ClA"], 3), "ClA_err": round(fit["ClA_err"], 3),
        "L_over_W_250kmh": round(fit["L_over_W_250"], 3), "r2": round(fit["r2"], 3),
        "corners_used": fit["n"]}
    if drag is not None:
        row.update({"CdA_m2": round(drag["CdA"], 3), "CdA_err": round(drag["CdA_err"], 3),
                    "power_kW": round(drag["power_kW"], 1),
                    "dCdA_DRS": round(drag["dCdA_DRS"], 3) if np.isfinite(drag["dCdA_DRS"]) else "",
                    "drag_note": drag_note})
    if summ:
        row.update({"L_over_D": round(summ["L_over_D"], 3), "L_over_D_err": round(summ["L_over_D_err"], 3)})
    pd.DataFrame([row]).to_csv(os.path.join(out_dir, "aero_results.csv"), index=False)
    corners.to_csv(os.path.join(out_dir, "aero_corners.csv"), index=False)
    print(f"\nFigures and CSVs saved in {out_dir}")


if __name__ == "__main__":
    main()
