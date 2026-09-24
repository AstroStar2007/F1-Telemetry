"""
aero_compare.py - downforce vs drag across circuits: the aero map.

    python aero_compare.py --year 2024 --events Monza Silverstone Hungary --top 5

For each event it takes the qualifying laps of the fastest --top drivers,
estimates downforce area (ClA, from cornering) and drag area (CdA, from
full-throttle straights) for every lap, and plots ClA against CdA with lines
of constant L/D.

Teams bring low-downforce wings to Monza and high-downforce wings to
Hungary, so a working method should put Monza bottom-left (low drag, low
downforce) and Hungary top-right. That is a check against something known,
not just a number.

Outputs in figures/ (next to this file): aero_map.png and aero_compare.csv
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

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]   # colour-blind-checked, max 3 events
GREY, INK, MUTED, GRID = "#52514e", "#0b0b0b", "#8a8985", "#e6e5e1"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "axes.edgecolor": MUTED, "xtick.color": GREY, "ytick.color": GREY,
    "legend.frameon": False,
})


def air_density(session):
    try:
        w = session.weather_data
        rho = float(w["Pressure"].median()) * 100 / (287.05 * (float(w["AirTemp"].median()) + 273.15))
        if 1.0 < rho < 1.35:
            return rho
    except Exception:
        pass
    return 1.20


def analyse_lap(lap, rho, mass):
    g = A.g_forces(lap.get_telemetry())
    down = A.downforce_fit(A.corner_table(g), mass_kg=mass, rho=rho)
    drag, note = A.drag_fit_auto(g, mass_kg=mass, rho=rho)
    summ = A.aero_summary(down, drag)
    if down is None or drag is None or summ is None:
        return None
    return {
        "ClA": down["ClA"], "ClA_err": down["ClA_err"], "mu": down["mu"],
        "CdA": drag["CdA"], "CdA_err": drag["CdA_err"], "power_kW": drag["power_kW"],
        "dCdA_DRS": drag["dCdA_DRS"], "L_over_D": summ["L_over_D"],
        "L_over_D_err": summ["L_over_D_err"], "drag_note": note,
        "ok": bool(down["physical"] and drag["physical"]),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--events", nargs="+", default=["Monza", "Silverstone", "Hungary"])
    ap.add_argument("--session", default="Q")
    ap.add_argument("--top", type=int, default=5, help="fastest N drivers per event")
    ap.add_argument("--mass", type=float, default=800.0)
    ap.add_argument("--out", default=os.path.join(HERE, "figures"), help="output folder")
    args = ap.parse_args(argv)
    if len(args.events) > 3:
        raise SystemExit("Use at most 3 events so the map stays readable.")

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for ev in args.events:
        print(f"Loading {args.year} {ev} {args.session} ...")
        session = fastf1.get_session(args.year, ev, args.session)
        session.load(laps=True, telemetry=True, weather=True)
        rho = air_density(session)
        laps = []
        for drv in session.laps["Driver"].dropna().unique():
            lap = session.laps.pick_drivers(drv).pick_fastest()
            if lap is not None and pd.notna(lap["LapTime"]):
                laps.append(lap)
        laps.sort(key=lambda lp: lp["LapTime"])
        for lap in laps[:args.top]:
            if str(lap["Compound"]).upper() in ("INTERMEDIATE", "WET"):
                print(f"  {lap['Driver']}: wet tyres, skipped")
                continue
            res = analyse_lap(lap, rho, args.mass)
            if res is None:
                print(f"  {lap['Driver']}: not enough corners or straights, skipped")
                continue
            res.update({"event": ev, "driver": lap["Driver"], "team": lap["Team"],
                        "lap_s": lap["LapTime"].total_seconds(), "rho": rho})
            rows.append(res)
            flag = "" if res["ok"] else "   <- outside sensible range, excluded"
            print(f"  {lap['Driver']:>3}: ClA {res['ClA']:.2f}  CdA {res['CdA']:.2f}  "
                  f"L/D {res['L_over_D']:.2f}  P {res['power_kW']:.0f} kW  [{res['drag_note']}]{flag}")

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No usable laps.")
    df.to_csv(os.path.join(out_dir, "aero_compare.csv"), index=False)
    good = df[df["ok"]]
    if good.empty:
        raise SystemExit("Every lap fell outside the sensible range - don't publish; check the data.")

    summary = good.groupby("event", sort=False)[["ClA", "CdA", "L_over_D", "power_kW"]].median()
    print("\nMedian per event (laps inside the sensible range only):")
    print(summary.round(2).to_string())

    # ---- The aero map -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    cmin, cmax = good["CdA"].min() * 0.85, good["CdA"].max() * 1.12
    lmin, lmax = good["ClA"].min() * 0.8, good["ClA"].max() * 1.15
    cc = np.linspace(cmin, cmax, 50)
    for ld in (2, 3, 4, 5, 6):
        yy = ld * cc
        if yy.max() < lmin or yy.min() > lmax:
            continue
        ax.plot(cc, yy, color=GRID, lw=1.1, zorder=0)
        x_lab = min(cmax, lmax / ld) * 0.985           # where the line leaves the plot
        ax.text(x_lab, ld * x_lab, f"L/D {ld:g}", fontsize=7.5, color=MUTED,
                ha="right", va="bottom", rotation=0)
    for colour, ev in zip(SERIES, summary.index):
        sub = good[good["event"] == ev]
        ax.errorbar(sub["CdA"], sub["ClA"], xerr=sub["CdA_err"], yerr=sub["ClA_err"],
                    fmt="o", ms=4, color=colour, alpha=0.45, elinewidth=0.8, capsize=0)
        m = summary.loc[ev]
        ax.plot(m["CdA"], m["ClA"], "o", ms=11, color=colour, mec="white", mew=1.5,
                label=f"{ev}: L/D {m['L_over_D']:.1f}")
    ax.set_xlim(cmin, cmax)
    ax.set_ylim(lmin, lmax)
    ax.set_xlabel("Drag area C$_D$A (m²)")
    ax.set_ylabel("Downforce area C$_L$A (m²)")
    ax.set_title(f"{args.year} qualifying: downforce vs drag (top {args.top} laps per event)")
    ax.legend(loc="upper left", title="Median per event", fontsize=9)
    fig.savefig(os.path.join(out_dir, "aero_map.png"))
    plt.close(fig)
    print(f"\nSaved aero_map.png and aero_compare.csv in {out_dir}")


if __name__ == "__main__":
    main()
