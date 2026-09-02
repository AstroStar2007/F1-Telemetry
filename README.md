# F1 Telemetry Analysis Tool

Compares two F1 drivers' fastest laps using FastF1's 10Hz telemetry API: speed, throttle, and brake traces aligned by track distance, sector time deltas, and corner-by-corner minimum-speed comparison.


## What it does
- Pulls telemetry for any two drivers in any session (2018–present) via the [FastF1](https://docs.fastf1.dev/) API
- Aligns both traces by track distance (not time) so speed/throttle/brake compare apples-to-apples at the same physical point on track
- Detects corner apexes via local speed minima and reports the biggest min-speed deltas between drivers
- Computes sector-by-sector time deltas from each driver's fastest lap
- **Signal filtering:** applies a 3rd-order zero-phase Butterworth low-pass filter to raw ~10Hz speed/position data before differentiating, to avoid amplifying sensor noise into the acceleration estimate — configurable cutoff frequency in the UI
- **G-force estimation:** derives longitudinal G (from filtered speed vs time) and lateral G (from track curvature computed off X/Y position and speed) at every point on the lap, with peak braking/cornering G reported per driver
- **DRS effect analysis:** compares average speed with the rear-wing drag reduction system active vs inactive, and DRS-active percentage of the lap, per driver

Filtering + differentiation-based load estimation is the same workflow used for post-processing IMU/flight-test data, which made this a natural extension given an aerospace background.

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Tech
Python, FastF1, Pandas, NumPy, SciPy (Butterworth filtering), Matplotlib, Streamlit

## Example finding
!(example1.png)
!(example2.png)
