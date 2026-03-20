# Predictive Maintenance Dashboard (NASA CMAPSS)

A production-style predictive maintenance demo for SMEs: it estimates **Remaining Useful Life (RUL)** from multivariate time-series, detects the **Healthy → Impaired** transition, and visualizes fleet health with a real-time, cycle-by-cycle simulation.

## What it does

The app is an interactive, dark-industrial **Streamlit dashboard** with two levels:

- **Global Overview (main page)**: a fleet table with RUL + health labels, plus charts for RUL distribution, fleet health, real-time sensor trends, and recent anomaly detections.
- **Individual Machine Drill-down (details page)**: a multi-sensor dashboard (with rolling mean and change-point markers), an LSTM-based RUL trend with an uncertainty band, sensor contribution (feature importance) scores, latest readings, and a simulated maintenance log.

It also includes:

- **Real-time simulation**: click `Simulate Next Cycle (All)` (or enable `Auto-Simulate`) to advance machines and automatically refresh predictions, charts, and logs.
- **Exports**: download fleet summary (CSV) and per-machine sensor data + RUL report.
- **AI assistant (Gemini)**: a floating chat widget that answers in English and references the live dashboard state (health status, RUL values, and where to inspect).

## Model & signals

- **RUL regression**: a PyTorch **LSTM** model trained on NASA CMAPSS sequences (sequence length `30`) using operational settings + sensor measurements.
- **Data robustness**: median imputation + z-score clipping for noisy signals.
- **Consistency**: monotonicity constraints during training and a monotonic post-processing step at inference time.
- **Calibration**: isotonic regression calibrated against the provided test-set RUL targets.
- **Health & change-point detection**: health is derived from predicted RUL thresholds, and the **change-point** is detected around the predicted **Healthy → Impaired** transition.
- **Interpretability**: a practical, dashboard-friendly sensor contribution ranking (correlation-based scoring).

## Data (NASA CMAPSS)

The dashboard uses the **NASA CMAPSS** dataset in the `dataset/` folder (FD00x).
Each test trajectory is loaded from standard CMAPSS text files in the repository:

- `train_FD001.txt` / `test_FD001.txt`
- `RUL_FD001.txt` (ground-truth RUL for calibration/evaluation)

Each row represents a cycle snapshot with:

- 3 operational settings (`op_setting_1..3`)
- 21 sensor measurements (`s_1..s_21`)

## 🧩 Hackathon Story: Predictive Maintenance for SME Resilience

SMEs cannot afford downtime, and they usually cannot afford “black-box” systems that only a data scientist can operate. So we built something the factory team can actually *drive*: a live dashboard that estimates **Remaining Useful Life (RUL)**, detects the **Healthy → Impaired** transition, and converts that into an actionable maintenance narrative. 🚀🏭

### 🎯 Hackathon Question → Our Answer

| Hackathon prompt | What you can see in the app |
|---|---|
| SMEs run aging equipment with thin margins | “Fleet Overview” with instant health labels + “Time-to-Failure” countdowns for every registered unit |
| Reactive vs. preventative maintenance | RUL-driven health thresholds: **Healthy / Impaired / Failing** (with a top alert banner when RUL < 15) |
| Need AI that analyzes sensor data (vibration/temp/load-like) | LSTM-based RUL regression trained on NASA CMAPSS signals (op settings + `s_1..s_21`) |
| Identify the exact Healthy → Impaired moment | Change-point detector scans the predicted cycle-by-cycle health transition and timestamps the switch |
| Noisy/real-world sensor behavior | Median imputation + z-score clipping during feature preparation |
| Non-technical interpretability | Sensor contribution ranking (correlation-based) + “Latest Sensor Readings” table |
| Scalability and modularity | Add machines from the UI; the app assigns CMAPSS units (random unit #) and simulates cycle-by-cycle per unit |
| Functional demo | Fleet simulation buttons + real-time chart refreshes (`st.rerun()` + fragments) + CSV/TXT exports |

### 🏅 Battles Won (End-to-End, No Placeholders)

We didn’t stop at “training a model” — we shipped the full loop you need for a factory demo:

| 🧠 Capability | 💪 Proof inside the repo |
|---|---|
| Training + persistence | Click `🚀 Train AI Model`; artifacts are saved to `models/saved/` (`lstm_model.pth`, `scaler.pkl`, etc.) |
| Robust inference pipeline | Inference runs on the selected unit’s latest cycle window (sequence length `30`) with the same cleaning rules as training |
| Monotonic RUL behavior | During prediction, RUL is forced to be non-increasing via monotonic post-processing (cumulative min) |
| “Exact” health transition timing | `detect_change_points()` uses `rul_sequence` to find the first `Healthy → Impaired` switch, then estimates severity from normalized sensor divergence |
| Explainability you can show | “Degradation Feature Importance” chart based on sensor ranking |
| A maintenance log that reads like a plan | “Maintenance Log (simulated)” uses RUL thresholds to output practical actions and reasons |
| Live system AI assistant | Floating Gemini assistant that answers using the dashboard state snapshot |

### 📦 Expected Deliverables (What Judges Can Evaluate)

| Deliverable | What’s included |
|---|---|
| Predictive model with metrics | After training, the UI shows LSTM `RMSE` and `MAE` in the `🤖 AI Model` panel |
| Functional dashboard | Global Overview + Individual Machine Drill-down with real-time simulation and anomaly timeline |
| Demonstrable “time-to-failure” | Fleet table + machine page countdowns computed from predicted RUL |
| Real-world-friendly UX | Dark industrial styling, tables, charts, exports, and “Reset Demo Data” |

## 🧨 Why this is a Hackathon-Ready Demo

Because it feels like production: a factory manager can add machines, hit “Simulate Next Cycle”, see the transition moment, and export a report — without touching a notebook. We turned NASA CMAPSS text trajectories into a complete SME-ready predictive maintenance workflow. 👨‍🔧📈

## Run it

1. Install dependencies:
   - `pip install -r requirements.txt`
2. (Optional) Configure Gemini for the AI assistant:
   - set `GEMINI_API_KEY` in your environment (or update `.env`)
3. Start the dashboard:
   - `streamlit run app.py`
4. Train the AI model (if it's not saved locally yet):
   - Open the expander `🤖 AI Model` in the app sidebar.
   - The app checks for `models/saved/lstm_model.pth` and `models/saved/scaler.pkl`.
   - If you see `Model not trained yet`, click `🚀 Train AI Model`.
   - Training will run using the CMAPSS files from `dataset/` (default FD001) and save artifacts to `models/saved/`:
     - `lstm_model.pth`, `scaler.pkl`, and `metrics.json` (plus `calibrator.pkl` when available).


