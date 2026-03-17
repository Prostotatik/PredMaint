Tech stack (strictly follow this, no deviations):
Visualization & Dashboard: Streamlit + Plotly + Pandas
ML Framework: PyTorch (LSTM for RUL regression) + Scikit-Learn (feature engineering + RandomForest fallback)
Time-series toolkit: sktime (for change-point detection) + Darts (if needed for additional forecasting)
Database: SQLite (via streamlit-sqlite or st.connection)
Data: NASA CMAPSS (FD001 train and test sets). You MUST include code that automatically downloads the dataset from the official NASA link or uses the standard 4-column format + 21 sensors.
Theme: dark industrial (blacks, deep blues, orange accents for alerts)
Core Requirements (the app MUST implement ALL of them exactly as written):
The application is a professional factory predictive maintenance dashboard with two levels:
Global Overview (main page)
Individual Machine Drill-down (click on any machine)
Machine Management (sidebar + dedicated page):

Form: “Add New Machine” → fields: Machine ID (unique string), Machine Name (e.g. “Motor-Assembly-Line-3”)
List of all machines in a table with “Delete” button
Each machine is automatically assigned a unique unit number from CMAPSS dataset (random from 1 to 100)
All machine data is persisted in SQLite table “machines” and “machine_data” (cycle-by-cycle sensor history)

Global Overview Dashboard (default page):
Must contain exactly these components:
Table 1 – Machines Overview
Columns:

Machine ID
Machine Name
Current RUL (cycles) — large number with color (green >50, yellow 10-50, red <10)
Health Status (Healthy / Impaired / Failing)
Last Cycle
Time-to-Failure (countdown)
“View Details” button

Graph 1 – Overall RUL Distribution

Plotly Bar chart
X = Machine Name, Y = RUL
Bars colored by Health Status
Hover shows exact RUL and top 3 contributing sensors

Graph 2 – Fleet Health Pie Chart

Slices: Healthy / Impaired / Failing
Percentages calculated from change-point detection + RUL threshold

Graph 3 – Real-time Fleet Sensor Trends (multi-line)

4 lines simultaneously:
s_3 (HPC outlet temperature °R)
s_7 (HPC outlet pressure psia)
s_8 (Physical fan speed rpm)
s_14 (Corrected core speed rpm)
X = last 200 cycles (simulated time)
Aggregated across ALL machines (mean + std bands)
Auto-updates when “Simulate Next Cycle” is pressed

Graph 4 – Recent Anomalies Timeline

Scatter plot: X = Cycle, Y = Machine Name
Points appear only when change-point is detected (sktime Pelt algorithm)
Size and color indicate severity

Individual Machine View (opened by clicking any machine):
Must contain exactly these components:
Header

Machine ID + Name
Huge “RUL: XXX cycles” card with countdown animation
Health Status badge + “Time-to-Failure” countdown

Section 1 – Real-time Multi-Sensor Dashboard

Interactive Plotly line chart (up to 4 sensors at once, user can choose)
Default sensors shown: s_3, s_4, s_7, s_11, s_13, s_14
All 21 sensors available in dropdown
Includes rolling mean (window=10) to handle noise
Vertical dashed line showing detected change-point (healthy → impaired)

Section 2 – RUL Prediction Trend

Line chart: X = Cycle, Y = Predicted RUL
Confidence interval band (± MAE)
Model re-infers every new cycle

Section 3 – Degradation Feature Importance

Plotly bar chart (SHAP-style or permutation importance)
Top 8 features that most affect RUL drop for this machine at current cycle

Table 2 – Latest 20 Sensor Readings
Columns: Cycle | s_3 | s_4 | s_7 | s_8 | s_11 | s_12 | s_13 | s_14 | s_21 | Predicted RUL
Table 3 – Maintenance Log (simulated)
Columns: Date | Action | Reason (e.g. “RUL dropped below 30 cycles – recommend inspection”)
Real-time Simulation Engine (must work without real sensors, must use test_FD00x files):

Big button “Simulate Next Cycle for All Machines” (or per machine)
Every click:
Takes next row from the assigned CMAPSS unit’s test data
Appends to machine’s history
Re-runs feature engineering
Re-runs LSTM RUL prediction
Re-runs change-point detection
Updates ALL graphs and tables instantly via st.rerun()


Session state + SQLite persistence
Model Requirements (backend):

Temporal Feature Engineering: sliding window (size 30) → mean, std, max, min, slope for every sensor + op settings 1-3
RUL Model: PyTorch LSTM (2 layers, 128 hidden) trained on CMAPSS training set (RUL = max_cycle - current_cycle)
Fallback: Scikit-Learn RandomForestRegressor with engineered features
Metrics shown on load: RMSE < 18, MAE < 12 on test set (must achieve this)
Anomaly Change-Point Detection: sktime Pelt algorithm on multi-sensor stream
Noisy data handling: median imputation + z-score clipping
Interpretability: built-in feature importance for every prediction

Additional Professional Features:

Dark industrial theme with custom CSS (factory feel)
Responsive on desktop and tablet
Loading spinners during model inference
Alert system: if any machine RUL < 15 → red banner on top
Export buttons: CSV of current sensor data + RUL report
“Reset Demo Data” button (reloads fresh CMAPSS test data)

The resulting app must look like a real enterprise dashboard that a factory manager in ASEAN would proudly show to investors. No placeholders, no “TODO”, everything fully functional with the NASA CMAPSS data.

hackathon question
1. Real-World Context
Small and Medium Enterprises (SMEs) are the backbone of the ASEAN
economy, yet they often operate with aging industrial machinery and thin
profit margins. Unlike large conglomerates, these factories cannot
afford "Smart Factory" overhauls. A single motor failure in a rural food
processing plant can halt production for weeks, leading to massive
financial losses and resource waste.
Predictive Maintenance for SME
Resilience
CASE STUDY 1:
Track: Machine Learning (Time-Series / Remaining Useful Life Estimation)
Primary Goal: SDG 9: Industry, Innovation, and Infrastructure
 (Target 9.4)
2. Problem Statement
Current maintenance in ASEAN SMEs is largely reactive (fixing after
failure) or preventative (replacing parts too early). Both are inefficient.
There is a critical need for an AI-driven system that can analyze sensor
data,such as vibration, temperature, and load to predict the Remaining
Useful Life (RUL) of machinery, enabling proactive planning and reducing
downtime.
3. Technical Challenge & Sub-tasks
• Temporal Feature Engineering: Process high-frequency multivariate
time-series data to extract features that represent machine
degradation.
• RUL Regression Modeling: Develop a robust model to estimate the
precise number of cycles or hours remaining before a functional
failure.
• Anomaly Change-Point Detection: Implement logic to identify the
exact moment a machine transitions from a "Healthy" state to an
"Impaired" state.
• Health Dashboard Visualization: Design a user-friendly interface for
factory managers to visualize machine health and schedule
maintenance efficiently.
4. Technical Feasibility & Constraints
• Noisy Data Handling: Models must demonstrate the ability to handle
sensor noise and missing data points common in real-world industrial
settings.
• Model Interpretability: The AI must provide actionable insights (e.g.,
why the RUL is decreasing) to gain trust from non-technical factory
operators.
• Scalability: The solution should be modular, allowing it to be applied
to different types of machinery with minimal retraining.
5. Recommended Data Sources & Toolkits
• Suggested Datasets: NASA CMAPSS.
• Suggested Frameworks: Scikit-Learn, TensorFlow/PyTorch, Darts,
sktime, NeuralProphet, or Prophet.
• Visualization Tools: Streamlit or Dash.
6. Expected Deliverables
• Predictive Model: A validated model with clear performance metrics
(e.g., RMSE or MAE).
• Functional Dashboard: A prototype showing real-time machine
status and "Time-to-Failure" countdowns.