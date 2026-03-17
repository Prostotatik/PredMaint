import pandas as pd
import numpy as np
from datetime import date, timedelta


def calculate_rul(unit_number, current_cycle_idx, test_data, rul_data):
    max_cycles = len(test_data[test_data["unit"] == unit_number])
    rul_final = rul_data.get(unit_number, 0)
    return max(0, rul_final + (max_cycles - current_cycle_idx))


def get_health_status(rul):
    if rul > 50:
        return "Healthy", "#00c853"
    elif rul >= 10:
        return "Impaired", "#ffd600"
    else:
        return "Failing", "#ff1744"


def get_health_icon(status):
    return {"Healthy": "🟢", "Impaired": "🟡", "Failing": "🔴"}.get(status, "⚪")


def detect_change_points(unit_data, sensors=None):
    if sensors is None:
        sensors = ["s_3", "s_7", "s_8", "s_14"]
    available = [s for s in sensors if s in unit_data.columns]
    if not available or len(unit_data) < 15:
        return []

    change_points = []
    baseline = unit_data[available].iloc[:8]
    bl_mean = baseline.mean()
    bl_std = baseline.std().replace(0, 1e-6)

    rolling = unit_data[available].rolling(window=5, min_periods=1).mean()
    z_scores = ((rolling - bl_mean) / bl_std).abs().mean(axis=1)

    for i in range(8, len(z_scores)):
        if z_scores.iloc[i] > 2.0 and (
            not change_points or i - change_points[-1]["idx"] > 5
        ):
            change_points.append(
                {
                    "idx": i,
                    "cycle": int(unit_data.iloc[i]["cycle"]),
                    "severity": min(1.0, z_scores.iloc[i] / 5.0),
                    "z_score": float(z_scores.iloc[i]),
                }
            )

    return change_points


def calculate_feature_importance(unit_data):
    sensor_cols = [f"s_{i}" for i in range(1, 22)]
    importances = {}

    if len(unit_data) < 5:
        return {c: 0.0 for c in sensor_cols}

    cycle = np.arange(len(unit_data), dtype=float)
    for col in sensor_cols:
        vals = unit_data[col].values.astype(float)
        std = np.std(vals)
        if std > 1e-8:
            corr = abs(np.corrcoef(cycle, vals)[0, 1])
            importances[col] = 0.0 if np.isnan(corr) else corr
        else:
            importances[col] = 0.0

    total = sum(importances.values())
    if total > 0:
        importances = {k: v / total for k, v in importances.items()}

    return importances


def get_top_contributing_sensors(unit_data, n=3):
    importances = calculate_feature_importance(unit_data)
    return sorted(importances.items(), key=lambda x: x[1], reverse=True)[:n]


def generate_maintenance_log(rul, machine_name, current_cycle, max_cycle):
    today = date.today()
    logs = []

    def _date(cycle):
        days_ago = max(0, max_cycle - cycle)
        return (today - timedelta(days=days_ago)).strftime("%Y-%m-%d")

    if rul < 5:
        logs.append(
            {
                "Date": _date(current_cycle),
                "Action": "EMERGENCY SHUTDOWN",
                "Reason": f"RUL critically low ({rul} cycles) — immediate replacement required",
                "Priority": "🔴 Critical",
            }
        )
    if rul < 15:
        logs.append(
            {
                "Date": _date(max(1, current_cycle - 2)),
                "Action": "Emergency Inspection Ordered",
                "Reason": "RUL dropped below 15 cycles — urgent maintenance required",
                "Priority": "🔴 High",
            }
        )
    if rul < 30:
        logs.append(
            {
                "Date": _date(max(1, current_cycle - 6)),
                "Action": "Vibration Analysis Scheduled",
                "Reason": "Anomalous sensor patterns detected — degradation accelerating",
                "Priority": "🟡 Medium",
            }
        )
    if rul < 50:
        logs.append(
            {
                "Date": _date(max(1, current_cycle - 12)),
                "Action": "Preventive Inspection",
                "Reason": "RUL approaching maintenance threshold — schedule intervention",
                "Priority": "🟡 Medium",
            }
        )

    logs.append(
        {
            "Date": _date(1),
            "Action": "System Commissioning",
            "Reason": f"{machine_name} registered — baseline established",
            "Priority": "🟢 Info",
        }
    )

    return logs
