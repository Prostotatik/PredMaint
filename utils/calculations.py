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


def detect_change_points(unit_data, sensors=None, rul_sequence=None):
    if sensors is None:
        sensors = ["s_3", "s_7", "s_8", "s_14"]
    available = [s for s in sensors if s in unit_data.columns]

    # We need a minimal context window to compare against the "baseline" segment
    if not available or len(unit_data) < 15:
        return []

    raw_signal = unit_data[available].values.astype(np.float64)
    baseline_len = min(8, max(2, len(raw_signal) // 3))
    bl_mean = raw_signal[:baseline_len].mean(axis=0)
    bl_std = raw_signal[:baseline_len].std(axis=0)
    bl_std = np.where(bl_std < 1e-6, 1e-6, bl_std)

    # Normalize sensors relative to an early baseline window
    z_signal = (raw_signal - bl_mean) / bl_std
    z_signal = np.clip(z_signal, -3.0, 3.0)

    def _severity_at(bp: int) -> float:
        before = z_signal[max(0, bp - 5) : bp]
        after = z_signal[bp : min(len(z_signal), bp + 5)]
        if len(before) > 0 and len(after) > 0:
            diff = np.abs(after.mean(axis=0) - before.mean(axis=0))
            base_std = np.std(z_signal, axis=0).mean()
            return min(1.0, float(np.mean(diff) / (base_std + 1e-6)))
        return 0.5

    # ------------------------------------------------------------------
    # "Exact moment Healthy -> Impaired"
    # If we have a predicted RUL series for each cycle (`rul_sequence`),
    # then we look for the health-mode switch exactly by the RUL thresholds.
    # ------------------------------------------------------------------
    if rul_sequence is not None:
        rul_arr = np.array(rul_sequence, dtype=float).reshape(-1)
        n = min(len(rul_arr), len(unit_data))
        if n >= 2:
            rul_arr = rul_arr[:n]
            health_labels = [get_health_status(float(r))[0] for r in rul_arr]

            bp = None
            for i in range(1, n):
                if health_labels[i - 1] == "Healthy" and health_labels[i] == "Impaired":
                    bp = i
                    break

            # If we didn't find the specific Healthy->Impaired transition (e.g., it jumped to Failing),
            # take the first change from Healthy to any non-Healthy state.
            if bp is None:
                for i in range(1, n):
                    if health_labels[i - 1] == "Healthy" and health_labels[i] != "Healthy":
                        bp = i
                        break

            if bp is not None:
                severity = _severity_at(int(bp))
                return sorted(
                    [
                        {
                            "idx": int(bp),
                            "cycle": int(
                                unit_data.iloc[min(int(bp), len(unit_data) - 1)][
                                    "cycle"
                                ]
                            ),
                            "severity": severity,
                            "z_score": severity * 5,
                        }
                    ],
                    key=lambda x: x["idx"],
                )

    # If `rul_sequence` wasn't provided or we couldn't find the Healthy -> Impaired transition,
    # - return an empty list. This matches the hackathon requirement:
    # - the anomaly detector is built around the exact moment of the health-state change.
    return []


def calculate_feature_importance(unit_data, predictor=None):
    sensor_cols = [f"s_{i}" for i in range(1, 22)]

    if predictor is not None and predictor.ready:
        imp = predictor.get_feature_importance(unit_data)
        if imp:
            full = {c: imp.get(c, 0.0) for c in sensor_cols}
            total = sum(full.values())
            if total > 0:
                return {k: v / total for k, v in full.items()}
            return full

    if len(unit_data) < 5:
        return {c: 0.0 for c in sensor_cols}

    importances = {}
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


def get_top_contributing_sensors(unit_data, predictor=None, n=3):
    importances = calculate_feature_importance(unit_data, predictor)
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
