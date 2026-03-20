import streamlit as st
import pandas as pd
import numpy as np

from utils.data_loader import (
    get_unit_data,
    get_max_cycles,
    SENSOR_NAMES,
    SENSOR_COLS,
    DEFAULT_DETAIL_SENSORS,
)
from utils.database import get_machine, advance_cycle, advance_cycles_by
from utils.calculations import (
    get_health_status,
    get_health_icon,
    detect_change_points,
    calculate_feature_importance,
    generate_maintenance_log,
)
from utils.charts import (
    sensor_line_chart,
    rul_trend_chart,
    feature_importance_chart,
)


def _format_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60.0
    if minutes < 60:
        return f"{minutes:.1f}min"
    hours = minutes / 60.0
    return f"{hours:.1f}h"


def render_machine_detail(
    test_data,
    rul_data,
    machine_id,
    predictor=None,
    cycle_delay_sec: float = 0.5,
):
    if predictor is None or not predictor.ready:
        st.warning("AI model not ready. Train/load it to display RUL and anomalies.")
        return

    machine = get_machine(machine_id)
    if not machine:
        st.error("Machine not found.")
        if st.button("← Back to Fleet Overview"):
            st.session_state.current_page = "overview"
            st.rerun()
        return

    # ── Back button ──────────────────────────────────────────────────────
    if st.button("← Back to Fleet Overview"):
        st.session_state.current_page = "overview"
        st.rerun()

    unit = machine["unit_number"]
    cycle_idx = machine["current_cycle_idx"]
    max_cycles = get_max_cycles(test_data, unit)
    unit_data = get_unit_data(test_data, unit, cycle_idx)

    if len(unit_data) == 0:
        st.warning("No unit data for current cycle index.")
        return

    with st.spinner("Running LSTM inference..."):
        rul_preds = predictor.predict_rul_over_cycles(unit_data)
        metrics = predictor.get_metrics()
        model_mae = metrics.get("lstm_mae", 12.0)
    rul = int(round(rul_preds[-1])) if rul_preds else 0

    status, color = get_health_status(rul)
    icon = get_health_icon(status)
    change_points = detect_change_points(unit_data, rul_sequence=rul_preds)
    ttf_seconds = float(rul) * float(cycle_delay_sec)

    current_cycle = (
        int(unit_data.iloc[-1]["cycle"]) if len(unit_data) > 0 else 0
    )

    # ── Header ───────────────────────────────────────────────────────────
    st.markdown(
        f"## {machine['machine_id']} — {machine['machine_name']}"
    )

    c1, c2, c3, c4 = st.columns([2, 1.5, 1.5, 1.5])

    with c1:
        st.markdown(
            f'<div style="background:linear-gradient(135deg,#1a1a2e,#16213e);'
            f"border-radius:16px;padding:24px;border:2px solid {color};"
            f'text-align:center">'
            f'<div style="font-size:0.9rem;color:#8888aa">'
            f"Remaining Useful Life</div>"
            f'<div style="font-size:3.5rem;font-weight:800;color:{color};'
            f'line-height:1.1">{rul}</div>'
            f'<div style="font-size:1rem;color:#8888aa">cycles</div></div>',
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f'<div style="background:linear-gradient(135deg,#1a1a2e,#16213e);'
            f"border-radius:12px;padding:20px;border:1px solid #2a2a3e;"
            f'text-align:center">'
            f'<div style="font-size:0.9rem;color:#8888aa">Health Status</div>'
            f'<div style="font-size:2rem;margin-top:10px">{icon}</div>'
            f'<div style="font-size:1.3rem;font-weight:600;color:{color}">'
            f"{status}</div></div>",
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f'<div style="background:linear-gradient(135deg,#1a1a2e,#16213e);'
            f"border-radius:12px;padding:20px;border:1px solid #2a2a3e;"
            f'text-align:center">'
            f'<div style="font-size:0.9rem;color:#8888aa">'
            f"Time-to-Failure</div>"
            f'<div style="font-size:2.2rem;font-weight:700;color:{color};'
            f'margin-top:8px">{_format_seconds(ttf_seconds)}</div>'
            f'<div style="font-size:0.85rem;color:#8888aa">'
            f"remaining time</div></div>",
            unsafe_allow_html=True,
        )

    with c4:
        st.markdown(
            f'<div style="background:linear-gradient(135deg,#1a1a2e,#16213e);'
            f"border-radius:12px;padding:20px;border:1px solid #2a2a3e;"
            f'text-align:center">'
            f'<div style="font-size:0.9rem;color:#8888aa">'
            f"Current Cycle</div>"
            f'<div style="font-size:2.2rem;font-weight:700;color:#448aff;'
            f'margin-top:8px">{current_cycle}</div>'
            f'<div style="font-size:0.85rem;color:#8888aa">'
            f"of {max_cycles} in test set</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Sensor Selection ─────────────────────────────────────────────────
    selected_sensors = st.multiselect(
        "Select sensors to display (max 4)",
        options=SENSOR_COLS,
        default=DEFAULT_DETAIL_SENSORS[:4],
        format_func=lambda s: f"{s} — {SENSOR_NAMES.get(s, '')}",
        max_selections=4,
        key="detail_sensor_select",
    )

    if not selected_sensors:
        selected_sensors = DEFAULT_DETAIL_SENSORS[:4]

    # ── Section 1: Multi-Sensor Dashboard ────────────────────────────────
    _ph = st.empty()
    _ph.plotly_chart(
        sensor_line_chart(unit_data, selected_sensors, SENSOR_NAMES, change_points),
        use_container_width=True,
        key="detail_sensor_chart",
    )

    # ── Section 2 & 3: RUL Trend + Feature Importance ───────────────────
    col_l, col_r = st.columns(2)
    with col_l:
        rul_final = rul_data.get(unit, 0)
        _ph = st.empty()
        _ph.plotly_chart(
            rul_trend_chart(
                unit_data,
                rul_predictions=rul_preds,
                rul_final=rul_final,
                max_cycles=max_cycles,
                mae=model_mae,
            ),
            use_container_width=True,
            key="detail_rul_trend",
        )
    with col_r:
        importances = calculate_feature_importance(unit_data, predictor)
        _ph = st.empty()
        _ph.plotly_chart(
            feature_importance_chart(importances, SENSOR_NAMES),
            use_container_width=True,
            key="detail_feature_importance",
        )

    st.markdown("---")

    # ── Table: Latest 20 Sensor Readings ─────────────────────────────────
    st.markdown("### 📋 Latest Sensor Readings")
    display_cols = [
        "cycle",
        "s_3",
        "s_4",
        "s_7",
        "s_8",
        "s_11",
        "s_12",
        "s_13",
        "s_14",
        "s_21",
    ]
    available_cols = [c for c in display_cols if c in unit_data.columns]

    tail_n = min(20, len(unit_data))
    tail_data = unit_data[available_cols].tail(tail_n).copy()

    if rul_preds is not None:
        tail_preds = rul_preds[-tail_n:]
        tail_data = tail_data.reset_index(drop=True)
        tail_data["Predicted RUL"] = [int(round(p)) for p in tail_preds]
    else:
        # Если предсказания недоступны, не используем ground-truth fallback.
        tail_data = tail_data.reset_index(drop=True)
        tail_data["Predicted RUL"] = np.full((tail_n,), np.nan)

    tail_data = tail_data.iloc[::-1].reset_index(drop=True)

    _ph = st.empty()
    _ph.dataframe(
        tail_data, use_container_width=True, height=400, key="detail_sensor_readings"
    )

    # ── Table: Maintenance Log ───────────────────────────────────────────
    st.markdown("### 🔧 Maintenance Log")
    log = generate_maintenance_log(
        rul, machine["machine_name"], current_cycle, max_cycles
    )
    _ph = st.empty()
    _ph.dataframe(
        pd.DataFrame(log),
        use_container_width=True,
        hide_index=True,
        key="detail_maintenance_log",
    )

    # ── Export ────────────────────────────────────────────────────────────
    st.markdown("---")
    ex1, ex2, _ = st.columns([1.5, 1.5, 4])
    with ex1:
        st.download_button(
            "📥 Export Sensor Data (CSV)",
            unit_data.to_csv(index=False),
            f"{machine_id}_sensor_data.csv",
            "text/csv",
            use_container_width=True,
        )
    with ex2:
        report_lines = [
            f"RUL Report — {machine['machine_name']}",
            f"Machine ID: {machine_id}",
            f"CMAPSS Unit: {unit}",
            f"Current RUL: {rul} cycles",
            f"Time-to-Failure: {_format_seconds(ttf_seconds)}",
            f"Health Status: {status}",
            f"Current Cycle: {current_cycle}",
            f"Max Test Cycles: {max_cycles}",
            f"Anomalies Detected: {len(change_points)}",
            "Model: LSTM (PyTorch)",
        ]
        if predictor is not None and predictor.ready:
            metrics = predictor.get_metrics()
            report_lines.append(
                f"LSTM RMSE: {metrics.get('lstm_rmse', 'N/A'):.2f}"
            )
            report_lines.append(
                f"LSTM MAE: {metrics.get('lstm_mae', 'N/A'):.2f}"
            )
        st.download_button(
            "📥 Export RUL Report",
            "\n".join(report_lines),
            f"{machine_id}_rul_report.txt",
            "text/plain",
            use_container_width=True,
        )
