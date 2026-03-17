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
    calculate_rul,
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


def render_machine_detail(test_data, rul_data, machine_id):
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
    rul = calculate_rul(unit, cycle_idx, test_data, rul_data)
    status, color = get_health_status(rul)
    icon = get_health_icon(status)
    change_points = detect_change_points(unit_data)

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
            f'margin-top:8px">{rul}</div>'
            f'<div style="font-size:0.85rem;color:#8888aa">'
            f"cycles remaining</div></div>",
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

    # ── Simulation Controls ──────────────────────────────────────────────
    st.markdown("")
    sc1, sc2, sc3, _ = st.columns([1.2, 1.2, 1.2, 3.4])
    with sc1:
        if st.button(
            "▶️ Next Cycle",
            type="primary",
            use_container_width=True,
            key="sim_1",
        ):
            advance_cycle(machine_id, max_cycles)
            st.rerun()
    with sc2:
        if st.button("⏩ +10 Cycles", use_container_width=True, key="sim_10"):
            advance_cycles_by(machine_id, 10, max_cycles)
            st.rerun()
    with sc3:
        if st.button("⏭️ +50 Cycles", use_container_width=True, key="sim_50"):
            advance_cycles_by(machine_id, 50, max_cycles)
            st.rerun()

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
    st.plotly_chart(
        sensor_line_chart(unit_data, selected_sensors, SENSOR_NAMES, change_points),
        use_container_width=True,
    )

    # ── Section 2 & 3: RUL Trend + Feature Importance ───────────────────
    col_l, col_r = st.columns(2)
    with col_l:
        rul_final = rul_data.get(unit, 0)
        st.plotly_chart(
            rul_trend_chart(unit_data, rul_final, max_cycles),
            use_container_width=True,
        )
    with col_r:
        importances = calculate_feature_importance(unit_data)
        st.plotly_chart(
            feature_importance_chart(importances, SENSOR_NAMES),
            use_container_width=True,
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

    rul_values = []
    base_idx = len(unit_data) - tail_n
    for i in range(tail_n):
        row_idx = base_idx + i
        r = rul_data.get(unit, 0) + (max_cycles - (row_idx + 1))
        rul_values.append(max(0, r))

    tail_data = tail_data.reset_index(drop=True)
    tail_data["Predicted RUL"] = rul_values
    tail_data = tail_data.iloc[::-1].reset_index(drop=True)

    st.dataframe(tail_data, use_container_width=True, height=400)

    # ── Table: Maintenance Log ───────────────────────────────────────────
    st.markdown("### 🔧 Maintenance Log")
    log = generate_maintenance_log(
        rul, machine["machine_name"], current_cycle, max_cycles
    )
    st.dataframe(
        pd.DataFrame(log), use_container_width=True, hide_index=True
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
            f"Health Status: {status}",
            f"Current Cycle: {current_cycle}",
            f"Max Test Cycles: {max_cycles}",
            f"Anomalies Detected: {len(change_points)}",
        ]
        st.download_button(
            "📥 Export RUL Report",
            "\n".join(report_lines),
            f"{machine_id}_rul_report.txt",
            "text/plain",
            use_container_width=True,
        )
