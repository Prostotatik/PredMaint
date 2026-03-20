import streamlit as st
import pandas as pd

from utils.data_loader import get_unit_data, get_max_cycles, SENSOR_NAMES
from utils.database import get_all_machines
from utils.calculations import (
    get_health_status,
    get_health_icon,
    detect_change_points,
    get_top_contributing_sensors,
)
from utils.charts import (
    rul_distribution_chart,
    fleet_health_pie,
    fleet_sensor_trends,
    anomaly_timeline,
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


def render_overview(
    test_data,
    rul_data,
    predictor=None,
    cycle_delay_sec: float = 0.5,
):
    st.markdown("## 📊 Global Fleet Overview")

    if predictor is None or not predictor.ready:
        st.warning("AI model not ready. Train/load it to display RUL and anomalies.")
        return

    machines = get_all_machines()
    if not machines:
        st.info(
            "No machines registered. Use the sidebar to add a machine."
        )
        return

    machines_info = []
    anomaly_data = []

    for m in machines:
        unit_data = get_unit_data(
            test_data, m["unit_number"], m["current_cycle_idx"]
        )

        if len(unit_data) == 0:
            continue

        # Forecast RUL for each cycle so the anomaly detector can find
        # the exact Healthy -> Impaired moment in the predicted health.
        rul_series = predictor.predict_rul_over_cycles(unit_data)
        rul = int(round(rul_series[-1])) if rul_series else 0

        status, color = get_health_status(rul)
        top_sensors = get_top_contributing_sensors(unit_data, predictor)
        change_pts = detect_change_points(unit_data, rul_sequence=rul_series)

        for cp in change_pts:
            anomaly_data.append(
                {
                    "machine": m["machine_name"],
                    "cycle": cp["cycle"],
                    "severity": cp["severity"],
                }
            )

        last_cycle = (
            int(unit_data.iloc[-1]["cycle"]) if len(unit_data) > 0 else 0
        )
        max_cyc = get_max_cycles(test_data, m["unit_number"])
        ttf_seconds = float(rul) * float(cycle_delay_sec)

        machines_info.append(
            {
                "machine_id": m["machine_id"],
                "machine_name": m["machine_name"],
                "unit_number": m["unit_number"],
                "current_cycle_idx": m["current_cycle_idx"],
                "rul": rul,
                "ttf_seconds": ttf_seconds,
                "status": status,
                "color": color,
                "top_sensors": top_sensors,
                "last_cycle": last_cycle,
                "max_cycles": max_cyc,
            }
        )

    # ── Summary Metrics ──────────────────────────────────────────────────
    healthy = sum(1 for m in machines_info if m["status"] == "Healthy")
    impaired = sum(1 for m in machines_info if m["status"] == "Impaired")
    failing = sum(1 for m in machines_info if m["status"] == "Failing")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Machines", len(machines_info))
    c2.metric("🟢 Healthy", healthy)
    c3.metric("🟡 Impaired", impaired)
    c4.metric("🔴 Failing", failing)

    st.markdown("---")

    # ── Machine Table ────────────────────────────────────────────────────
    st.markdown("### Machine Fleet Status")

    header = st.columns([1.5, 2.2, 1.3, 1.2, 1, 1.3, 0.8])
    titles = [
        "Machine ID",
        "Machine Name",
        "RUL (cycles)",
        "Health",
        "Last Cycle",
        "Time-to-Failure",
        "",
    ]
    for col, t in zip(header, titles):
        col.markdown(f"**{t}**")

    for m in machines_info:
        cols = st.columns([1.5, 2.2, 1.3, 1.2, 1, 1.3, 0.8])
        cols[0].code(m["machine_id"], language=None)
        cols[1].markdown(m["machine_name"])
        cols[2].markdown(
            f'<span style="color:{m["color"]};font-weight:700;'
            f'font-size:1.3em">{m["rul"]}</span>',
            unsafe_allow_html=True,
        )
        icon = get_health_icon(m["status"])
        cols[3].markdown(f"{icon} {m['status']}")
        cols[4].markdown(str(m["last_cycle"]))
        cols[5].markdown(f"**{_format_seconds(m['ttf_seconds'])}**")
        if cols[6].button("🔍", key=f"view_{m['machine_id']}"):
            st.session_state.current_page = "machine_detail"
            st.session_state.selected_machine_id = m["machine_id"]
            st.rerun()

    st.markdown("---")

    # ── Charts Row 1: RUL Bar + Health Pie ───────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        _ph = st.empty()
        _ph.plotly_chart(
            rul_distribution_chart(machines_info),
            use_container_width=True,
            key="overview_rul_dist",
        )
    with col_b:
        _ph = st.empty()
        _ph.plotly_chart(
            fleet_health_pie(machines_info),
            use_container_width=True,
            key="overview_health_pie",
        )

    # ── Charts Row 2: Sensor Trends + Anomaly Timeline ───────────────────
    col_c, col_d = st.columns([3, 2])
    with col_c:
        _ph = st.empty()
        _ph.plotly_chart(
            fleet_sensor_trends(test_data, machines),
            use_container_width=True,
            key="overview_sensor_trends",
        )
    with col_d:
        _ph = st.empty()
        _ph.plotly_chart(
            anomaly_timeline(anomaly_data),
            use_container_width=True,
            key="overview_anomaly_timeline",
        )

    # ── Export ────────────────────────────────────────────────────────────
    st.markdown("---")
    summary_df = pd.DataFrame(
        [
            {
                "Machine ID": m["machine_id"],
                "Machine Name": m["machine_name"],
                "RUL": m["rul"],
                "Health": m["status"],
                "Last Cycle": m["last_cycle"],
            }
            for m in machines_info
        ]
    )
    st.download_button(
        "📥 Export Fleet Summary (CSV)",
        summary_df.to_csv(index=False),
        "fleet_summary.csv",
        "text/csv",
    )
