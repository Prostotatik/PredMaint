import streamlit as st
import os
import random

from utils.database import (
    init_db,
    get_all_machines,
    add_machine,
    delete_machine,
    advance_all_cycles,
    advance_all_cycles_by,
    reset_all_cycles,
)
from utils.data_loader import load_test_data, load_rul_data, get_available_units
from utils.calculations import calculate_rul, get_health_status
from views.overview import render_overview
from views.machine_detail import render_machine_detail

# ── Page Config (must be first st command) ───────────────────────────────
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────
css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Database + Data ──────────────────────────────────────────────────────
init_db()
test_data = load_test_data()
rul_data = load_rul_data()

# ── Session State ────────────────────────────────────────────────────────
if "current_page" not in st.session_state:
    st.session_state.current_page = "overview"
if "selected_machine_id" not in st.session_state:
    st.session_state.selected_machine_id = None


# ── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div style="text-align:center;padding:10px 0">'
        '<h2 style="color:#ff6b35;margin:0">⚙️ PredMaint</h2>'
        '<p style="color:#8888aa;font-size:0.82rem;margin:4px 0 0">'
        "Industrial Predictive Maintenance</p></div>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Add Machine ──────────────────────────────────────────────────────
    with st.expander("➕ Add New Machine"):
        with st.form("add_machine_form", clear_on_submit=True):
            new_id = st.text_input("Machine ID", placeholder="e.g. ENG-006")
            new_name = st.text_input(
                "Machine Name", placeholder="e.g. Motor Assembly Line 6"
            )
            submitted = st.form_submit_button(
                "Add Machine", use_container_width=True
            )
            if submitted:
                if new_id and new_name:
                    machines = get_all_machines()
                    used = {m["unit_number"] for m in machines}
                    available = get_available_units(used)
                    if available:
                        unit = random.choice(available)
                        if add_machine(new_id.strip(), new_name.strip(), unit):
                            st.success(f"✅ {new_name} added (Unit #{unit})")
                            st.rerun()
                        else:
                            st.error("Machine ID already exists")
                    else:
                        st.error("No available CMAPSS units (max 100)")
                else:
                    st.warning("Please fill in both fields")

    # ── Machine List ─────────────────────────────────────────────────────
    st.markdown("**Registered Machines**")
    machines = get_all_machines()
    if not machines:
        st.caption("No machines yet.")
    for m in machines:
        col_name, col_del = st.columns([5, 1])
        rul = calculate_rul(
            m["unit_number"], m["current_cycle_idx"], test_data, rul_data
        )
        _, color = get_health_status(rul)
        with col_name:
            st.markdown(
                f'<span style="font-size:0.85rem">{m["machine_name"]}</span>'
                f' <span style="color:{color};font-weight:600">({rul})</span>',
                unsafe_allow_html=True,
            )
        with col_del:
            if st.button(
                "🗑️",
                key=f"del_{m['machine_id']}",
                help=f"Delete {m['machine_name']}",
            ):
                delete_machine(m["machine_id"])
                st.rerun()

    st.divider()

    # ── Navigation ───────────────────────────────────────────────────────
    if st.button("📊 Fleet Overview", use_container_width=True):
        st.session_state.current_page = "overview"
        st.rerun()

    st.divider()

    # ── Simulation Controls ──────────────────────────────────────────────
    st.markdown("**Simulation**")
    if st.button(
        "▶️ Simulate Next Cycle (All)",
        use_container_width=True,
        type="primary",
    ):
        advance_all_cycles(test_data)
        st.rerun()

    s1, s2 = st.columns(2)
    with s1:
        if st.button("⏩ +10 All", use_container_width=True):
            advance_all_cycles_by(test_data, 10)
            st.rerun()
    with s2:
        if st.button("🔄 Reset", use_container_width=True):
            reset_all_cycles()
            st.rerun()


# ── Alert Banner ─────────────────────────────────────────────────────────
machines = get_all_machines()
critical = []
for m in machines:
    rul = calculate_rul(
        m["unit_number"], m["current_cycle_idx"], test_data, rul_data
    )
    if rul < 15:
        critical.append((m["machine_name"], rul))

if critical:
    parts = ", ".join(f"**{name}** (RUL: {rul})" for name, rul in critical)
    st.error(
        f"🚨 CRITICAL ALERT — Machines below 15 cycles RUL: {parts}. "
        "Immediate inspection required!"
    )

# ── Main Content ─────────────────────────────────────────────────────────
if st.session_state.current_page == "machine_detail" and st.session_state.selected_machine_id:
    render_machine_detail(
        test_data, rul_data, st.session_state.selected_machine_id
    )
else:
    render_overview(test_data, rul_data)
