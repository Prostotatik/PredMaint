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
from utils.data_loader import (
    load_test_data,
    load_train_data,
    load_rul_data,
    get_available_units,
    get_unit_data,
)
from utils.calculations import get_health_status
from models.predictor import RULPredictor
from views.overview import render_overview
from views.machine_detail import render_machine_detail
from views.ai_assistant import render_ai_assistant_widget

# ── AUTO-SIMULATION SPEED ────────────────────────────────────────────────
# How many demo seconds correspond to one "cycle" (current_cycle_idx++)
DEFAULT_CYCLE_DELAY_SEC = 0.5
# ─────────────────────────────────────────────────────────────────────────

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


# ── AI Model Initialization ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_predictor() -> RULPredictor:
    predictor = RULPredictor()
    if predictor.has_saved_models():
        predictor.load()
    return predictor


predictor = get_predictor()


def _train_model():
    train_data = load_train_data()
    progress_bar = st.progress(0, text="Training LSTM model...")
    status_text = st.empty()

    def on_progress(epoch, total, val_rmse):
        pct = min(epoch / total, 0.95)
        progress_bar.progress(pct, text=f"Epoch {epoch}/{total} — Val RMSE: {val_rmse:.2f}")
        status_text.caption(f"Epoch {epoch}/{total} · Validation RMSE: {val_rmse:.2f}")

    predictor.train(train_data, test_data, rul_data, progress_cb=on_progress)
    progress_bar.progress(1.0, text="Training complete!")
    status_text.empty()
    st.cache_resource.clear()


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
        unit_df = get_unit_data(
            test_data, m["unit_number"], m["current_cycle_idx"]
        )
        if predictor.ready and len(unit_df) > 0:
            # Keep RUL consistent with `Machine Fleet Status`:
            # that panel uses `predict_rul_over_cycles()[-1]` (monotonic post-processing).
            rul_series = predictor.predict_rul_over_cycles(unit_df)
            rul = int(round(rul_series[-1])) if rul_series else 0
            _, color = get_health_status(rul)
        else:
            rul = "—"
            color = "#666680"
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

    # ── Auto-Simulation ──────────────────────────────────────────────────
    st.markdown("**Auto-Simulation**")
    c_auto, c_delay = st.columns([2, 1])
    with c_auto:
        st.toggle(
            "🔄 Auto-Simulate",
            key="auto_simulate",
            help="Automatically advance all machines by 1 cycle every interval",
        )
    with c_delay:
        st.number_input(
            "Sec / cycle",
            min_value=0.05,
            max_value=10.0,
            value=float(st.session_state.get("cycle_delay_sec", DEFAULT_CYCLE_DELAY_SEC)),
            step=0.05,
            key="cycle_delay_sec",
        )

    cycle_delay_sec = float(st.session_state.get("cycle_delay_sec", DEFAULT_CYCLE_DELAY_SEC))
    if st.session_state.get("auto_simulate", False):
        st.caption(f"⏱️ Running: 1 cycle every {cycle_delay_sec:.2f}s")

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

    st.divider()

    # ── Model Status ─────────────────────────────────────────────────────
    with st.expander("🤖 AI Model", expanded=True):
        if predictor.ready:
            metrics = predictor.get_metrics()
            st.success("Model loaded ✓")
            m1, m2 = st.columns(2)
            m1.metric("LSTM RMSE", f"{metrics.get('lstm_rmse', 0):.2f}")
            m2.metric("LSTM MAE", f"{metrics.get('lstm_mae', 0):.2f}")
        else:
            st.warning("Model not trained yet")
            if st.button(
                "🚀 Train AI Model",
                use_container_width=True,
                type="primary",
            ):
                _train_model()
                st.rerun()


# ── Main Content (fragment for flicker-free updates) ─────────────────────
_auto_on = st.session_state.get("auto_simulate", False)
cycle_delay_sec = float(st.session_state.get("cycle_delay_sec", DEFAULT_CYCLE_DELAY_SEC))


@st.fragment(run_every=cycle_delay_sec if _auto_on else None)
def _main_content():
    machines = get_all_machines()
    critical = []
    for m in machines:
        unit_df = get_unit_data(
            test_data, m["unit_number"], m["current_cycle_idx"]
        )
        if predictor.ready and len(unit_df) > 0:
            rul_series = predictor.predict_rul_over_cycles(unit_df)
            rul = int(round(rul_series[-1])) if rul_series else 0
            if rul < 15:
                critical.append((m["machine_name"], rul))

    if critical:
        parts = ", ".join(
            f"**{name}** (RUL: {rul})" for name, rul in critical
        )
        st.error(
            f"🚨 CRITICAL ALERT — Machines below 15 cycles RUL: {parts}. "
            "Immediate inspection required!"
        )

    if (
        st.session_state.current_page == "machine_detail"
        and st.session_state.selected_machine_id
    ):
        render_machine_detail(
            test_data,
            rul_data,
            st.session_state.selected_machine_id,
            predictor,
            cycle_delay_sec=cycle_delay_sec,
        )
    else:
        render_overview(
            test_data,
            rul_data,
            predictor,
            cycle_delay_sec=cycle_delay_sec,
        )

    if st.session_state.get("auto_simulate", False):
        _machines = get_all_machines()
        can_advance = any(
            m["current_cycle_idx"]
            < len(test_data[test_data["unit"] == m["unit_number"]])
            for m in _machines
        )
        if can_advance:
            advance_all_cycles(test_data)
        else:
            st.session_state.auto_simulate = False
            st.rerun()


_main_content()

# ── Floating AI Assistant (Gemini 2.5 Flash) ────────────────────────────
render_ai_assistant_widget(
    test_data=test_data,
    predictor=predictor,
    current_page=st.session_state.current_page,
    selected_machine_id=st.session_state.selected_machine_id,
    cycle_delay_sec=cycle_delay_sec,
)
