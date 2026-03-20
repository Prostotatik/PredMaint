import os
from typing import Any

import streamlit as st

try:
    # Preferred import path for google-genai
    from google import genai  # type: ignore[import]
except Exception:  # pragma: no cover
    genai = None  # type: ignore[assignment]

from utils.calculations import get_health_status
from utils.calculations import (
    calculate_feature_importance,
    detect_change_points,
    generate_maintenance_log,
)
from utils.data_loader import (
    DEFAULT_DETAIL_SENSORS,
    get_unit_data,
)
from utils.database import get_all_machines, get_machine

MODEL_NAME = "gemini-2.5-flash"

FRONTEND_GUIDE = """
Frontend structure and chart map:
- Page: Fleet Overview (`views/overview.py`)
  - "RUL Distribution by Machine" (bar chart)
  - "Fleet Health Overview" (donut pie chart)
  - "Real-time Fleet Sensor Trends" (2x2 sensor trend subplot)
  - "Recent Anomaly Detections (PELT)" (anomaly timeline scatter)
- Page: Machine Detail (`views/machine_detail.py`)
  - "Multi-Sensor Dashboard" (selected sensor lines by cycle)
  - "RUL Prediction Trend (LSTM)" (predicted RUL trend with MAE band)
  - "Degradation Feature Importance (RF)" (feature importance bar chart)
  - "Latest Sensor Readings" table
  - "Maintenance Log" table
Use these exact chart names when directing users.
""".strip()


def _resolve_gemini_api_key() -> str | None:
    # 1) Try environment variables
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    # 2) If not set, try to read from local .env in project root
    if not api_key:
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env_path = os.path.join(root_dir, ".env")
        if os.path.exists(env_path):
            try:
                with open(env_path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if (
                            line.startswith("GEMINI_API_KEY=")
                            or line.startswith("GOOGLE_API_KEY=")
                        ):
                            _, value = line.split("=", 1)
                            value = value.strip().strip("\"' ")
                            if line.startswith("GEMINI_API_KEY="):
                                os.environ["GEMINI_API_KEY"] = value
                            else:
                                os.environ["GOOGLE_API_KEY"] = value
                            api_key = value
                            break
            except Exception:
                pass

    return api_key


@st.cache_resource(show_spinner=False)
def _get_gemini_client(api_key: str) -> Any | None:
    if genai is None:
        return None
    return genai.Client(api_key=api_key)


@st.cache_data(show_spinner=False)
def _build_runtime_snapshot(
    test_data,
    _predictor,
    current_page: str,
    selected_machine_id: str | None,
    cycle_delay_sec: float,
) -> str:
    def _slope(y: list[float] | Any) -> float:
        arr = list(map(float, y))
        if len(arr) < 2:
            return 0.0
        x = list(range(len(arr)))
        # slope via linear fit; small and stable for short windows
        x_mean = sum(x) / len(x)
        y_mean = sum(arr) / len(arr)
        denom = sum((xi - x_mean) ** 2 for xi in x)
        if abs(denom) < 1e-12:
            return 0.0
        num = sum((x[i] - x_mean) * (arr[i] - y_mean) for i in range(len(arr)))
        return float(num / denom)

    def _summarize_tail(df, col: str, tail_n: int = 10) -> str:
        if col not in df.columns or len(df) == 0:
            return f"{col}: N/A"
        tail = df[col].tail(tail_n).astype(float)
        vals = tail.values.tolist()
        last_v = float(vals[-1])
        mn, mx = float(min(vals)), float(max(vals))
        sl = _slope(vals)
        return f"{col}: last={last_v:.3f}, slope={sl:+.4f}, min={mn:.3f}, max={mx:.3f}"

    machines = get_all_machines()
    if _predictor is None or not _predictor.ready:
        return "Model state: not ready."

    lines: list[str] = []
    lines.append(f"Current page: {current_page}")
    lines.append(f"Auto cycle delay (sec/cycle): {cycle_delay_sec:.2f}")
    lines.append(f"Registered machines: {len(machines)}")

    critical: list[tuple[str, int]] = []

    # Fleet key sensors used by overview chart (see utils/charts.py)
    fleet_sensors = ["s_3", "s_7", "s_8", "s_14"]
    fleet_sensor_tail_last: dict[str, list[float]] = {s: [] for s in fleet_sensors}
    fleet_sensor_tail_slope: dict[str, list[float]] = {s: [] for s in fleet_sensors}

    anomalies_snapshot: list[str] = []

    for machine in machines:
        unit_df = get_unit_data(
            test_data, machine["unit_number"], machine["current_cycle_idx"]
        )
        if len(unit_df) == 0:
            continue
        # Use the same RUL source as the UI panels
        # (`predict_rul_over_cycles()[-1]` with monotonic post-processing).
        rul_series = _predictor.predict_rul_over_cycles(unit_df)
        rul = int(round(rul_series[-1])) if rul_series else 0
        status, _ = get_health_status(rul)
        last_cycle = int(unit_df.iloc[-1]["cycle"])
        lines.append(
            (
                f"- {machine['machine_id']} | {machine['machine_name']} | "
                f"RUL={rul} | Health={status} | LastCycle={last_cycle}"
            )
        )
        if rul < 15:
            critical.append((machine["machine_name"], rul))

        # Fleet sensor trend aggregates (tail-based)
        for s in fleet_sensors:
            if s in unit_df.columns and len(unit_df) > 1:
                tail = unit_df[s].tail(10).astype(float).values.tolist()
                fleet_sensor_tail_last[s].append(float(tail[-1]))
                fleet_sensor_tail_slope[s].append(_slope(tail))

        # Anomaly timeline candidates (Healthy -> Impaired transitions)
        try:
            rul_series = _predictor.predict_rul_over_cycles(unit_df)
            change_pts = detect_change_points(unit_df, rul_sequence=rul_series)
            for cp in change_pts[-2:]:  # last 2 per machine
                anomalies_snapshot.append(
                    f"- {machine['machine_name']}: cycle={cp['cycle']} severity={cp['severity']:.3f}"
                )
        except Exception:
            # If inference/anomaly detection fails for a unit, still allow other answers
            pass

    if critical:
        crit = ", ".join(f"{name} (RUL={rul})" for name, rul in critical)
        lines.append(f"Critical alert candidates (RUL < 15): {crit}")
    else:
        lines.append("Critical alert candidates (RUL < 15): none")

    if selected_machine_id:
        machine = get_machine(selected_machine_id)
        if machine:
            lines.append(
                f"Selected machine for detail view: {machine['machine_name']} ({selected_machine_id})"
            )

            unit_df = get_unit_data(
                test_data, machine["unit_number"], machine["current_cycle_idx"]
            )
            if len(unit_df) > 0:
                # RUL summary
                rul_series = _predictor.predict_rul_over_cycles(unit_df)
                if rul_series:
                    selected_rul = int(round(rul_series[-1]))
                    selected_status, _color = get_health_status(selected_rul)
                    lines.append(
                        f"Selected detail summary: RUL={selected_rul} ({selected_status}), cycles shown={len(unit_df)}"
                    )

                    tail_rul = [float(x) for x in rul_series[-10:]]
                    lines.append(
                        "Selected RUL (last 10): " + ", ".join(f"{x:.0f}" for x in tail_rul)
                    )

                # Sensor summaries for currently selected multiselect (if present)
                selected_sensors = st.session_state.get("detail_sensor_select") or list(
                    DEFAULT_DETAIL_SENSORS
                )[:4]
                selected_sensors = [s for s in selected_sensors if s in unit_df.columns]
                if selected_sensors:
                    lines.append("Selected sensor trends (tail-based):")
                    for s in selected_sensors:
                        lines.append("  " + _summarize_tail(unit_df, s, tail_n=10))

                # Feature importance (chart: Degradation Feature Importance (RF))
                try:
                    importances = calculate_feature_importance(unit_df, _predictor)
                    top_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)[
                        :5
                    ]
                    imp_str = ", ".join(
                        f"{s}={v:.3f}" for s, v in top_imp
                    )
                    lines.append("Top feature importance (RF): " + imp_str)
                except Exception:
                    pass

                # Maintenance log (chart/table: Maintenance Log)
                try:
                    current_cycle = int(unit_df.iloc[-1]["cycle"])
                    max_cycles = int(get_unit_data(test_data, machine["unit_number"]).shape[0])
                    selected_rul_for_log = int(round(rul_series[-1])) if rul_series else 0
                    log = generate_maintenance_log(
                        selected_rul_for_log, machine["machine_name"], current_cycle, max_cycles
                    )
                    if log:
                        lines.append("Maintenance log (top entries):")
                        for entry in log[:5]:
                            lines.append(
                                f"- {entry['Priority']}: {entry['Action']} | {entry['Reason']} (Date: {entry['Date']})"
                            )
                except Exception:
                    pass

                # Latest sensor readings (table: Latest Sensor Readings)
                try:
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
                    available_cols = [c for c in display_cols if c in unit_df.columns]
                    tail_n = min(5, len(unit_df))
                    tail_rows = unit_df[available_cols].tail(tail_n).copy()
                    tail_rows = tail_rows.reset_index(drop=True)
                    if rul_series and len(rul_series) >= tail_n:
                        tail_preds = rul_series[-tail_n:]
                        # show only predicted RUL values to connect table to RUL chart
                        tail_rows["Predicted RUL"] = [int(round(p)) for p in tail_preds]
                        lines.append("Latest readings (last cycles):")
                        for i in range(tail_n):
                            r = tail_rows.iloc[i]
                            lines.append(
                                "  cycle="
                                + str(int(r["cycle"]))
                                + ", Predicted RUL="
                                + str(r.get("Predicted RUL"))
                            )
                    else:
                        lines.append("Latest readings (last cycles):")
                        for i in range(tail_n):
                            r = tail_rows.iloc[i]
                            lines.append("  cycle=" + str(int(r["cycle"])))
                except Exception:
                    pass

    # Fleet sensor trend summary across machines
    lines.append("Fleet sensor trends (tail-based aggregate):")
    for s in fleet_sensors:
        last_vals = fleet_sensor_tail_last[s]
        slopes = fleet_sensor_tail_slope[s]
        if not last_vals:
            lines.append(f"- {s}: N/A")
            continue
        last_avg = sum(last_vals) / len(last_vals)
        slope_avg = sum(slopes) / len(slopes) if slopes else 0.0
        lines.append(
            f"- {s}: avg_last={last_avg:.3f}, avg_slope={slope_avg:+.4f}"
        )

    lines.append("Recent anomaly candidates (timeline):")
    if anomalies_snapshot:
        # keep it compact: max 12 lines total
        lines.extend(anomalies_snapshot[-12:])
    else:
        lines.append("- none detected")

    return "\n".join(lines)


def _ask_gemini(
    user_message: str,
    history: list[dict[str, str]],
    snapshot: str,
) -> str:
    api_key = _resolve_gemini_api_key()
    client = _get_gemini_client(api_key) if api_key else None
    if client is None:
        if genai is None:
            return (
                "Gemini client is not available (package `google-genai` is not installed "
                "or import failed). Install `google-genai` in the same environment "
                "where you run `streamlit`."
            )
        return (
            "Gemini API key not found. Set `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) "
            "in environment to enable AI assistant."
        )

    history_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in history[-10:]
    )
    system_prompt = (
        "You are an AI assistant inside a predictive maintenance dashboard. "
        "You must answer in English. Be concise and practical. "
        "When user asks if everything is normal, always reference health statuses, "
        "RUL values, and where to inspect charts. "
        "If maintenance is needed, list machines by priority and explain why.\n\n"
        f"{FRONTEND_GUIDE}\n\n"
        f"Live dashboard snapshot:\n{snapshot}\n\n"
        f"Recent chat history:\n{history_text}"
    )

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                {"role": "user", "parts": [{"text": system_prompt}]},
                {"role": "user", "parts": [{"text": user_message}]},
            ],
        )
        text = (response.text or "").strip()
        if not text:
            return "Couldn’t get a response from Gemini. Please rephrase the question."
        return text
    except Exception as exc:
        return f"Gemini request failed: {exc}"


@st.fragment
def render_ai_assistant_widget(
    test_data,
    predictor,
    current_page: str,
    selected_machine_id: str | None,
    cycle_delay_sec: float,
) -> None:
    if "ai_chat_open" not in st.session_state:
        st.session_state.ai_chat_open = False
    if "ai_chat_messages" not in st.session_state:
        st.session_state.ai_chat_messages = [
            {
                "role": "assistant",
                "content": (
                    "I’m an AI assistant for this dashboard. "
                    "Ask, for example: 'Review the charts—everything looks normal?'"
                ),
            }
        ]
    if "ai_chat_draft" not in st.session_state:
        st.session_state.ai_chat_draft = ""

    # NOTE: It's important to style the Streamlit container by `key`,
    # otherwise you may end up with the class drawn on top but not wrapping the chat components.
    st.markdown(
        """
        <style>
        .st-key-ai_chat_card {
            position: fixed;
            right: 20px;
            bottom: 20px;
            width: min(380px, calc(100vw - 30px));
            z-index: 9999;

            background: #12121f;
            border: 1px solid #2a2a3e;
            border-radius: 14px;
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.35);
            padding: 10px;
        }

        /* Closed state: only the clickable bot avatar (no big card). */
        .st-key-ai_chat_launcher {
            position: fixed;
            right: 20px;
            bottom: 20px;
            z-index: 9999;
            width: 66px;
            height: 66px;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 0;
            margin: 0;
            background: transparent;
            border: none;
            box-shadow: none;
        }

        /* The chat should scroll inside the card */
        .st-key-ai_chat_card [data-testid="stChatMessage"] {
            padding-top: 6px;
            padding-bottom: 6px;
        }

        /* Real scroll area (must wrap the st.chat_message DOM nodes). */
        .st-key-ai_chat_scroll_area {
            max-height: 500px;
            overflow-y: auto;
            padding-right: 4px; /* space so scrollbar doesn't overlap text */
        }

        /* Keep message text padding symmetric to avatar side */
        .st-key-ai_chat_card [data-testid="stChatMessageContent"] {
            padding-left: 10px;
            padding-right: 10px;
        }

        /* Floating launcher button (collapsed state) */
        .st-key-open_ai_chat button {
            width: 66px !important;
            min-width: 66px !important;
            max-width: 66px !important;
            height: 66px !important;
            border-radius: 9999px !important;
            padding: 0 !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            margin: 0 !important;
            background: #1f1f33 !important;
            border: 1px solid #2a2a3e !important;
            color: #ffffff !important;
            box-shadow: 0 6px 16px rgba(0, 0, 0, 0.35) !important;
            line-height: 1 !important;
        }

        .st-key-open_ai_chat button p,
        .st-key-open_ai_chat button span,
        .st-key-open_ai_chat button [data-testid="stMarkdownContainer"] p {
            font-size: 3rem !important;
            line-height: 1 !important;
            margin: 0 !important;
        }

        .st-key-open_ai_chat button [data-testid="stIconMaterial"] {
            color: #ffffff !important;
            font-size: 18px !important;
            line-height: 1 !important;
        }

        .st-key-open_ai_chat button:hover {
            background: #2a2a3e !important;
            border-color: #3a3a55 !important;
        }

        /* Center emoji icons in clear/close square buttons */
        .st-key-clear_ai_chat button,
        .st-key-close_ai_chat button {
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            text-align: center !important;
            padding: 0 !important;
            line-height: 1 !important;
            width: 36px !important;
            min-width: 36px !important;
            max-width: 36px !important;
            height: 36px !important;
            margin-left: auto !important;
            margin-right: 0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Closed state: render only the launcher avatar (no big rectangle).
    if not st.session_state.ai_chat_open:
        with st.container(key="ai_chat_launcher"):
            if st.button("🤖", use_container_width=False, key="open_ai_chat"):
                st.session_state.ai_chat_open = True
                st.rerun(scope="fragment")
        return

    # Open state: render the full chat card.
    with st.container(key="ai_chat_card"):
        c_title, c_spacer, c_clear, c_close = st.columns([3.0, 3.8, 0.9, 0.9])
        with c_title:
            st.markdown("**🤖 AI Assistant**")
        with c_spacer:
            st.markdown("")

        with c_clear:
            if st.button(
                "🧹", key="clear_ai_chat", help="Clear chat", use_container_width=False
            ):
                st.session_state.ai_chat_messages = st.session_state.ai_chat_messages[:1]

        with c_close:
            if st.button(
                "✖", key="close_ai_chat", help="Close assistant", use_container_width=False
            ):
                st.session_state.ai_chat_open = False
                st.rerun(scope="fragment")
                return

        # Placeholder for messages (we'll update it after a possible send)
        chat_box = st.empty()

        st.caption("Questions: chart status, maintenance priorities, and where to look in the UI.")
        c_inp, c_send = st.columns([4, 1])
        with c_inp:
            st.text_input(
                "Question",
                key="ai_chat_draft",
                placeholder="For example: What needs maintenance right now?",
                label_visibility="collapsed",
            )
        with c_send:
            send = st.button("Send", key="send_ai_chat", use_container_width=True)

        reply_needed = False
        user_text_for_reply = ""
        assistant_placeholder = None

        if send:
            user_text = st.session_state.ai_chat_draft.strip()
            if user_text:
                st.session_state.ai_chat_messages.append({"role": "user", "content": user_text})
                reply_needed = True
                user_text_for_reply = user_text
                # In Streamlit you can't modify the `session_state` key
                # that's bound to the widget (`st.text_input(key="ai_chat_draft")`)
                # after the widget has been instantiated within the same render pass.
                # That's why we don't clear the input field programmatically.

        # Render messages (including the just-added reply)
        with chat_box.container():
            with st.container(key="ai_chat_scroll_area"):
                for msg in st.session_state.ai_chat_messages[-14:]:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

                # Show "Thinking..." immediately, before the long request completes.
                # That way, the chat doesn't look empty/missing while waiting for the response.
                if reply_needed:
                    with st.chat_message("assistant"):
                        assistant_placeholder = st.empty()
                        assistant_placeholder.markdown("Thinking...")

        # Long Gemini call after the UI has been rendered.
        if reply_needed and user_text_for_reply:
            snapshot = _build_runtime_snapshot(
                test_data=test_data,
                _predictor=predictor,
                current_page=current_page,
                selected_machine_id=selected_machine_id,
                cycle_delay_sec=cycle_delay_sec,
            )
            reply = _ask_gemini(
                user_message=user_text_for_reply,
                history=st.session_state.ai_chat_messages,
                snapshot=snapshot,
            )
            st.session_state.ai_chat_messages.append(
                {"role": "assistant", "content": reply}
            )
            if assistant_placeholder is not None:
                assistant_placeholder.markdown(reply)
