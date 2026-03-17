"""Quick smoke test for backend logic."""
from utils.data_loader import load_test_data, load_rul_data, get_unit_data, get_max_cycles
from utils.database import init_db, get_all_machines
from utils.calculations import (
    calculate_rul, get_health_status, detect_change_points,
    calculate_feature_importance, generate_maintenance_log,
)

init_db()

# bypass st.cache_data
test_data = load_test_data.__wrapped__()
rul_data = load_rul_data.__wrapped__()

machines = get_all_machines()
print(f"Machines loaded: {len(machines)}")
for m in machines:
    unit = m["unit_number"]
    idx = m["current_cycle_idx"]
    rul = calculate_rul(unit, idx, test_data, rul_data)
    status, _ = get_health_status(rul)
    max_cyc = get_max_cycles(test_data, unit)
    ud = get_unit_data(test_data, unit, idx)
    cps = detect_change_points(ud)
    imp = calculate_feature_importance(ud)
    top3 = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:3]
    top_str = ", ".join(s for s, _ in top3)
    print(
        f"  {m['machine_name']:25s} | Unit {unit:3d} | "
        f"Cycle {idx:3d}/{max_cyc} | RUL {rul:4d} | "
        f"{status:8s} | CPs: {len(cps)} | Top: [{top_str}]"
    )

print("\nAll backend calculations OK!")
