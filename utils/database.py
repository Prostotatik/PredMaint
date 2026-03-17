import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "maintenance.db")

DEMO_MACHINES = [
    ("ENG-001", "Motor Assembly Line 1", 1, 15),       # RUL ~128 → Healthy
    ("ENG-002", "Compressor Unit A", 18, 120),          # RUL ~41  → Impaired
    ("ENG-003", "Turbine Generator 3", 34, 201),        # RUL ~9   → Failing
    ("ENG-004", "Pump Station B2", 42, 140),            # RUL ~26  → Impaired
    ("ENG-005", "Fan Module C1", 76, 202),              # RUL ~13  → Impaired/Alert
]


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    c = conn.cursor()

    try:
        c.execute("PRAGMA table_info(machines)")
        cols = {row["name"] for row in c.fetchall()}
        required = {"machine_id", "machine_name", "unit_number", "current_cycle_idx"}
        if cols and not required.issubset(cols):
            c.execute("DROP TABLE IF EXISTS machines")
    except Exception:
        pass

    c.execute(
        """
        CREATE TABLE IF NOT EXISTS machines (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            machine_id TEXT UNIQUE NOT NULL,
            machine_name TEXT NOT NULL,
            unit_number INTEGER NOT NULL,
            current_cycle_idx INTEGER DEFAULT 15,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    c.execute("SELECT COUNT(*) as cnt FROM machines")
    if c.fetchone()["cnt"] == 0:
        for mid, name, unit, idx in DEMO_MACHINES:
            c.execute(
                "INSERT OR IGNORE INTO machines "
                "(machine_id, machine_name, unit_number, current_cycle_idx) "
                "VALUES (?, ?, ?, ?)",
                (mid, name, unit, idx),
            )

    conn.commit()
    conn.close()


def get_all_machines():
    conn = get_connection()
    c = conn.cursor()
    c.execute(
        "SELECT machine_id, machine_name, unit_number, current_cycle_idx "
        "FROM machines ORDER BY id"
    )
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


def get_machine(machine_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute(
        "SELECT machine_id, machine_name, unit_number, current_cycle_idx "
        "FROM machines WHERE machine_id = ?",
        (machine_id,),
    )
    row = c.fetchone()
    conn.close()
    return dict(row) if row else None


def add_machine(machine_id, machine_name, unit_number):
    try:
        conn = get_connection()
        c = conn.cursor()
        c.execute(
            "INSERT INTO machines "
            "(machine_id, machine_name, unit_number, current_cycle_idx) "
            "VALUES (?, ?, ?, 15)",
            (machine_id, machine_name, unit_number),
        )
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False


def delete_machine(machine_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute("DELETE FROM machines WHERE machine_id = ?", (machine_id,))
    conn.commit()
    conn.close()


def advance_cycle(machine_id, max_cycles):
    conn = get_connection()
    c = conn.cursor()
    c.execute(
        "SELECT current_cycle_idx FROM machines WHERE machine_id = ?",
        (machine_id,),
    )
    row = c.fetchone()
    if row and row["current_cycle_idx"] < max_cycles:
        c.execute(
            "UPDATE machines SET current_cycle_idx = current_cycle_idx + 1 "
            "WHERE machine_id = ?",
            (machine_id,),
        )
    conn.commit()
    conn.close()


def advance_cycles_by(machine_id, steps, max_cycles):
    conn = get_connection()
    c = conn.cursor()
    c.execute(
        "SELECT current_cycle_idx FROM machines WHERE machine_id = ?",
        (machine_id,),
    )
    row = c.fetchone()
    if row:
        new_idx = min(row["current_cycle_idx"] + steps, max_cycles)
        c.execute(
            "UPDATE machines SET current_cycle_idx = ? WHERE machine_id = ?",
            (new_idx, machine_id),
        )
    conn.commit()
    conn.close()


def advance_all_cycles(test_data):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT machine_id, unit_number, current_cycle_idx FROM machines")
    for row in c.fetchall():
        max_cyc = len(test_data[test_data["unit"] == row["unit_number"]])
        if row["current_cycle_idx"] < max_cyc:
            conn.execute(
                "UPDATE machines SET current_cycle_idx = current_cycle_idx + 1 "
                "WHERE machine_id = ?",
                (row["machine_id"],),
            )
    conn.commit()
    conn.close()


def advance_all_cycles_by(test_data, steps=1):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT machine_id, unit_number, current_cycle_idx FROM machines")
    for row in c.fetchall():
        max_cyc = len(test_data[test_data["unit"] == row["unit_number"]])
        new_idx = min(row["current_cycle_idx"] + steps, max_cyc)
        if new_idx > row["current_cycle_idx"]:
            conn.execute(
                "UPDATE machines SET current_cycle_idx = ? WHERE machine_id = ?",
                (new_idx, row["machine_id"]),
            )
    conn.commit()
    conn.close()


def reset_all_cycles():
    conn = get_connection()
    c = conn.cursor()
    c.execute("DELETE FROM machines")
    for mid, name, unit, idx in DEMO_MACHINES:
        c.execute(
            "INSERT OR IGNORE INTO machines "
            "(machine_id, machine_name, unit_number, current_cycle_idx) "
            "VALUES (?, ?, ?, ?)",
            (mid, name, unit, idx),
        )
    conn.commit()
    conn.close()
