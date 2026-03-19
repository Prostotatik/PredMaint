import streamlit as st
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

COLUMN_NAMES = (
    ["unit", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"s_{i}" for i in range(1, 22)]
)

SENSOR_NAMES = {
    "s_1": "Fan inlet temp (°R)",
    "s_2": "LPC outlet temp (°R)",
    "s_3": "HPC outlet temp (°R)",
    "s_4": "LPT outlet temp (°R)",
    "s_5": "Fan inlet pressure (psia)",
    "s_6": "Bypass duct pressure (psia)",
    "s_7": "HPC outlet pressure (psia)",
    "s_8": "Physical fan speed (rpm)",
    "s_9": "Physical core speed (rpm)",
    "s_10": "Engine pressure ratio",
    "s_11": "HPC outlet static press (psia)",
    "s_12": "Fuel flow ratio (pps/psi)",
    "s_13": "Corrected fan speed (rpm)",
    "s_14": "Corrected core speed (rpm)",
    "s_15": "Bypass ratio",
    "s_16": "Burner fuel-air ratio",
    "s_17": "Bleed enthalpy",
    "s_18": "Demanded fan speed (rpm)",
    "s_19": "Demanded corr. fan speed (rpm)",
    "s_20": "HPT coolant bleed (lbm/s)",
    "s_21": "LPT coolant bleed (lbm/s)",
}

SENSOR_COLS = [f"s_{i}" for i in range(1, 22)]
KEY_SENSORS = ["s_3", "s_7", "s_8", "s_14"]
DEFAULT_DETAIL_SENSORS = ["s_3", "s_4", "s_7", "s_11", "s_13", "s_14"]


@st.cache_data
def load_test_data(fd="FD001"):
    path = os.path.join(DATASET_DIR, f"test_{fd}.txt")
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMN_NAMES)
    return df


@st.cache_data
def load_train_data(fd="FD001"):
    path = os.path.join(DATASET_DIR, f"train_{fd}.txt")
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMN_NAMES)
    return df


@st.cache_data
def load_rul_data(fd="FD001"):
    path = os.path.join(DATASET_DIR, f"RUL_{fd}.txt")
    df = pd.read_csv(path, header=None, names=["rul"])
    return dict(enumerate(df["rul"].values, start=1))


def get_unit_data(test_data, unit_number, max_cycle_idx=None):
    unit_df = test_data[test_data["unit"] == unit_number].reset_index(drop=True)
    if max_cycle_idx is not None:
        unit_df = unit_df.iloc[:max_cycle_idx]
    return unit_df


def get_max_cycles(test_data, unit_number):
    return len(test_data[test_data["unit"] == unit_number])


def get_available_units(used_units, total=100):
    return list(set(range(1, total + 1)) - set(used_units))
