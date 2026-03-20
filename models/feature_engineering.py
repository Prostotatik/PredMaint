import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = (
    ["op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"s_{i}" for i in range(1, 22)]
)

SENSOR_COLS = [f"s_{i}" for i in range(1, 22)]

MAX_RUL = 125
SEQUENCE_LENGTH = 30
WINDOW_SIZE = 30


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in FEATURE_COLS:
        if col not in df.columns:
            continue
        df[col] = df[col].astype(np.float64)
        median = df[col].median()
        df[col] = df[col].fillna(median)
        mean, std = df[col].mean(), df[col].std()
        if std > 1e-8:
            z = (df[col] - mean) / std
            mask = z.abs() > 3
            df.loc[mask, col] = df.loc[mask, col].clip(mean - 3 * std, mean + 3 * std)
    return df


def add_rul_column(train_df: pd.DataFrame, max_rul: int = MAX_RUL) -> pd.DataFrame:
    df = train_df.copy()
    max_cycles = df.groupby("unit")["cycle"].max().to_dict()
    df["rul"] = df.apply(
        lambda r: min(max_rul, max_cycles[r["unit"]] - r["cycle"]), axis=1
    )
    return df


def fit_scaler(train_df: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train_df[FEATURE_COLS].values)
    return scaler


def create_sequences(
    data: pd.DataFrame, scaler: StandardScaler, seq_length: int = SEQUENCE_LENGTH
):
    all_seq, all_tgt = [], []
    for uid in data["unit"].unique():
        udf = data[data["unit"] == uid].sort_values("cycle")
        feats = scaler.transform(udf[FEATURE_COLS].values)
        ruls = udf["rul"].values

        # To keep the input shape consistent during training, use padding
        # to obtain a fixed-length sequence.
        # (The stability of early behavior is further regulated
        # via monotonicity during inference.)
        if len(feats) < seq_length:
            pad = np.tile(feats[0], (seq_length - len(feats), 1))
            feats = np.vstack([pad, feats])
            ruls = np.concatenate(
                [np.full(seq_length - len(ruls), ruls[0]), ruls]
            )

        for i in range(len(feats) - seq_length + 1):
            all_seq.append(feats[i : i + seq_length])
            all_tgt.append(ruls[i + seq_length - 1])

    return np.array(all_seq, dtype=np.float32), np.array(all_tgt, dtype=np.float32)


def create_sequences_with_meta(
    data: pd.DataFrame, scaler: StandardScaler, seq_length: int = SEQUENCE_LENGTH
):
    """
    Like create_sequences(), but also returns:
    - unit_ids: int array of unit number for each sequence
    - cycle_ends: int array of ending cycle index (as in original CMAPSS cycle column)
    """
    all_seq, all_tgt, all_units, all_cycles = [], [], [], []
    for uid in data["unit"].unique():
        udf = data[data["unit"] == uid].sort_values("cycle")
        feats = scaler.transform(udf[FEATURE_COLS].values)
        ruls = udf["rul"].values
        cycles = udf["cycle"].values.astype(int)

        if len(feats) < seq_length:
            pad = np.tile(feats[0], (seq_length - len(feats), 1))
            feats = np.vstack([pad, feats])
            ruls = np.concatenate([np.full(seq_length - len(ruls), ruls[0]), ruls])
            cycles = np.concatenate(
                [np.full(seq_length - len(cycles), cycles[0]), cycles]
            )

        for i in range(len(feats) - seq_length + 1):
            end = i + seq_length - 1
            all_seq.append(feats[i : i + seq_length])
            all_tgt.append(ruls[end])
            all_units.append(int(uid))
            all_cycles.append(int(cycles[end]))

    return (
        np.array(all_seq, dtype=np.float32),
        np.array(all_tgt, dtype=np.float32),
        np.array(all_units, dtype=np.int64),
        np.array(all_cycles, dtype=np.int64),
    )


def create_inference_sequence(
    unit_data: pd.DataFrame, scaler: StandardScaler, seq_length: int = SEQUENCE_LENGTH
) -> np.ndarray:
    df = clean_data(unit_data)
    feats = scaler.transform(df[FEATURE_COLS].values)
    if len(feats) < seq_length:
        # Pad with the first frame (as in the original implementation),
        # so that metrics on the test set match the distribution seen during training.
        pad = np.tile(feats[0], (seq_length - len(feats), 1))
        feats = np.vstack([pad, feats])
    return feats[-seq_length:].astype(np.float32)


def _slope(col: np.ndarray) -> float:
    n = len(col)
    x = np.arange(n, dtype=np.float64)
    x_m = x.mean()
    denom = ((x - x_m) ** 2).sum()
    if denom < 1e-12:
        return 0.0
    return float(((x - x_m) * (col - col.mean())).sum() / denom)


def compute_window_features(
    unit_data: pd.DataFrame, scaler: StandardScaler, window_size: int = WINDOW_SIZE
) -> np.ndarray:
    df = clean_data(unit_data)
    feats = scaler.transform(df[FEATURE_COLS].values)
    if len(feats) < window_size:
        pad = np.tile(feats[0], (window_size - len(feats), 1))
        feats = np.vstack([pad, feats])
    w = feats[-window_size:]
    parts = [w.mean(0), w.std(0), w.max(0), w.min(0)]
    slopes = np.array([_slope(w[:, j]) for j in range(w.shape[1])])
    parts.append(slopes)
    return np.concatenate(parts).astype(np.float32)


def compute_all_window_features(
    data: pd.DataFrame, scaler: StandardScaler, window_size: int = WINDOW_SIZE
):
    X_list, y_list = [], []
    for uid in data["unit"].unique():
        udf = data[data["unit"] == uid].sort_values("cycle")
        feats = scaler.transform(udf[FEATURE_COLS].values)
        ruls = udf["rul"].values

        if len(feats) < window_size:
            pad = np.tile(feats[0], (window_size - len(feats), 1))
            feats = np.vstack([pad, feats])
            ruls = np.concatenate(
                [np.full(window_size - len(ruls), ruls[0]), ruls]
            )

        for i in range(window_size, len(feats) + 1):
            w = feats[i - window_size : i]
            parts = [w.mean(0), w.std(0), w.max(0), w.min(0)]
            slopes = np.array([_slope(w[:, j]) for j in range(w.shape[1])])
            parts.append(slopes)
            X_list.append(np.concatenate(parts))
            y_list.append(ruls[i - 1])

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)
