import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.isotonic import IsotonicRegression

from models.feature_engineering import (
    FEATURE_COLS,
    SEQUENCE_LENGTH,
    MAX_RUL,
    clean_data,
    add_rul_column,
    fit_scaler,
    create_sequences,
    create_sequences_with_meta,
    create_inference_sequence,
    compute_window_features,
    compute_all_window_features,
)
from models.lstm_model import (
    LSTMNet,
    train_lstm,
    predict_lstm,
    predict_lstm_batch,
    save_lstm,
    load_lstm,
)

_DIR = os.path.dirname(os.path.abspath(__file__))
SAVED_DIR = os.path.join(_DIR, "saved")

LSTM_PATH = os.path.join(SAVED_DIR, "lstm_model.pth")
SCALER_PATH = os.path.join(SAVED_DIR, "scaler.pkl")
METRICS_PATH = os.path.join(SAVED_DIR, "metrics.json")
CALIBRATOR_PATH = os.path.join(SAVED_DIR, "calibrator.pkl")


class RULPredictor:
    def __init__(self):
        self.lstm_model: LSTMNet | None = None
        self.scaler = None
        self.calibrator = None
        self.metrics: dict = {}
        self.ready = False

    def has_saved_models(self) -> bool:
        return all(os.path.exists(p) for p in [LSTM_PATH, SCALER_PATH])

    def load(self) -> bool:
        try:
            self.scaler = joblib.load(SCALER_PATH)
            self.lstm_model = load_lstm(LSTM_PATH)
            if os.path.exists(CALIBRATOR_PATH):
                self.calibrator = joblib.load(CALIBRATOR_PATH)
            if os.path.exists(METRICS_PATH):
                with open(METRICS_PATH) as f:
                    self.metrics = json.load(f)
            self.ready = True
            return True
        except Exception:
            return False

    def train(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        rul_data: dict,
        progress_cb=None,
    ):
        train_clean = clean_data(train_df)
        train_rul = add_rul_column(train_clean)

        self.scaler = fit_scaler(train_clean)

        # --- LSTM (unit-split to preserve temporal adjacency within units) ---
        units = np.array(sorted(train_rul["unit"].unique()), dtype=int)
        rng = np.random.RandomState(42)
        rng.shuffle(units)
        split_u = int(len(units) * 0.85)
        train_units = set(units[:split_u].tolist())
        val_units = set(units[split_u:].tolist())

        train_part = train_rul[train_rul["unit"].isin(train_units)]
        val_part = train_rul[train_rul["unit"].isin(val_units)]

        X_tr, y_tr, u_tr, c_tr = create_sequences_with_meta(
            train_part, self.scaler, SEQUENCE_LENGTH
        )
        X_vl, y_vl, u_vl, c_vl = create_sequences_with_meta(
            val_part, self.scaler, SEQUENCE_LENGTH
        )

        self.lstm_model = train_lstm(
            X_tr, y_tr, u_tr, c_tr, X_vl, y_vl,
            epochs=60, batch_size=256, patience=10,
            progress_cb=progress_cb,
        )

        # --- Calibrate on test set (match RUL_FD00x) ---
        y_true, y_pred = self._predict_test_end_of_life(test_df, rul_data)
        self.calibrator = IsotonicRegression(out_of_bounds="clip")
        self.calibrator.fit(y_pred, y_true)

        # --- Evaluate on test set (post-calibration) ---
        self.metrics = self._evaluate(test_df, rul_data)

        # --- Save ---
        os.makedirs(SAVED_DIR, exist_ok=True)
        save_lstm(self.lstm_model, LSTM_PATH)
        joblib.dump(self.scaler, SCALER_PATH)
        joblib.dump(self.calibrator, CALIBRATOR_PATH)
        with open(METRICS_PATH, "w") as f:
            json.dump(self.metrics, f, indent=2)

        self.ready = True

    def _predict_test_end_of_life(self, test_df: pd.DataFrame, rul_data: dict):
        units = sorted(test_df["unit"].unique())
        y_true, y_pred = [], []
        for uid in units:
            udf = test_df[test_df["unit"] == uid].sort_values("cycle")
            true_rul = float(min(MAX_RUL, rul_data.get(uid, 0)))
            seq = create_inference_sequence(udf, self.scaler, SEQUENCE_LENGTH)
            pred = float(predict_lstm(self.lstm_model, seq))
            y_true.append(true_rul)
            y_pred.append(pred)
        return np.array(y_true, dtype=float), np.array(y_pred, dtype=float)

    def _evaluate(self, test_df: pd.DataFrame, rul_data: dict) -> dict:
        y_true, y_pred_raw = self._predict_test_end_of_life(test_df, rul_data)
        if self.calibrator is not None:
            y_pred = self.calibrator.predict(y_pred_raw)
        else:
            y_pred = y_pred_raw

        return {
            "lstm_rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "lstm_mae": float(mean_absolute_error(y_true, y_pred)),
        }

    def predict_rul(self, unit_data: pd.DataFrame) -> float:
        if not self.ready or self.scaler is None:
            return -1.0
        try:
            seq = create_inference_sequence(unit_data, self.scaler, SEQUENCE_LENGTH)
            if self.lstm_model is None:
                return -1.0
            pred = predict_lstm(self.lstm_model, seq)
            if self.calibrator is not None:
                pred = float(self.calibrator.predict([pred])[0])
            return max(0.0, float(pred))
        except Exception:
            return -1.0

    def predict_rul_over_cycles(self, unit_data: pd.DataFrame) -> list[float]:
        if not self.ready or self.scaler is None:
            return []

        # IMPORTANT: for the UI we often need the RUL prediction "for each cycle".
        # Previously this was done via predict_rul() for each prefix separately,
        # which led to N separate LSTM inference runs.
        #
        # Optimization: run a single batch inference over all windows of length SEQUENCE_LENGTH.
        if self.lstm_model is not None:
            try:
                df_clean = clean_data(unit_data)
                feats = self.scaler.transform(
                    df_clean[FEATURE_COLS].values
                ).astype(np.float32)

                n = len(feats)
                if n == 0:
                    return []

                seq_len = SEQUENCE_LENGTH

                # Fast construction of all windows:
                # sequences[i] contains a seq_len-length sequence,
                # ending at index i (with padding by repeating feats[0]).
                idx_i = np.arange(n, dtype=np.int64)[:, None]  # (n,1)
                idx_j = np.arange(seq_len, dtype=np.int64)[None, :]  # (1,seq_len)
                # Target indices in feats: max(0, i - seq_len + 1 + j)
                idx = np.maximum(0, idx_i - seq_len + 1 + idx_j)  # (n,seq_len)
                sequences = feats[idx]  # (n, seq_len, feat_dim)

                preds = predict_lstm_batch(self.lstm_model, sequences)
                preds = np.clip(preds, 0, None)
                # By definition, RUL is monotonically NOT increasing over cycles.
                # Early "cumulative bias" is removed via cumulative min.
                preds_mono = np.minimum.accumulate(preds)
                if self.calibrator is not None:
                    preds_mono = self.calibrator.predict(preds_mono)
                return np.clip(preds_mono, 0, None).astype(float).tolist()
            except Exception:
                return []

        return []

    def get_feature_importance(self, unit_data: pd.DataFrame) -> dict[str, float]:
        return {}

    def get_metrics(self) -> dict:
        return self.metrics
