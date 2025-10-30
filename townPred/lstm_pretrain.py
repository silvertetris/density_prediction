from __future__ import annotations
import os, json, math, random
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
from tensorflow.keras.callbacks import ModelCheckpoint

from townPred.pretrain_plot import plot_trends

# (선택) 네 모듈이 있다면 사용, 없으면 데모 데이터로 대체
try:
    from townPred.data_prex import data_prex
except Exception:
    data_prex = None

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# =========================
# 설정값
# =========================
SEED = 42
N_SPLITS = 5
SEQ_LEN = 5
HORIZON = 1
BATCH_SIZE = 512
EPOCHS = 100
PATIENCE = 10
UNITS = 64
DROPOUT = 0.1
LEARNING_RATE = 1e-3
SHOCK_ENABLE   = False
SHOCK_START    = "2021-03-01"
SHOCK_END      = "2022-09-01"
SHOCK_AMP_FROM = -1000.0
SHOCK_AMP_TO   =  1000.0
SHOCK_TARGET_LOCS: List[str] | None = None   #지역코드 (shock)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
set_seed(SEED)

def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))

def mape(y_true, y_pred, eps: float = 1e-8):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100.0

def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values(["location_code","date"]).reset_index(drop=True)
    return out

def make_time_folds(df: pd.DataFrame, n_splits: int) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    uniq_dates = np.array(sorted(df["date"].unique()))
    fold_boundaries = np.linspace(SEQ_LEN + HORIZON, len(uniq_dates) - 1, n_splits + 1, dtype=int)
    folds = []
    for i in range(n_splits):
        test_start_idx = fold_boundaries[i]
        test_end_idx   = fold_boundaries[i + 1]
        test_start = uniq_dates[test_start_idx]
        test_end   = uniq_dates[test_end_idx]
        folds.append((test_start, test_end))
    return folds

def inject_shock(df: pd.DataFrame,
                 start: str,
                 end: str,
                 amp_start: float,
                 amp_end: float,
                 target_locs: List[str] | None = None) -> pd.DataFrame: #지역코드 locs
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    m = (out["date"] >= pd.Timestamp(start)) & (out["date"] <= pd.Timestamp(end))
    if target_locs is not None:
        m &= out["location_code"].isin(target_locs)

    # 해당 구간의 고유 날짜들에 대해 선형 shock 곡선 생성
    dates = out.loc[m, "date"].sort_values().unique()
    if len(dates) == 0:
        # 적용 구간이 없으면 그대로 반환
        return out

    shock_curve = np.linspace(amp_start, amp_end, len(dates))
    shock_map = {d: v for d, v in zip(dates, shock_curve)}

    # value에 shock 더하기
    out.loc[m, "value"] = (
            out.loc[m, "value"].astype(float)
            + out.loc[m, "date"].map(shock_map).astype(float)
    )
    out["value"] = out["value"].clip(lower=0)  # 음수 방지
    return out

def build_windows_for_range(
        df, train_end_date, test_start_date, test_end_date,
        seq_len, horizon, fit_scaler, scalers=None
):
    if scalers is None:
        scalers = {}

    X_train_list, y_train_list = [], []
    X_test_list,  y_test_list  = [], []
    meta_te = []

    for loc, g in df.groupby("location_code"):
        g = g.sort_values("date").reset_index(drop=True)

        g_train = g[g["date"] < test_start_date] if train_end_date is None else g[g["date"] <= train_end_date]
        g_test  = g[(g["date"] >= test_start_date) & (g["date"] <= test_end_date)]

        # ✅ train 샘플 없으면 이 지역은 완전히 스킵(스케일러도 만들지 않음)
        if len(g_train) == 0:
            continue

        # ✅ 이제 스케일러를 만든 뒤 fit
        if loc not in scalers:
            scalers[loc] = StandardScaler()
        if fit_scaler:
            train_vals_all = g_train["value"].to_numpy(dtype=float).reshape(-1, 1)
            scalers[loc].fit(train_vals_all)

        # ----- 윈도우 만들기 보조 -----
        def make_windows(arr_values, arr_dates, collect_meta: bool):
            Xs, ys, metas = [], [], []
            for t in range(len(arr_values) - seq_len - horizon + 1):
                Xs.append(arr_values[t:t+seq_len])
                ys.append(arr_values[t+seq_len + horizon - 1])
                if collect_meta:
                    metas.append((loc, pd.to_datetime(arr_dates[t+seq_len + horizon - 1])))
            return np.array(Xs), np.array(ys), metas

        # ---- Train ----
        if len(g_train) >= seq_len + horizon:
            train_vals = g_train["value"].to_numpy(dtype=float).reshape(-1, 1)
            train_sc = scalers[loc].transform(train_vals).squeeze(-1)
            Xtr, ytr, _ = make_windows(train_sc, g_train["date"].values, collect_meta=False)
            if len(Xtr):
                X_train_list.append(Xtr[..., np.newaxis])
                y_train_list.append(ytr.reshape(-1, 1))

        # ---- Test ----
        if len(g_test) >= seq_len + horizon:
            test_vals = g_test["value"].to_numpy(dtype=float).reshape(-1, 1)
            test_sc = scalers[loc].transform(test_vals).squeeze(-1)
            Xte, yte, metas = make_windows(test_sc, g_test["date"].values, collect_meta=True)
            if len(Xte):
                X_test_list.append(Xte[..., np.newaxis])
                y_test_list.append(yte.reshape(-1, 1))
                meta_te.extend(metas)

    def cat_or_empty(lst, shape):
        if lst:
            return np.concatenate(lst, axis=0)
        return np.zeros(shape, dtype=float)

    X_train = cat_or_empty(X_train_list, (0, seq_len, 1))
    y_train = np.concatenate(y_train_list, axis=0) if y_train_list else np.zeros((0, 1), dtype=float)
    X_test  = cat_or_empty(X_test_list,  (0, seq_len, 1))
    y_test  = np.concatenate(y_test_list,  axis=0) if y_test_list  else np.zeros((0, 1), dtype=float)

    return X_train, y_train, X_test, y_test, scalers, meta_te



def build_lstm(input_shape, units=64, dropout=0.1, bidirectional=False):
    inp = keras.Input(shape=input_shape)
    if bidirectional:
        x = keras.layers.Bidirectional(keras.layers.LSTM(units, return_sequences=False))(inp)
    else:
        x = keras.layers.LSTM(units, return_sequences=False)(inp)
    if dropout > 0:
        x = keras.layers.Dropout(dropout)(x)
    out = keras.layers.Dense(1)(x)
    model = keras.Model(inp, out)
    opt = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=opt, loss="mse")
    return model


# =========================
# 메인: 교차검증 루프
# =========================
@dataclass
class FoldResult:
    model_type: str
    fold_idx: int
    n_train: int
    n_test: int
    rmse: float
    mae: float
    mape: float


def run_cv_compare(df: pd.DataFrame, loc_focus: str | None = None, return_predictions: bool = True):
    """
    반환 (return_predictions=True일 때):
      df_pred: columns = ["date", "y_true_sc", "y_pred_uni_sc", "y_pred_bi_sc"]
    -> plot_trends(df_pred)로 바로 도식화 가능
    (주의: 이 버전은 '스케일 단위' 지표 계산. 원단위 지표가 필요하면 알려줘!)
    """
    df = prepare_df(df)
    folds = make_time_folds(df, N_SPLITS)

    all_results: List[FoldResult] = []
    pred_rows: List[Tuple[pd.Timestamp, float, float, float]] = []  # (date, y_true_sc, y_pred_uni_sc, y_pred_bi_sc)

    # 공통 콜백
    def make_callbacks(tag: str):
        os.makedirs("./checkpoints", exist_ok=True)
        return [
            ModelCheckpoint(
                filepath=f"./checkpoints/pretrained_{tag}.weights.h5",
                save_best_only=True,
                save_weights_only=True,
                monitor="val_loss",
                mode="min",
                verbose=1
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=PATIENCE, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=max(2, PATIENCE // 2), verbose=0
            ),
        ]

    for fi, (test_start, test_end) in enumerate(folds, start=1):
        # Train 구간은 test_start 이전 전체(확장형)
        train_end = test_start - pd.offsets.MonthBegin(1)  # 직전 달 말
        # 공용 스케일러 딕셔너리(지역별) — 같은 폴드 내 모델 간 공유해서 공정 비교
        scalers: Dict[str, StandardScaler] = {}

        # 데이터 나누기 (fit_scaler=True로 한 번만)
        Xtr, ytr, Xte, yte, scalers, meta_te = build_windows_for_range(
            df, train_end, test_start, test_end,
            seq_len=SEQ_LEN, horizon=HORIZON,
            fit_scaler=True, scalers=scalers
        )
        if len(Xtr) == 0 or len(Xte) == 0:
            print(f"[Fold {fi}] 샘플이 부족하여 건너뜀 (train {Xtr.shape}, test {Xte.shape})")
            continue

        input_shape = (SEQ_LEN, 1)

        # ---- 단방향 LSTM ----
        model_uni = build_lstm(input_shape, units=UNITS, dropout=DROPOUT, bidirectional=False)
        model_uni.fit(
            Xtr, ytr, validation_split=0.1,
            epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0,
            callbacks=make_callbacks("uni")
        )
        y_pred_uni_sc = model_uni.predict(Xte, batch_size=BATCH_SIZE, verbose=0).squeeze(-1)

        # ---- 양방향 LSTM ----
        model_bi = build_lstm(input_shape, units=UNITS, dropout=DROPOUT, bidirectional=True)
        model_bi.fit(
            Xtr, ytr, validation_split=0.1,
            epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0,
            callbacks=make_callbacks("bi")
        )
        y_pred_bi_sc = model_bi.predict(Xte, batch_size=BATCH_SIZE, verbose=0).squeeze(-1)

        # 평가(스케일 단위)
        y_true_sc = yte.squeeze(-1)
        res_uni = FoldResult(
            "LSTM-Uni", fi, len(Xtr), len(Xte),
            rmse=rmse(y_true_sc, y_pred_uni_sc),
            mae=mean_absolute_error(y_true_sc, y_pred_uni_sc),
            mape=mape(y_true_sc, y_pred_uni_sc)
        )
        res_bi = FoldResult(
            "LSTM-Bi", fi, len(Xtr), len(Xte),
            rmse=rmse(y_true_sc, y_pred_bi_sc),
            mae=mean_absolute_error(y_true_sc, y_pred_bi_sc),
            mape=mape(y_true_sc, y_pred_bi_sc)
        )
        all_results.extend([res_uni, res_bi])

        print(f"[Fold {fi}] Uni  RMSE={res_uni.rmse:.4f} MAE={res_uni.mae:.4f} MAPE={res_uni.mape:.2f}%")
        print(f"[Fold {fi}] Bi   RMSE={res_bi.rmse:.4f} MAE={res_bi.mae:.4f} MAPE={res_bi.mape:.2f}%")

        # === 도식화를 위한 예측 수집 ===
        for (loc, d), yt, pu, pb in zip(meta_te, y_true_sc, y_pred_uni_sc, y_pred_bi_sc):
            if (loc_focus is None) or (loc == loc_focus):
                pred_rows.append((pd.to_datetime(d), float(yt), float(pu), float(pb)))

    # 요약 테이블
    if not all_results:
        print("유효한 폴드가 없습니다.")
        if return_predictions:
            return pd.DataFrame(columns=["date", "y_true_sc", "y_pred_uni_sc", "y_pred_bi_sc"])
        return

    summary = (
        pd.DataFrame([r.__dict__ for r in all_results])
        .groupby("model_type")[["rmse","mae","mape"]]
        .mean()
        .sort_values("rmse")
    )
    print("\n=== 교차검증 평균 성능(스케일 단위) ===")
    print(summary)

    best = summary.index[0]
    print(f"\n👉 평균 RMSE 기준 최고 모델: {best}")

    # 최종 예측 DF 생성 & 리턴
    if return_predictions:
        df_pred = pd.DataFrame(
            pred_rows,
            columns=["date","y_true_sc","y_pred_uni_sc","y_pred_bi_sc"]
        ).sort_values("date").reset_index(drop=True)
        return df_pred



if data_prex is not None:
    try:
        df = data_prex()  # columns: date, location_code, value
    except Exception:
        df = None
else:
    df = None
if SHOCK_ENABLE:
    df = inject_shock(
        df,
        start=SHOCK_START,
        end=SHOCK_END,
        amp_start=SHOCK_AMP_FROM,
        amp_end=SHOCK_AMP_TO,
        target_locs=SHOCK_TARGET_LOCS
    )
    print(f"[INFO] Shock injected: {SHOCK_START} ~ {SHOCK_END}, "
          f"{SHOCK_AMP_FROM} → {SHOCK_AMP_TO}, "
          f"targets={SHOCK_TARGET_LOCS if SHOCK_TARGET_LOCS else 'ALL'}")

df_pred = run_cv_compare(df, loc_focus="1114060500", return_predictions=True)
plot_trends(df_pred)

