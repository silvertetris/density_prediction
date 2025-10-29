from __future__ import annotations
import os, math, argparse
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd

# ── 선학습 모듈에서 재사용 ──
from townPred.lstm_pretrain import (
    prepare_df, build_windows_for_range, build_lstm,
    SEQ_LEN, HORIZON, UNITS, DROPOUT, LEARNING_RATE, BATCH_SIZE, EPOCHS, PATIENCE
)
from townPred.pretrain_plot import plot_trends

# (선택) 데이터 로더
try:
    from townPred.data_prex import data_prex
except Exception:
    data_prex = None

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler


# =========================
# 유틸
# =========================
def rmse(y_true, y_pred) -> float:
    return math.sqrt(mean_squared_error(y_true, y_pred))

def mape(y_true, y_pred, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100.0

def split_8020_dates(df: pd.DataFrame) -> Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """유니크 월 기준 80:20 분할(날짜) 반환: (train_end, test_start, test_end)"""
    uniq = np.array(sorted(df["date"].unique()))
    cut = int(len(uniq) * 0.8)
    if cut < (SEQ_LEN + HORIZON):
        raise ValueError("데이터가 너무 짧아서 80:20 분할에서 학습 윈도우를 만들기 어렵습니다. SEQ_LEN/HORIZON을 줄여보세요.")
    train_end  = pd.Timestamp(uniq[cut-1])
    test_start = pd.Timestamp(uniq[cut])
    test_end   = pd.Timestamp(uniq[-1])
    return train_end, test_start, test_end


# =========================
# 미래 예측 (recursive) 유틸
# =========================
def _make_future_dates(start_date: pd.Timestamp, months: int) -> List[pd.Timestamp]:
    dates = []
    cur = pd.Timestamp(start_date)
    for _ in range(months):
        # 다음 달 1일로 증가 (월 단위)
        cur = cur + pd.offsets.MonthBegin(1)
        dates.append(pd.Timestamp(cur))
    return dates


def _predict_future_scaled(
    df: pd.DataFrame,
    scalers: Dict[str, StandardScaler],
    model_uni: keras.Model,
    model_bi: keras.Model,
    months: int,
    seq_len: int,
    loc_focus: str | None = None,
) -> pd.DataFrame:
    """
    표준화(스케일) 단위에서 future months 예측을 수행.
    - 각 지역별로 마지막 seq_len 길이의 스케일 값으로 시작해 recursive 예측
    - 결과 컬럼: ["date","y_true_sc","y_pred_uni_sc","y_pred_bi_sc"]
      (y_true_sc는 NaN으로 채움)
    """
    if months <= 0:
        return pd.DataFrame(columns=["date","y_true_sc","y_pred_uni_sc","y_pred_bi_sc"])

    # 전체 마지막 날짜 (전 지역 공통으로 간주)
    global_last_date = pd.to_datetime(df["date"]).max()

    rows: List[Tuple[pd.Timestamp, float, float, float]] = []

    for loc, g in df.groupby("location_code"):
        if (loc_focus is not None) and (loc != loc_focus):
            continue
        if loc not in scalers:
            # 이 지역은 학습 스케일러가 없음 → 건너뜀
            continue

        g = g.sort_values("date").reset_index(drop=True)
        vals = g["value"].to_numpy(dtype=float).reshape(-1, 1)
        if len(vals) < seq_len:
            # 윈도우 길이보다 짧으면 미래 예측 불가
            continue

        sc = scalers[loc]
        sc_vals = sc.transform(vals).squeeze(-1).tolist()  # list로 편의

        # 마지막 seq_len 시퀀스 준비
        window = sc_vals[-seq_len:]

        # 미래 날짜들 (공통 타임라인)
        fut_dates = _make_future_dates(global_last_date, months)

        for d in fut_dates:
            x = np.array(window, dtype=float)[np.newaxis, :, np.newaxis]  # (1, seq_len, 1)

            # 두 모델 모두 예측
            pu = float(model_uni.predict(x, batch_size=1, verbose=0).squeeze(-1))
            pb = float(model_bi.predict(x, batch_size=1, verbose=0).squeeze(-1))

            # y_true_sc는 미래라 없음 → NaN
            rows.append((pd.to_datetime(d), np.nan, pu, pb))

            # recursive 업데이트: 여기서는 uni 예측으로 업데이트하는 대신
            # 더 보수적으로 두 모델 평균을 피드백할 수도 있지만,
            # 일관성을 위해 "uni 예측"을 넣는 방식으로 진행. 원하면 전략 변경 가능.
            next_val = pu
            window = window[1:] + [next_val]

    # 여러 지역의 미래 예측이 섞일 수 있으므로, 날짜별 평균을 취해 하나의 추세로 표시
    if not rows:
        return pd.DataFrame(columns=["date","y_true_sc","y_pred_uni_sc","y_pred_bi_sc"])

    df_future = pd.DataFrame(rows, columns=["date","y_true_sc","y_pred_uni_sc","y_pred_bi_sc"])
    df_future = (
        df_future.groupby("date", as_index=False)
                 .agg({"y_true_sc": "mean", "y_pred_uni_sc": "mean", "y_pred_bi_sc": "mean"})
                 .sort_values("date")
                 .reset_index(drop=True)
    )
    return df_future


# =========================
# 핵심: 80:20에서 Uni+Bi를 같은 타임라인으로 예측 + 미래 N개월
# =========================
def run_8020_both(
    df: pd.DataFrame,
    loc_focus: str | None = None,
    pretrained_uni: str | None = None,
    pretrained_bi: str | None = None,
    save_uni: str | None = "./final_uni.keras",
    save_bi: str | None = "./final_bi.keras",
    finetune_strategy: str = "full",  # "full" or "freeze_lstm"
    forecast_months: int = 0,         # 미래 예측 개월 수 (0이면 안 함)
) -> pd.DataFrame:
    """
    80:20 고정 분할로 Unidirectional & Bidirectional LSTM을
    같은 X_train/X_test(meta_te)에서 학습·예측 후 단일 df_pred로 반환.
    추가로 forecast_months>0이면 미래 예측도 이어 붙임.
    반환 DF 컬럼: ["date","y_true_sc","y_pred_uni_sc","y_pred_bi_sc"] (plot_trends 바로 사용)
    """
    df = prepare_df(df)
    train_end, test_start, test_end = split_8020_dates(df)

    # 지역별 스케일러 & 윈도우 (한 번만 만들고 두 모델 공용 사용)
    scalers: Dict[str, StandardScaler] = {}
    Xtr, ytr, Xte, yte, scalers, meta_te = build_windows_for_range(
        df, train_end, test_start, test_end,
        seq_len=SEQ_LEN, horizon=HORIZON,
        fit_scaler=True, scalers=scalers
    )
    if len(Xtr) == 0 or len(Xte) == 0:
        raise RuntimeError(f"샘플 부족: train {Xtr.shape}, test {Xte.shape}. SEQ_LEN/HORIZON 또는 기간을 조정하세요.")

    input_shape = (SEQ_LEN, 1)
    y_true_sc = yte.squeeze(-1)

    def _train_predict(bidir: bool, pretrained_path: str | None, save_path: str | None, tag: str):
        model = build_lstm(input_shape, units=UNITS, dropout=DROPOUT, bidirectional=bidir)

        # 사전학습 가중치 로드(선택)
        if pretrained_path and os.path.exists(pretrained_path):
            try:
                model.load_weights(pretrained_path)
                print(f"[INFO] Loaded pretrained weights for {tag}: {pretrained_path}")
            except Exception as e:
                print(f"[WARN] Failed to load pretrained weights for {tag}: {e}")

        # 파인튜닝 전략
        if finetune_strategy == "freeze_lstm":
            for layer in model.layers:
                if isinstance(layer, keras.layers.LSTM) or isinstance(layer, keras.layers.Bidirectional):
                    layer.trainable = False
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss="mse")
            model.fit(Xtr, ytr, validation_split=0.1, epochs=max(5, EPOCHS//5), batch_size=BATCH_SIZE, verbose=0)
            for layer in model.layers:
                layer.trainable = True

        # 최종 학습
        cb = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(2, PATIENCE//2), verbose=0),
        ]
        model.fit(Xtr, ytr, validation_split=0.1, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0, callbacks=cb)

        # 저장(선택)
        if save_path:
            try:
                model.save(save_path)
                print(f"[INFO] Saved {tag} model to: {save_path}")
            except Exception as e:
                print(f"[WARN] Failed to save {tag} model: {e}")

        # 예측 (테스트 구간)
        y_pred = model.predict(Xte, batch_size=BATCH_SIZE, verbose=0).squeeze(-1)

        # 로그
        print(f"[Final 80/20] {tag:3s}  RMSE={rmse(y_true_sc, y_pred):.4f} "
              f"MAE={mean_absolute_error(y_true_sc, y_pred):.4f} "
              f"MAPE={mape(y_true_sc, y_pred):.2f}%")
        return y_pred, model

    # 같은 meta_te/Xte 기준으로 두 모델 학습/예측
    y_pred_uni_sc, model_uni = _train_predict(False, pretrained_uni, save_uni, "Uni")
    y_pred_bi_sc,  model_bi  = _train_predict(True,  pretrained_bi,  save_bi,  "Bi")

    # 도식화용 DF 구성 (테스트 구간)
    rows_hist: List[Tuple[pd.Timestamp, float, float, float]] = []
    for (loc, d), yt, pu, pb in zip(meta_te, y_true_sc, y_pred_uni_sc, y_pred_bi_sc):
        if (loc_focus is None) or (loc == loc_focus):
            rows_hist.append((pd.to_datetime(d), float(yt), float(pu), float(pb)))
    df_hist = pd.DataFrame(rows_hist, columns=["date","y_true_sc","y_pred_uni_sc","y_pred_bi_sc"]).sort_values("date")

    # 미래 예측
    if forecast_months > 0:
        df_future = _predict_future_scaled(
            df=df, scalers=scalers,
            model_uni=model_uni, model_bi=model_bi,
            months=forecast_months, seq_len=SEQ_LEN,
            loc_focus=loc_focus
        )
        # 과거 테스트+미래 이어 붙이기 (날짜 기준 정렬)
        df_pred = pd.concat([df_hist, df_future], axis=0, ignore_index=True)
        df_pred = df_pred.sort_values("date").reset_index(drop=True)
    else:
        df_pred = df_hist

    # loc_focus가 너무 타이트해서 비면 전체로 재시도(시각화 확인용)
    if df_pred.empty and (loc_focus is not None):
        print(f"[WARN] loc_focus='{loc_focus}' 대상 테스트/미래 샘플이 없습니다. 전체 평균 추세로 재시도합니다.")
        # 전체로 재수집
        rows_hist = [(pd.to_datetime(d), float(yt), float(pu), float(pb))
                     for (loc, d), yt, pu, pb in zip(meta_te, y_true_sc, y_pred_uni_sc, y_pred_bi_sc)]
        df_hist = pd.DataFrame(rows_hist, columns=["date","y_true_sc","y_pred_uni_sc","y_pred_bi_sc"]).sort_values("date")
        if forecast_months > 0:
            df_future = _predict_future_scaled(
                df=df, scalers=scalers, model_uni=model_uni, model_bi=model_bi,
                months=forecast_months, seq_len=SEQ_LEN, loc_focus=None
            )
            df_pred = pd.concat([df_hist, df_future], axis=0, ignore_index=True).sort_values("date").reset_index(drop=True)
        else:
            df_pred = df_hist

    return df_pred


# =========================
# 실행부
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loc_focus", type=str, default=None, help="특정 지역코드만 도식화(학습은 전체). 예: 11110")
    parser.add_argument("--pretrained_uni", type=str, default="./checkpoints/pretrained_uni.weights.h5")
    parser.add_argument("--pretrained_bi",  type=str, default="./checkpoints/pretrained_bi.weights.h5")
    parser.add_argument("--save_uni", type=str, default="./final_uni.keras")
    parser.add_argument("--save_bi",  type=str, default="./final_bi.keras")
    parser.add_argument("--finetune", choices=["full","freeze_lstm"], default="full")
    parser.add_argument("--forecast_months", type=int, default=12, help="미래 예측 개월 수 (0이면 수행 안 함)")
    parser.add_argument("--no_plot", action="store_true")
    args = parser.parse_args()

    # 데이터 로드
    if data_prex is not None:
        try:
            df = data_prex()   # columns: date, location_code, value
        except Exception:
            df = None
    else:
        df = None

    if df is None:
        raise RuntimeError("data_prex()에서 데이터를 불러오지 못했습니다. 데이터 로더를 연결하세요.")

    # 80:20에서 Uni+Bi 한 번에 예측 (같은 meta_te) + 미래 N개월
    df_pred = run_8020_both(
        df=df,
        loc_focus=args.loc_focus,
        pretrained_uni=args.pretrained_uni,
        pretrained_bi=args.pretrained_bi,
        save_uni=args.save_uni,
        save_bi=args.save_bi,
        finetune_strategy=args.finetune,
        forecast_months=args.forecast_months,
    )

    print("[DEBUG] df_pred shape:", df_pred.shape)
    if not args.no_plot and len(df_pred):
        plot_trends(df_pred)
    else:
        print("[WARN] 도식화할 예측이 없습니다. loc_focus/SEQ_LEN/HORIZON/forecast_months를 조정해보세요.")


main()
