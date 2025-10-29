import pandas as pd
import matplotlib.pyplot as plt

def plot_trends(df_pred):
    """
    df_pred columns:
      - date (datetime64[ns])
      - y_true_sc (표준화 스케일의 정답; 원단위면 이름만 동일하게)
      - y_pred_uni_sc (단방향 LSTM 예측)
      - y_pred_bi_sc  (양방향 LSTM 예측)
    """
    dfm = (df_pred.groupby("date", as_index=False)
                  .agg({"y_true_sc":"mean","y_pred_uni_sc":"mean","y_pred_bi_sc":"mean"})
                  .sort_values("date"))
    # ① 단방향
    plt.figure(figsize=(10,4))
    plt.plot(dfm["date"], dfm["y_true_sc"], label="True")
    plt.plot(dfm["date"], dfm["y_pred_uni_sc"], label="LSTM-Uni")
    plt.title("Prediction Trend - Unidirectional LSTM")
    plt.xlabel("Date"); plt.ylabel("Value (scaled or original)"); plt.legend(); plt.tight_layout()
    plt.show()

    # ② 양방향
    plt.figure(figsize=(10,4))
    plt.plot(dfm["date"], dfm["y_true_sc"], label="True")
    plt.plot(dfm["date"], dfm["y_pred_bi_sc"], label="LSTM-Bi")
    plt.title("Prediction Trend - Bidirectional LSTM")
    plt.xlabel("Date"); plt.ylabel("Value (scaled or original)"); plt.legend(); plt.tight_layout()
    plt.show()

    # ③ 두 모델 오버레이
    plt.figure(figsize=(10,4))
    plt.plot(dfm["date"], dfm["y_true_sc"], label="True")
    plt.plot(dfm["date"], dfm["y_pred_uni_sc"], label="LSTM-Uni")
    plt.plot(dfm["date"], dfm["y_pred_bi_sc"],  label="LSTM-Bi")
    plt.title("Prediction Trend Comparison (Uni vs Bi)")
    plt.xlabel("Date"); plt.ylabel("Value (scaled or original)"); plt.legend(); plt.tight_layout()
    plt.show()
