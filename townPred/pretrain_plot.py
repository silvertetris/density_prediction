import pandas as pd
import matplotlib.pyplot as plt

def plot_trends(df_pred: pd.DataFrame):
    if df_pred is None or df_pred.empty:
        print("[PLOT] empty df_pred.")
        return

    # 보정: 타입/NaN
    dfp = df_pred.copy()
    dfp["date"] = pd.to_datetime(dfp["date"], errors="coerce")
    dfp = dfp.dropna(subset=["date"])
    for c in ["y_true_sc","y_pred_uni_sc","y_pred_bi_sc"]:
        if c not in dfp.columns:
            dfp[c] = np.nan
        dfp[c] = pd.to_numeric(dfp[c], errors="coerce")

    # 날짜별 평균, 전부 NaN이면 조기 종료
    dfm = (dfp.groupby("date", as_index=False)
           .agg({"y_true_sc":"mean","y_pred_uni_sc":"mean","y_pred_bi_sc":"mean"})
           .sort_values("date"))
    if dfm[["y_true_sc","y_pred_uni_sc","y_pred_bi_sc"]].isna().all().all():
        print("[PLOT] all series are NaN after grouping.")
        return

    # ① 단방향
    plt.figure(figsize=(10,4))
    plt.plot(dfm["date"], dfm["y_true_sc"], label="True")
    if not dfm["y_pred_uni_sc"].isna().all():
        plt.plot(dfm["date"], dfm["y_pred_uni_sc"], label="LSTM-Uni")
    plt.title("Prediction Trend - Unidirectional LSTM")
    plt.xlabel("Date"); plt.ylabel("Value (scaled or original)"); plt.legend(); plt.tight_layout()
    plt.show()

    # ② 양방향
    plt.figure(figsize=(10,4))
    plt.plot(dfm["date"], dfm["y_true_sc"], label="True")
    if not dfm["y_pred_bi_sc"].isna().all():
        plt.plot(dfm["date"], dfm["y_pred_bi_sc"], label="LSTM-Bi")
    plt.title("Prediction Trend - Bidirectional LSTM")
    plt.xlabel("Date"); plt.ylabel("Value (scaled or original)"); plt.legend(); plt.tight_layout()
    plt.show()

    # ③ 비교
    plt.figure(figsize=(10,4))
    plt.plot(dfm["date"], dfm["y_true_sc"], label="True")
    if not dfm["y_pred_uni_sc"].isna().all():
        plt.plot(dfm["date"], dfm["y_pred_uni_sc"], label="LSTM-Uni")
    if not dfm["y_pred_bi_sc"].isna().all():
        plt.plot(dfm["date"], dfm["y_pred_bi_sc"], label="LSTM-Bi")
    plt.title("Prediction Trend Comparison (Uni vs Bi)")
    plt.xlabel("Date"); plt.ylabel("Value (scaled or original)"); plt.legend(); plt.tight_layout()
    plt.show()
