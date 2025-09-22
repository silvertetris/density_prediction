import warnings

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


def _prep_monthly_index(df: pd.DataFrame) -> pd.DataFrame:
    """DatetimeIndex를 월간 빈도(MS)로 정렬/보정."""
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        raise ValueError("wide DataFrame의 index는 DatetimeIndex여야 합니다.")
    out = out.sort_index()
    freq = pd.infer_freq(out.index) or "MS"
    out = out.asfreq(freq)  # 중간 누락 월은 NaN으로
    return out


def _clean_series(s: pd.Series) -> pd.Series:
    """결측/이상치 간단 처리: 선형보간 → 남은 NaN은 ffill/bfill."""
    if s.isna().any():
        s = s.interpolate(limit_direction="both")
        s = s.fillna(method="ffill").fillna(method="bfill")
    return s.astype(float)


def arima_forecast_all(
        wide_df: pd.DataFrame,
        n_years: int = 2,  # 예측 년수
        order: tuple[int, int, int] = (1, 1, 1),
        seasonal_order: tuple[int, int, int, int] = (0, 1, 1, 12),
        enforce_stationarity: bool = False,
        enforce_invertibility: bool = False,
        disp_progress: bool = True,
) -> pd.DataFrame:
    """
    입력
      - wide_df: index=월(DatetimeIndex), columns=동코드(10자리), values=값
    출력
      - 예측 DataFrame (행=미래 월, 열=동코드)
    """
    warnings.filterwarnings("ignore")
    df = _prep_monthly_index(wide_df)
    steps = int(n_years * 12)
    if steps <= 0:
        raise ValueError("n_years는 1 이상의 값이어야 합니다.")

    # 미래 인덱스 만들기
    last_ts = df.index[-1]
    freq = pd.infer_freq(df.index) or "MS"
    future_idx = pd.date_range(start=last_ts, periods=steps + 1, freq=freq)[1:]  # 다음 달부터

    forecasts: dict[str, pd.Series] = {}
    for i, col in enumerate(df.columns, start=1):
        y = _clean_series(df[col])

        # 데이터가 너무 짧으면 건너뜀
        if y.dropna().shape[0] < max(order[1] + seasonal_order[1] * seasonal_order[3] + 5, 12):
            if disp_progress:
                print(f"[SKIP] {col}: 시계열 길이가 짧아 모델링 생략")
            continue

        if disp_progress:
            print(f"[{i}/{df.shape[1]}] Fitting SARIMAX for {col} ...")

        try:
            model = SARIMAX(
                y,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=enforce_stationarity,
                enforce_invertibility=enforce_invertibility,
                # 측정오차(노이즈) 허용 시 measurement_error=True 고려 가능
            )
            res = model.fit(disp=False)
            fc = res.forecast(steps=steps)
            fc.index = future_idx  # 깔끔하게 통일
            forecasts[col] = fc
        except Exception as e:
            if disp_progress:
                print(f"[WARN] {col} 모델 실패 → {e}")

    if not forecasts:
        raise RuntimeError("예측에 성공한 시리즈가 없습니다. 데이터/파라미터를 확인하세요.")

    fc_df = pd.DataFrame(forecasts, index=future_idx).sort_index()
    return fc_df


# 네가 가진 전처리 함수
"""df_long = data_prex(path="../PopulationData/townScale/")
wide = town_wide_data(df_long)  # 행=월, 열=동코드(10자리)

future_arima = arima_forecast_all(
    wide_df=wide,
    n_years=3,                 # 3년 = 36개월
    order=(1,1,1),             # 필요 시 조정
    seasonal_order=(0,1,1,12), # 월별 계절성
    disp_progress=True
)
print(future_arima.shape)
print(future_arima.head())"""
