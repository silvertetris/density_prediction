import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import data_prex

def prepare_features(df: pd.DataFrame, lags=(1, 2, 3, 6, 12), ma_windows=(3, 6, 12)):
    """
    df: index=DatetimeIndex, columns=['born','death','immigrants','population']
    반환: feature DF (라그, 이동평균 포함), 타깃 시리즈 dict
    """
    data = df.copy().reset_index().rename(columns={'index': 'DT'})
    data['year'] = data['DT'].dt.year
    data['month'] = data['DT'].dt.month

    # 라그 특징
    for lag in lags:
        for col in ['born', 'death', 'immigrants']:
            data[f'{col}_lag{lag}'] = data[col].shift(lag)

    # 이동평균 특징
    for w in ma_windows:
        for col in ['born', 'death', 'immigrants']:
            data[f'{col}_ma{w}'] = data[col].rolling(window=w).mean()

    # 학습에 쓸 타깃 저장
    targets = {
        'born': data['born'],
        'death': data['death'],
        'immigrants': data['immigrants']
    }

    # 결측치 있는 행 제거
    data.dropna(inplace=True)
    # 피처로 쓸 컬럼
    feature_cols = [c for c in data.columns
                    if c not in ['DT', 'population', 'born', 'death', 'immigrants']]
    return data, feature_cols, targets


def train_models(data: pd.DataFrame, feature_cols, targets: dict):
    """
    XGBoostRegressor 모델 3개를 Callback 방식의 early stopping으로 학습
    """
    models = {}
    tscv = TimeSeriesSplit(n_splits=5)

    # 마지막 Fold를 validation으로 사용
    splits = list(tscv.split(data))
    train_idx, val_idx = splits[-1]
    X_train, X_val = data.iloc[train_idx][feature_cols], data.iloc[val_idx][feature_cols]
    y_train_dict = {name: targ.iloc[train_idx] for name, targ in targets.items()}
    y_val_dict   = {name: targ.iloc[val_idx] for name, targ in targets.items()}

    for name in ['born', 'death', 'immigrants']:
        model = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=4,
            random_state=42
        )
        # Callback 방식으로 early stopping 설정
        es = callback.EarlyStopping(rounds=20, save_best=True)
        model.fit(
            X_train,
            y_train_dict[name],
            eval_set=[(X_val, y_val_dict[name])],
            eval_metric='mae',
            callbacks=[es],
            verbose=False
        )

        # 검증 성능 출력
        y_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val_dict[name], y_pred)
        print(f"{name} MAE: {mae:.2f}")
        models[name] = model

    return models


def forecast_next_year(df: pd.DataFrame, models: dict, lags=(1, 2, 3, 6, 12), ma_windows=(3, 6, 12)):
    """
    df: 원본 시계열 DataFrame (DT 인덱스)
    models: {'born':model, 'death':model, 'immigrants':model}
    반환: pred_df (내년 12개월 예측: columns=['born','death','immigrants','population'])
    """
    last_date = df.index.max()
    # 1개월 단위로 내년 12개 인덱스 생성
    future_idx = pd.date_range(start=last_date + pd.offsets.MonthBegin(),
                               periods=12, freq='MS')
    # 결과 DataFrame 초기화
    pred = pd.DataFrame(index=future_idx, columns=['born', 'death', 'immigrants', 'population'])
    # combined = df[['born','death','immigrants']].append(pred[['born','death','immigrants']])
    combined = df[['born', 'death', 'immigrants']].copy()

    # 순차 예측
    for date in future_idx:
        row = {}
        # 추출한 과거 combined에서 라그·이동평균 계산
        temp = combined.copy()
        temp.loc[date] = [np.nan, np.nan, np.nan]  # 빈 행
        # 라그
        for lag in lags:
            for col in ['born', 'death', 'immigrants']:
                row[f'{col}_lag{lag}'] = temp[col].shift(lag).loc[date]
        # 이동평균
        for w in ma_windows:
            for col in ['born', 'death', 'immigrants']:
                row[f'{col}_ma{w}'] = temp[col].rolling(window=w).mean().loc[date]

        # 연·월
        row['year'] = date.year
        row['month'] = date.month

        Xpred = pd.DataFrame([row])
        # 개별 예측
        pred.loc[date, 'born'] = models['born'].predict(Xpred)[0]
        pred.loc[date, 'death'] = models['death'].predict(Xpred)[0]
        pred.loc[date, 'immigrants'] = models['immigrants'].predict(Xpred)[0]

        # 예측값을 combined에 추가해 다음 예측에 사용
        combined.loc[date] = pred.loc[date, ['born', 'death', 'immigrants']]

    # population 예측 (누적 방식)
    last_pop = df['population'].iloc[-1]
    pop = last_pop
    for date in future_idx:
        pop = pop + pred.loc[date, 'born'] - pred.loc[date, 'death'] + pred.loc[date, 'immigrants']
        pred.loc[date, 'population'] = pop

    return pred


# ─── 실행 예시 ───────────────────────────────────────────────────────────────

# 1) 전처리
result, _, _ = data_prex.data_prex()
# 2) 피처 준비
data_feat, feat_cols, targets = prepare_features(result)
# 3) 모델 학습
models = train_models(data_feat, feat_cols, targets)
# 4) 내년 예측
forecast_df = forecast_next_year(result, models)

print("\n--- Next Year Forecast ---")
print(forecast_df.round(0))
