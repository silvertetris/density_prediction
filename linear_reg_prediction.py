import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

import data_prex


def run_linear_regression(df: pd.DataFrame):
    """
    인구 밀도 예측을 위한 선형 회귀 모델을 학습하고 평가

    Parameters:
    df (pd.DataFrame): ['PRD_DE', 'natural_increase', 'abroad_move', 'internal_move', 'grdp', 'population_density']

    Returns:
    dict: 평가 지표 및 회귀 계수 데이터프레임 포함
    """
    # 1. 컬럼 정리 및 이전 밀도 추가
    df = df.copy()
    df.columns = ['PRD_DE', 'natural_increase', 'abroad_move', 'internal_move', 'grdp', 'population_density']
    df['prev_density'] = df['population_density'].shift(1)
    df = df.dropna().reset_index(drop=True)

    # 2. 특성과 타겟 정의
    features = ['natural_increase', 'abroad_move', 'internal_move', 'grdp', 'prev_density']
    target = 'population_density'

    X = df[features]
    y = df[target]

    # 3. 정규화
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()

    # 4. 학습/테스트 분할
    train_size = int(len(X_scaled) * 0.8)
    X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
    y_train, y_test = y_scaled[:train_size], y_scaled[train_size:]

    # 5. 모델 학습
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 6. 예측
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
    y_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1))

    # 7. 평가 지표
    mse = mean_squared_error(y_actual, y_pred)
    r2 = r2_score(y_actual, y_pred)

    print(f"\n📈 [Linear Regression Results]")
    print(f"MSE: {mse:.4f}")
    print(f"R^2 Score: {r2:.4f}")

    # 8. 회귀 계수
    coef_df = pd.DataFrame({
        'Feature': features,
        'Coefficient': model.coef_
    }).sort_values(by='Coefficient', key=abs, ascending=False)

    print("\n[회귀 계수 (기여도)]")
    print(coef_df)

    # 9. 시각화
    plt.figure(figsize=(14, 5))
    plt.plot(y_actual, label='Actual Density')
    plt.plot(y_pred, label='Predicted Density', linestyle='--', color='red')
    plt.title('Test Set: Actual vs Predicted Population Density (Linear Regression)')
    plt.xlabel('Time Index (Test)')
    plt.ylabel('Population Density')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 10. 결과 반환
    return {
        'model': model,
        'mse': mse,
        'r2': r2,
        'coefficients': coef_df
    }


results = run_linear_regression(data_prex.data_prex())


print(results['coefficients'])
