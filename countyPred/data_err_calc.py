import pandas as pd
import data_prex
import matplotlib.pyplot as plt


def data_err_calc(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df['population_change'] = df['population'].diff()
    # 해당 월에 측정한거니까 ->전 인구수에 현재 태어난거 하고 이런걸 더해야지 shift(1)하면 안되잖아
    df['expected_change'] = (
            df['born'] -
            df['death'] +
            df['immigrants']
    )

    df['residual'] = df['population_change'] - df['expected_change']  # 오차 값
    print(df)
    # 이러면 첫월에만 expected_change가 나오는게 당연함

    # 통계 출력
    print(f"(Mean): {df['residual'].mean():.2f}")
    print(f"(Std): {df['residual'].std():.2f}")
    print(f"(Max): {df['residual'].max():.2f}")
    print(f"(Min): {df['residual'].min():.2f}")
    print(f"(MAE): {df['residual'].abs().mean():.2f}")

    plt.rcParams['font.family'] = 'AppleGothic'
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(12, 4))
    plt.plot(df.index, df['residual'], label='Residual (Actual - Expected)', color='blue')
    plt.axhline(0, color='gray', linestyle='--')
    plt.title("인구 변화 예측 잔차 (Residual over Time)")
    plt.xlabel("Date")
    plt.ylabel("Residual (people)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    '''
    2021년 3월
    2022년 9월
    인구가 갑자기 확 빠짐, 왜 인지 알아내야함
    '''
    return df


a, b, c = data_prex.data_prex()
data_err_calc(a)
