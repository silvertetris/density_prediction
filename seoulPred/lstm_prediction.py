import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

import data_prex


def prepare_lstm_data(df):
    df = df.copy()
    df.columns = ['PRD_DE', 'natural_increase', 'abroad_move', 'internal_move', 'grdp', 'population_density']
    df['prev_density'] = df['population_density'].shift(1)
    df = df.dropna().reset_index(drop=True)

    features = ['natural_increase', 'abroad_move', 'internal_move', 'grdp', 'prev_density']
    target = 'population_density'

    X = df[features]
    y = df[target]

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

    X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

    return X_scaled, y_scaled, scaler_X, scaler_y, df, features


def split_data(X, y, train_ratio=0.8):
    train_size = int(len(X) * train_ratio)
    return X[:train_size], X[train_size:], y[:train_size], y[train_size:]


def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, activation='relu', input_shape=input_shape))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def train_lstm_model(model, X_train, y_train, epochs=50, batch_size=32):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    return model


def evaluate_model(model, X_test, y_test, scaler_y):
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_actual = scaler_y.inverse_transform(y_test)

    mse = mean_squared_error(y_actual, y_pred)
    r2 = r2_score(y_actual, y_pred)

    print("MSE:", mse)
    print("R^2:", r2)

    plt.figure(figsize=(14, 5))
    plt.plot(y_actual, label='Actual Density')
    plt.plot(y_pred, label='Predicted Density', linestyle='--', color='red')
    plt.title('Test Set: Actual vs Predicted Population Density')
    plt.xlabel('Time Index (Test)')
    plt.ylabel('Population Density')
    plt.legend()
    plt.grid(True)
    plt.show()

    return y_pred, y_actual


def forecast_future(model, last_row, scaler_X, scaler_y, features, months=240):
    future_preds = []
    for _ in range(months):
        input_df = pd.DataFrame([[
            last_row['natural_increase'],
            last_row['abroad_move'],
            last_row['internal_move'],
            last_row['grdp'],
            last_row['population_density']
        ]], columns=features)

        input_scaled = scaler_X.transform(input_df)
        input_scaled = input_scaled.reshape((1, 1, len(features)))

        next_scaled = model.predict(input_scaled)
        next_density = scaler_y.inverse_transform(next_scaled)[0][0]
        future_preds.append(next_density)

        last_row['prev_density'] = last_row['population_density']
        last_row['population_density'] = next_density

    return future_preds


def plot_full_prediction(y_scaled, model, X_scaled, scaler_y, future_preds):
    full_pred_scaled = model.predict(X_scaled)
    full_pred = scaler_y.inverse_transform(full_pred_scaled)

    full_density = list(scaler_y.inverse_transform(y_scaled)) + future_preds
    x_range = list(range(len(full_density)))
    train_end = len(full_pred)

    plt.figure(figsize=(16, 5))
    plt.plot(x_range[:train_end], scaler_y.inverse_transform(y_scaled), label='Actual Density')
    plt.plot(x_range[:train_end], full_pred, label='Model Fit', linestyle='--')
    plt.plot(x_range[train_end:], future_preds, label='Future Prediction (20y)', color='red')
    plt.title('Full Range Population Density Prediction')
    plt.xlabel('Time (Months)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.show()



def run_lstm_pipeline(df):
    X_scaled, y_scaled, scaler_X, scaler_y, df, features = prepare_lstm_data(df)
    X_train, X_test, y_train, y_test = split_data(X_scaled, y_scaled)

    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    model = train_lstm_model(model, X_train, y_train)

    y_pred, y_actual = evaluate_model(model, X_test, y_test, scaler_y)

    last_row = df.iloc[-1].copy()
    future_preds = forecast_future(model, last_row, scaler_X, scaler_y, features)
    plot_full_prediction(y_scaled, model, X_scaled, scaler_y, future_preds)


result = run_lstm_pipeline(data_prex.data_prex())
print(result)
