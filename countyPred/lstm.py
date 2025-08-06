from sklearn.preprocessing import MinMaxScaler

import data_prex
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping


def create_seq(data, window_size=5):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)


def lstm_init(input_shape, units=50):
    model = Sequential([
        LSTM(units, input_shape=input_shape),
        Dense(input_shape[-1])
    ])
    model.compile('adam', 'mse')
    return model


def lstm_pred(df, window_size=5, n_forecast=12,
              epochs=50, batch_size=16, lstm_units=50):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df.values)
    X, y = create_seq(scaled, window_size)

    model = lstm_init((window_size, df.shape[1]), lstm_units)
    es = EarlyStopping('val_loss', patience=5, restore_best_weights=True)
    model.fit(
        X, y,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es],
        verbose=1
    )
    preds_scaled = []
    window = scaled[-window_size:].tolist()
    for _ in range(n_forecast):
        x_in = np.array(window[-window_size:]).reshape(1, window_size, df.shape[1])
        y_pred = model.predict(x_in)[0]
        preds_scaled.append(y_pred)
        window.append(y_pred)

    preds_scaled = np.vstack(preds_scaled)  # n_forecast, features
    preds = scaler.inverse_transform(preds_scaled)
    last_date = df.index[-1]
    freq = pd.infer_freq(df.index) or 'MS'
    future_idx = pd.date_range(
        start=last_date + pd.tseries.frequencies.to_offset(freq),
        periods=n_forecast, freq=freq
    )
    forecast_df = pd.DataFrame(preds, index=future_idx, columns=df.columns)

    return model, forecast_df


result, _, _ = data_prex.data_prex()
model, future = lstm_pred(result, window_size=5, n_forecast=12, epochs=30, batch_size=8, lstm_units=50)
print(future)
