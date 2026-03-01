from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential


def scaler(df: pd.DataFrame) -> pd.DataFrame:
    """Simple robust normalization used in original experiments."""
    denom = (np.max(df, axis=0) - np.min(df, axis=0)).replace(0, np.nan)
    scaled = (df - np.mean(df, axis=0)) / denom
    return scaled.replace([np.inf, -np.inf], np.nan)


def _prepare_lstm_xy(df: pd.DataFrame, time_step: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if "return" not in df.columns:
        raise ValueError("Input DataFrame must contain 'return' column.")

    y = df["return"]
    X = df.drop(columns=["return"])

    X = scaler(X).dropna(axis=1)
    num_factor = X.shape[1]
    if num_factor == 0:
        raise ValueError("No valid features remain after scaling.")

    today_value_X = np.array(X.iloc[-time_step:, :]).reshape(1, time_step, num_factor)

    y = np.array(y)[1:].reshape(-1, 1)
    X = X.iloc[:-1, :]

    n_samples = X.shape[0] - time_step
    if n_samples <= 0:
        raise ValueError("Not enough rows for selected time_step.")

    X_new = np.zeros((n_samples, time_step, num_factor), dtype=np.float32)
    y_new = np.zeros((n_samples,), dtype=np.float32)

    for i in range(n_samples):
        X_new[i, :, :] = X.values[i : i + time_step, :]
        y_new[i] = y[i + time_step]

    return X_new, y_new, today_value_X


def DL_train(
    df: pd.DataFrame,
    time_step: int = 30,
    epochs: int = 100,
    batch_size: int = 64,
    random_seed: int = 42,
    verbose: int = 0,
) -> float:
    tf.keras.utils.set_random_seed(random_seed)

    X_train, y_train, X_today = _prepare_lstm_xy(df=df, time_step=time_step)
    num_factor = X_train.shape[2]

    model = Sequential(
        [
            LSTM(units=50, return_sequences=True, input_shape=(time_step, num_factor)),
            Dropout(0.2),
            LSTM(units=50, return_sequences=True),
            Dropout(0.2),
            LSTM(units=50, return_sequences=True),
            Dropout(0.2),
            LSTM(units=50),
            Dropout(0.2),
            Dense(units=1),
        ]
    )
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=[tf.keras.metrics.MeanAbsoluteError()])

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=4, min_lr=1e-4),
    ]

    model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=verbose,
    )

    tomorrow_pred = model.predict(X_today, verbose=0)
    return float(np.squeeze(tomorrow_pred))


if __name__ == "__main__":
    sample = pd.read_csv("data_v2/stock_data_DL_folder/CII.csv", index_col=0)
    pred = DL_train(sample)
    print(pred)
