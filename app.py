
import streamlit as st
import pandas as pd
import numpy as np
import ccxt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="Crypto Candle Prediction AI",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Crypto Candle Prediction AI")
st.caption("Experimental ML model • Paper testing only")

coin = st.selectbox(
    "Crypto Pair",
    ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD"]
)

timeframe = st.selectbox(
    "Candle Timeframe",
    ["1m", "5m", "15m", "1h", "4h", "1d"]
)


def make_features(df):
    close = df["Close"]
    features = pd.DataFrame(index=df.index)

    features["return_1"] = close.pct_change()
    features["return_3"] = close.pct_change(3)
    features["return_5"] = close.pct_change(5)

    features["body"] = (
        (df["Close"] - df["Open"]) / df["Open"]
    )

    features["range"] = (
        (df["High"] - df["Low"]) / df["Close"]
    )

    for window in [5, 10, 20]:
        features[f"sma_{window}"] = (
            close / close.rolling(window).mean() - 1
        )

    features["volume_change"] = df["Volume"].pct_change()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()

    rs = gain / loss.replace(0, np.nan)
    features["rsi"] = 100 - (100 / (1 + rs))

    features = features.replace([np.inf, -np.inf], np.nan)
    return features


if st.button("🤖 Train Model & Predict Next Candle"):
    try:
        with st.spinner("Fetching market data and training model..."):
            exchange = ccxt.kraken({
    "enableRateLimit": True
})

            candles = exchange.fetch_ohlcv(
                coin,
                timeframe=timeframe,
                limit=1000
            )

            df = pd.DataFrame(
                candles,
                columns=[
                    "Time", "Open", "High",
                    "Low", "Close", "Volume"
                ]
            )

            df["Time"] = pd.to_datetime(
                df["Time"], unit="ms", utc=True
            )

            # Exclude the potentially unfinished candle.
            df = df.iloc[:-1].copy()

            if len(df) < 150:
                st.error("Not enough candle data to train.")
                st.stop()

            features = make_features(df)

            # Label: whether the following candle closes higher.
            target = (df["Close"].shift(-1) > df["Close"]).astype(float)
            target.iloc[-1] = np.nan

            dataset = features.copy()
            dataset["target"] = target
            dataset = dataset.dropna()

            if len(dataset) < 100:
                st.error("Not enough usable data after feature preparation.")
                st.stop()

            X = dataset.drop(columns=["target"])
            y = dataset["target"].astype(int)

            # Time-ordered split: no random shuffling.
            split = int(len(dataset) * 0.8)
            X_train, X_test = X.iloc[:split], X.iloc[split:]
            y_train, y_test = y.iloc[:split], y.iloc[split:]

            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )

            model.fit(X_train, y_train)

            test_predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, test_predictions)

            latest_features = features.dropna().iloc[[-1]]
            prediction = int(model.predict(latest_features)[0])

            probabilities = model.predict_proba(latest_features)[0]
            up_index = list(model.classes_).index(1)
            up_probability = float(probabilities[up_index])

        st.success("Model training and prediction completed.")

        st.subheader(f"{coin} • {timeframe}")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Next Candle Estimate",
                "UP 🟢" if prediction == 1 else "DOWN 🔴"
            )

        with c2:
            st.metric(
                "Model UP Probability",
                f"{up_probability * 100:.1f}%"
            )

        with c3:
            st.metric(
                "Held-out Test Accuracy",
                f"{accuracy * 100:.1f}%"
            )

        st.warning(
            "Experimental estimate only. Model probability is not a "
            "guaranteed real-world chance. Accuracy on historical data "
            "does not guarantee future performance."
        )

        chart = df.set_index("Time")["Close"]
        st.subheader("Historical Closing Prices")
        st.line_chart(chart)

        st.subheader("Latest Completed Candles")
        st.dataframe(
            df[["Time", "Open", "High", "Low", "Close", "Volume"]]
            .tail(10),
            use_container_width=True
        )

        st.caption(
            f"Training candles: {len(X_train)} | "
            f"Test candles: {len(X_test)}"
        )

    except Exception as e:
        st.error(f"Something went wrong: {e}")
        st.info(
            "Check your internet connection and try again. "
            "The exchange may temporarily limit requests."
        )
