
import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import sqlite3
from datetime import datetime, timezone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="Crypto Candle AI",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Crypto Candle Prediction AI")
st.caption("Machine Learning • Paper Testing • No real trades")

DB_FILE = "paper_history.db"

def get_db():
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            coin TEXT,
            timeframe TEXT,
            candle_time TEXT,
            close_price REAL,
            prediction INTEGER,
            probability REAL,
            actual INTEGER,
            created_at TEXT
        )
    """)
    conn.commit()
    return conn

def features_for(df):
    f = pd.DataFrame(index=df.index)
    close = df["Close"]

    f["return_1"] = close.pct_change()
    f["return_3"] = close.pct_change(3)
    f["return_5"] = close.pct_change(5)
    f["body"] = (df["Close"] - df["Open"]) / df["Open"]
    f["range"] = (df["High"] - df["Low"]) / close

    for n in [5, 10, 20]:
        f[f"sma_{n}"] = close / close.rolling(n).mean() - 1

    f["volume_change"] = df["Volume"].pct_change()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    f["rsi"] = 100 - 100 / (1 + rs)

    return f.replace([np.inf, -np.inf], np.nan)

def fetch_data(coin, timeframe):
    exchange = ccxt.kraken({"enableRateLimit": True})
    candles = exchange.fetch_ohlcv(
        coin, timeframe=timeframe, limit=1000
    )
    df = pd.DataFrame(candles, columns=[
        "Time", "Open", "High", "Low", "Close", "Volume"
    ])
    df["Time"] = pd.to_datetime(df["Time"], unit="ms", utc=True)
    # Exclude the potentially unfinished candle.
    return df.iloc[:-1].copy()

conn = get_db()

coin = st.selectbox(
    "Crypto Pair",
    ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD"]
)
timeframe = st.selectbox(
    "Timeframe",
    ["1m", "5m", "15m", "1h", "4h", "1d"]
)

if st.button("🤖 Train & Record Prediction"):
    try:
        with st.spinner("Fetching candles and training model..."):
            df = fetch_data(coin, timeframe)
            f = features_for(df)

            # Label each candle by the direction of the next candle.
            y = (df["Close"].shift(-1) > df["Close"]).astype(float)
            y.iloc[-1] = np.nan

            data = f.copy()
            data["target"] = y
            data = data.dropna()

            if len(data) < 150:
                st.error("Not enough usable historical data.")
                st.stop()

            X = data.drop(columns=["target"])
            labels = data["target"].astype(int)
            split = int(len(data) * 0.8)

            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
            model.fit(X.iloc[:split], labels.iloc[:split])

            test_pred = model.predict(X.iloc[split:])
            accuracy = accuracy_score(
                labels.iloc[split:], test_pred
            )

            latest = f.dropna().iloc[[-1]]
            pred = int(model.predict(latest)[0])
            probs = model.predict_proba(latest)[0]
            up_index = list(model.classes_).index(1)
            up_prob = float(probs[up_index])

            candle_time = df.iloc[-1]["Time"].isoformat()
            close_price = float(df.iloc[-1]["Close"])

            # Save or update one record for this completed candle.
            conn.execute("""
                INSERT INTO predictions
                (coin, timeframe, candle_time, close_price,
                 prediction, probability, actual, created_at)
                VALUES (?, ?, ?, ?, ?, ?, NULL, ?)
                ON CONFLICT DO NOTHING
            """, (
                coin, timeframe, candle_time, close_price,
                pred, up_prob,
                datetime.now(timezone.utc).isoformat()
            ))

            # Avoid duplicate predictions for the same candle.
            conn.execute("""
                DELETE FROM predictions
                WHERE coin=? AND timeframe=? AND candle_time=?
                  AND id NOT IN (
                    SELECT MAX(id) FROM predictions
                    WHERE coin=? AND timeframe=? AND candle_time=?
                  )
            """, (coin, timeframe, candle_time,
                  coin, timeframe, candle_time))
            conn.commit()

        st.success("Prediction recorded!")
        a, b, c = st.columns(3)
        a.metric("Next Candle Estimate", "UP 🟢" if pred else "DOWN 🔴")
        b.metric("Model UP Probability", f"{up_prob * 100:.1f}%")
        c.metric("Historical Test Accuracy", f"{accuracy * 100:.1f}%")

        st.caption(
            "Test accuracy uses a chronological holdout. "
            "It does not guarantee future results."
        )
        st.line_chart(df.set_index("Time")["Close"])

    except Exception as e:
        st.error(f"Error: {e}")

st.divider()
st.header("🧪 Paper Testing Results")

if st.button("🔄 Refresh Results"):
    st.rerun()

# Resolve outcomes only after a later candle has closed.
try:
    df_history = pd.read_sql_query(
        "SELECT * FROM predictions ORDER BY id DESC",
        conn
    )

    for _, row in df_history[df_history["actual"].isna()].iterrows():
        try:
            current = fetch_data(row["coin"], row["timeframe"])
            match = current[current["Time"].map(
                lambda t: t.isoformat() == row["candle_time"]
            )]

            if match.empty:
                continue

            candle_idx = match.index[0]
            next_rows = current[current.index > candle_idx]

            if next_rows.empty:
                continue

            actual_close = float(next_rows.iloc[0]["Close"])
            actual = int(actual_close > float(row["close_price"]))

            conn.execute(
                "UPDATE predictions SET actual=? WHERE id=?",
                (actual, int(row["id"]))
            )
            conn.commit()

        except Exception:
            continue

    df_history = pd.read_sql_query(
        "SELECT * FROM predictions ORDER BY id DESC",
        conn
    )

    if df_history.empty:
        st.info("No predictions recorded yet. Train and record your first prediction.")
    else:
        resolved = df_history.dropna(subset=["actual"]).copy()

        if not resolved.empty:
            resolved["Correct"] = (
                resolved["prediction"].astype(int)
                == resolved["actual"].astype(int)
            )
            total = len(resolved)
            correct = int(resolved["Correct"].sum())
            c1, c2, c3 = st.columns(3)
            c1.metric("Resolved Predictions", total)
            c2.metric("Correct", correct)
            c3.metric("Paper Accuracy", f"{correct / total * 100:.1f}%")
        else:
            st.info("Waiting for later candles to close before scoring predictions.")

        display = df_history.copy()
        display["Prediction"] = display["prediction"].map(
            {1: "UP", 0: "DOWN"}
        )
        display["Outcome"] = display["actual"].map(
            {1.0: "UP", 0.0: "DOWN"}
        ).fillna("Pending")
        st.dataframe(
            display[[
                "created_at", "coin", "timeframe",
                "close_price", "Prediction", "Outcome"
            ]],
            use_container_width=True
        )

except Exception as e:
    st.error(f"Could not load paper history: {e}")

conn.close()
