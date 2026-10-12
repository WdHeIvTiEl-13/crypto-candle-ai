import streamlit as st
import pandas as pd
import numpy as np
import ccxt

from datetime import datetime, timezone
from supabase import create_client
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# =========================================
# PAGE SETUP
# =========================================

st.set_page_config(
    page_title="Crypto Candle AI",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Crypto Candle Prediction AI")
st.caption(
    "Machine Learning • Paper Testing • No real trades"
)


# =========================================
# SUPABASE CONNECTION
# =========================================

@st.cache_resource
def get_supabase():
    return create_client(
        st.secrets["SUPABASE_URL"],
        st.secrets["SUPABASE_KEY"]
    )


try:
    supabase = get_supabase()
except Exception:
    st.error(
        "Supabase connection failed. Check "
        "SUPABASE_URL and SUPABASE_KEY in Streamlit Secrets."
    )
    st.stop()


# =========================================
# MARKET DATA
# =========================================

@st.cache_resource
def get_exchange():
    return ccxt.kraken({
        "enableRateLimit": True
    })


def fetch_data(coin, timeframe):
    exchange = get_exchange()

    candles = exchange.fetch_ohlcv(
        coin,
        timeframe=timeframe,
        limit=1000
    )

    if not candles or len(candles) < 100:
        raise ValueError("Not enough candle data.")

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

    # Ignore the currently forming candle.
    df = df.iloc[:-1].copy()

    if len(df) < 100:
        raise ValueError("Not enough completed candles.")

    return df.reset_index(drop=True)


# =========================================
# MACHINE LEARNING FEATURES
# =========================================

def features_for(df):
    f = pd.DataFrame(index=df.index)
    close = df["Close"]

    f["return_1"] = close.pct_change()
    f["return_3"] = close.pct_change(3)
    f["return_5"] = close.pct_change(5)

    f["body"] = (
        (df["Close"] - df["Open"]) / df["Open"]
    )

    f["range"] = (
        (df["High"] - df["Low"]) / df["Open"]
    )

    for n in [5, 10, 20]:
        f[f"sma_{n}"] = (
            close / close.rolling(n).mean() - 1
        )

    f["volume_change"] = df["Volume"].pct_change()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()

    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    rsi = rsi.mask((gain == 0) & (loss == 0), 50)
    rsi = rsi.mask((loss == 0) & (gain > 0), 100)
    rsi = rsi.mask((gain == 0) & (loss > 0), 0)

    f["rsi"] = rsi

    return f.replace([np.inf, -np.inf], np.nan)


# =========================================
# CREATE AND SAVE A PREDICTION
# =========================================

def make_prediction(coin, timeframe):
    df = fetch_data(coin, timeframe)
    f = features_for(df)

    # Predict whether the NEXT candle closes
    # above its own opening price.
    target = (
        df["Close"].shift(-1) > df["Open"].shift(-1)
    ).astype(float)

    target.iloc[-1] = np.nan

    training = f.copy()
    training["target"] = target
    training = training.dropna()

    if len(training) < 150:
        raise ValueError(
            "Not enough usable historical training data."
        )

    X = training.drop(columns=["target"])
    y = training["target"].astype(int)

    split = int(len(training) * 0.8)

    if split < 100 or len(training) - split < 20:
        raise ValueError("Not enough data for model testing.")

    # First evaluate on later, unseen candles.
    test_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    test_model.fit(X.iloc[:split], y.iloc[:split])

    test_predictions = test_model.predict(X.iloc[split:])

    test_accuracy = accuracy_score(
        y.iloc[split:],
        test_predictions
    )

    baseline_label = int(y.iloc[:split].mode().iloc[0])

    baseline_accuracy = accuracy_score(
        y.iloc[split:],
        np.full(len(y.iloc[split:]), baseline_label)
    )

    # Retrain on all available labelled data
    # before forecasting the next candle.
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X, y)

    latest_features = f.dropna()

    if latest_features.empty:
        raise ValueError("Could not calculate latest features.")

    latest_idx = latest_features.index[-1]
    latest = latest_features.iloc[[-1]]

    pred = int(model.predict(latest)[0])

    probabilities = model.predict_proba(latest)[0]
    classes = list(model.classes_)

    up_prob = (
        float(probabilities[classes.index(1)])
        if 1 in classes else 0.0
    )

    reference = df.loc[latest_idx]

    candle_time = pd.Timestamp(
        reference["Time"]
    ).isoformat()

    close_price = float(reference["Close"])

    payload = {
        "coin": coin,
        "timeframe": timeframe,
        "candle_time": candle_time,
        "close_price": close_price,
        "prediction": pred,
        "up_probability": up_prob
    }

    # A single record per reference candle.
    supabase.table("predictions").upsert(
        payload,
        on_conflict="coin,timeframe,candle_time"
    ).execute()

    return {
        "prediction": pred,
        "up_probability": up_prob,
        "test_accuracy": test_accuracy,
        "baseline_accuracy": baseline_accuracy,
        "chart": df
    }


# =========================================
# EVALUATE COMPLETED OUTCOMES
# =========================================

def resolve_pending():
    response = (
        supabase.table("predictions")
        .select("*")
        .is_("actual", "null")
        .order("created_at", desc=True)
        .limit(100)
        .execute()
    )

    pending = response.data or []
    cache = {}
    updated = 0

    for row in pending:
        try:
            key = (row["coin"], row["timeframe"])

            if key not in cache:
                cache[key] = fetch_data(*key)

            candles = cache[key]

            reference_time = pd.to_datetime(
                row["candle_time"], utc=True
            )

            # Find the first completed candle after
            # the candle used for the prediction.
            future = candles[
                candles["Time"] > reference_time
            ]

            if future.empty:
                continue

            outcome = future.iloc[0]

            open_price = float(outcome["Open"])
            close_price = float(outcome["Close"])
            high_price = float(outcome["High"])
            low_price = float(outcome["Low"])

            if open_price <= 0:
                continue

            # The predicted direction is evaluated
            # against the next candle's open and close.
            actual = int(close_price > open_price)

            body_change_pct = (
                (close_price - open_price)
                / open_price * 100
            )

            full_range_pct = (
                (high_price - low_price)
                / open_price * 100
            )

            update_data = {
                "actual": actual,
                "outcome_open": open_price,
                "outcome_close": close_price,
                "outcome_high": high_price,
                "outcome_low": low_price,
                "body_change_pct": body_change_pct,
                "full_range_pct": full_range_pct,
                "evaluated_at": datetime.now(
                    timezone.utc
                ).isoformat()
            }

            (
                supabase.table("predictions")
                .update(update_data)
                .eq("id", row["id"])
                .execute()
            )

            updated += 1

        except Exception:
            # One failed record should not stop the rest.
            continue

    return updated


# =========================================
# USER CONTROLS
# =========================================

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
            result = make_prediction(coin, timeframe)

        st.success("Prediction saved to Supabase.")

        a, b, c = st.columns(3)

        a.metric(
            "Next Candle Estimate",
            "UP 🟢" if result["prediction"] == 1
            else "DOWN 🔴"
        )

        b.metric(
            "Model UP Probability",
            f"{result['up_probability'] * 100:.1f}%"
        )

        c.metric(
            "Historical Test Accuracy",
            f"{result['test_accuracy'] * 100:.1f}%"
        )

        st.write(
            "Baseline accuracy:",
            f"{result['baseline_accuracy'] * 100:.1f}%"
        )

        st.caption(
            "Past test accuracy and model probabilities "
            "do not guarantee future results."
        )

        st.line_chart(
            result["chart"].set_index("Time")["Close"]
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")


# =========================================
# PAPER TESTING HISTORY
# =========================================

st.divider()
st.header("🧪 Paper Testing Results")

if st.button("🔄 Refresh & Evaluate Outcomes"):
    with st.spinner("Checking completed candles..."):
        count = resolve_pending()

    st.success(f"Evaluation complete. Updated records: {count}")

try:
    response = (
        supabase.table("predictions")
        .select("*")
        .order("created_at", desc=True)
        .limit(500)
        .execute()
    )

    history = pd.DataFrame(response.data or [])

    if history.empty:
        st.info(
            "No predictions yet. Record your first prediction above."
        )

    else:
        resolved = history.dropna(subset=["actual"]).copy()

        if not resolved.empty:
            resolved["Correct"] = (
                resolved["prediction"].astype(int)
                == resolved["actual"].astype(int)
            )

            total = len(resolved)
            correct = int(resolved["Correct"].sum())

            a, b, c = st.columns(3)
            a.metric("Evaluated", total)
            b.metric("Correct", correct)
            c.metric(
                "Paper Accuracy",
                f"{correct / total * 100:.1f}%"
            )
        else:
            st.info(
                "Outcomes are pending. Refresh after the "
                "next candle has closed."
            )

        def movement_label(value):
            if pd.isna(value):
                return "Pending"

            value = abs(float(value))

            if value < 0.1:
                return "Small move"
            elif value < 0.5:
                return "Moderate move"
            return "Large move"

        history["Prediction"] = history["prediction"].map(
            {1: "UP", 0: "DOWN"}
        )

        history["Outcome"] = history["actual"].map(
            {1.0: "UP", 0.0: "DOWN", 1: "UP", 0: "DOWN"}
        ).fillna("Pending")

        history["Movement"] = history[
            "body_change_pct"
        ].apply(movement_label)

        history["Correct?"] = history.apply(
            lambda row: (
                "Yes" if int(row["prediction"]) == int(row["actual"])
                else "No"
            ) if pd.notna(row["actual"]) else "Pending",
            axis=1
        )

        # Display price values and movement measurements.
        # Ensure columns exist, including for older records.
        for col in [
            "close_price",
            "outcome_open",
            "outcome_close",
            "outcome_high",
            "outcome_low",
            "body_change_pct",
            "full_range_pct",
            "actual",
            "prediction",
            "created_at",
            "coin",
            "timeframe"
        ]:
            if col not in history.columns:
                history[col] = np.nan

        history["Prediction"] = history["prediction"].map(
            {1: "UP", 0: "DOWN"}
        )

        history["Outcome"] = history["actual"].map(
            {1: "UP", 0: "DOWN"}
        ).fillna("Pending")

        def movement_label(value):
            if pd.isna(value):
                return "Pending"

            value = abs(float(value))

            if value < 0.1:
                return "Small move"
            elif value < 0.5:
                return "Moderate move"

            return "Large move"

        history["Movement"] = history[
            "body_change_pct"
        ].apply(movement_label)

        history["Correct?"] = history.apply(
            lambda row: (
                "Yes"
                if int(row["prediction"]) == int(row["actual"])
                else "No"
            ) if pd.notna(row["actual"]) else "Pending",
            axis=1
        )

        display = history[[
            "created_at",
            "coin",
            "timeframe",
            "close_price",
            "Prediction",
            "outcome_open",
            "outcome_close",
            "outcome_high",
            "outcome_low",
            "body_change_pct",
            "full_range_pct",
            "Movement",
            "Correct?"
        ]].rename(columns={
            "created_at": "Prediction Time",
            "coin": "Pair",
            "timeframe": "Timeframe",
            "close_price": "Reference Close",
            "outcome_open": "Next Open",
            "outcome_close": "Next Close",
            "outcome_high": "Next High",
            "outcome_low": "Next Low",
            "body_change_pct": "Body Change %",
            "full_range_pct": "Full Range %"
        })

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Small/Moderate/Large describes candle body movement, "
            "not guaranteed safety. Full Range % shows the high-low "
            "range relative to the candle open. Large wicks can mean "
            "substantial movement even when the candle body is small."
        )

except Exception as e:
    st.error(f"Could not load Supabase history: {e}")
