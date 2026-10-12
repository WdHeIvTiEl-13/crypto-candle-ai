import streamlit as st
import pandas as pd
import numpy as np
import ccxt

from datetime import datetime, timezone
from supabase import create_client
from streamlit_autorefresh import st_autorefresh
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# =========================================
# PAGE SETUP AND AUTO REFRESH
# =========================================

st.set_page_config(
    page_title="Crypto Candle AI",
    page_icon="📈",
    layout="wide"
)

# Rerun this Streamlit page every 60 seconds.
st_autorefresh(
    interval=60_000,
    limit=None,
    key="crypto_auto_refresh"
)

st.title("📈 Crypto Candle Prediction AI")
st.caption(
    "Automatic refresh • Automatic prediction • Paper testing only"
)

st.caption(
    "Last check (UTC): "
    + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
)


# =========================================
# SUPABASE
# =========================================

@st.cache_resource
def get_supabase():
    return create_client(
        st.secrets["SUPABASE_URL"],
        st.secrets["SUPABASE_KEY"]
    )


try:
    supabase = get_supabase()
except Exception as e:
    st.error(f"Supabase connection failed: {e}")
    st.stop()


# =========================================
# KRAKEN MARKET DATA
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

    # Exclude the possibly unfinished current candle.
    df = df.iloc[:-1].copy().reset_index(drop=True)

    if len(df) < 100:
        raise ValueError("Not enough completed candles.")

    return df


# =========================================
# FEATURE ENGINEERING
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
# LATEST SAVED MODEL METRICS
# =========================================

def latest_model_metrics(coin, timeframe):
    try:
        response = (
            supabase.table("model_runs")
            .select("test_accuracy,baseline_accuracy")
            .eq("coin", coin)
            .eq("timeframe", timeframe)
            .order("trained_at", desc=True)
            .limit(1)
            .execute()
        )

        if response.data:
            return response.data[0]

    except Exception:
        pass

    return {}


# =========================================
# AUTOMATIC PREDICTION
# =========================================

def make_prediction(coin, timeframe):
    df = fetch_data(coin, timeframe)
    f = features_for(df)

    latest_features = f.dropna()

    if latest_features.empty:
        raise ValueError("Could not calculate the latest features.")

    latest_idx = latest_features.index[-1]
    latest = latest_features.iloc[[-1]]

    reference = df.loc[latest_idx]
    candle_time = pd.Timestamp(reference["Time"]).isoformat()
    close_price = float(reference["Close"])

    # Check before training: do not create another
    # prediction for the same completed reference candle.
    existing = (
        supabase.table("predictions")
        .select("id,prediction,up_probability")
        .eq("coin", coin)
        .eq("timeframe", timeframe)
        .eq("candle_time", candle_time)
        .limit(1)
        .execute()
    )

    if existing.data:
        saved = existing.data[0]

        return {
            "is_new": False,
            "prediction": int(saved["prediction"]),
            "up_probability": saved.get("up_probability"),
            "chart": df,
            "candle_time": candle_time
        }

    # Label each candle by the next candle's
    # close relative to that next candle's open.
    target = (
        df["Close"].shift(-1) > df["Open"].shift(-1)
    ).astype(float)

    target.iloc[-1] = np.nan

    training = f.copy()
    training["target"] = target
    training = training.dropna()

    if len(training) < 150:
        raise ValueError("Not enough usable training data.")

    X = training.drop(columns=["target"])
    y = training["target"].astype(int)

    split = int(len(training) * 0.8)

    if split < 100 or len(training) - split < 20:
        raise ValueError("Not enough data for model testing.")

    # Chronological holdout test.
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

    test_accuracy = float(
        accuracy_score(y.iloc[split:], test_predictions)
    )

    baseline_label = int(y.iloc[:split].mode().iloc[0])

    baseline_accuracy = float(
        accuracy_score(
            y.iloc[split:],
            np.full(len(y.iloc[split:]), baseline_label)
        )
    )

    # Fit the forecasting model using all labelled data.
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X, y)

    prediction = int(model.predict(latest)[0])

    probabilities = model.predict_proba(latest)[0]
    classes = list(model.classes_)

    up_probability = (
        float(probabilities[classes.index(1)])
        if 1 in classes else 0.0
    )

    payload = {
        "coin": coin,
        "timeframe": timeframe,
        "candle_time": candle_time,
        "close_price": close_price,
        "prediction": prediction,
        "up_probability": up_probability
    }

    # Insert rather than upsert so an existing prediction
    # is not silently overwritten.
    supabase.table("predictions").insert(payload).execute()

    # Save model test metrics for later display.
    try:
        supabase.table("model_runs").insert({
            "coin": coin,
            "timeframe": timeframe,
            "train_rows": int(len(X)),
            "test_accuracy": test_accuracy,
            "baseline_accuracy": baseline_accuracy,
            "accepted": False,
            "model_version": "random-forest-v1",
            "notes": "Chronological 80/20 holdout; paper testing"
        }).execute()
    except Exception as e:
        st.warning(f"Prediction saved, but model metrics were not saved: {e}")

    return {
        "is_new": True,
        "prediction": prediction,
        "up_probability": up_probability,
        "test_accuracy": test_accuracy,
        "baseline_accuracy": baseline_accuracy,
        "chart": df,
        "candle_time": candle_time
    }


# =========================================
# OUTCOME EVALUATION
# =========================================

def resolve_pending(coin=None, timeframe=None, limit=100):
    errors = []
    updated = 0

    try:
        query = (
            supabase.table("predictions")
            .select("*")
            .is_("actual", "null")
            .order("created_at", desc=True)
            .limit(limit)
        )

        if coin is not None:
            query = query.eq("coin", coin)

        if timeframe is not None:
            query = query.eq("timeframe", timeframe)

        response = query.execute()
        pending = response.data or []

    except Exception as e:
        return 0, [f"Could not load pending predictions: {e}"]

    # Reuse candle data for records with the same market pair
    # and timeframe during this evaluation pass.
    data_cache = {}

    for row in pending:
        try:
            key = (row["coin"], row["timeframe"])

            if key not in data_cache:
                data_cache[key] = fetch_data(*key)

            candles = data_cache[key]

            reference_time = pd.to_datetime(
                row["candle_time"], utc=True
            )

            # The first completed candle after the
            # reference candle is used as the outcome.
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
                errors.append(
                    f"{key}: invalid outcome opening price."
                )
                continue

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

        except Exception as e:
            errors.append(
                f"{row.get('coin')} {row.get('timeframe')} "
                f"record {row.get('id')}: {e}"
            )

    return updated, errors


# =========================================
# USER CONTROLS
# =========================================

col1, col2 = st.columns(2)

with col1:
    coin = st.selectbox(
        "Crypto Pair",
        ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD"]
    )

with col2:
    timeframe = st.selectbox(
        "Timeframe",
        ["1m", "5m", "15m", "1h", "4h", "1d"]
    )

st.caption(
    "Auto mode handles the currently selected pair and timeframe. "
    "Keep this page open for periodic checks."
)

evaluate_all_clicked = st.button(
    "🧪 Evaluate all pending outcomes"
)


# =========================================
# AUTOMATIC RUN
# =========================================

with st.spinner("Checking candle outcomes and predictions..."):

    if evaluate_all_clicked:
        resolved_count, outcome_errors = resolve_pending()
    else:
        resolved_count, outcome_errors = resolve_pending(
            coin=coin,
            timeframe=timeframe
        )

    prediction_result = None
    prediction_error = None

    try:
        prediction_result = make_prediction(coin, timeframe)
    except Exception as e:
        prediction_error = str(e)


if resolved_count:
    st.success(f"Updated outcomes: {resolved_count}")

if outcome_errors:
    with st.expander(
        f"Outcome evaluation warnings ({len(outcome_errors)})"
    ):
        for message in outcome_errors[:10]:
            st.warning(message)

if prediction_error:
    st.error(f"Automatic prediction failed: {prediction_error}")

elif prediction_result:
    if prediction_result["is_new"]:
        st.success(
            "New prediction saved to Supabase for the latest completed candle."
        )
    else:
        st.info(
            "This candle already has a saved prediction. "
            "Duplicate prediction skipped."
        )

    a, b, c = st.columns(3)

    a.metric(
        "Next Candle Estimate",
        "UP 🟢"
        if prediction_result["prediction"] == 1
        else "DOWN 🔴"
    )

    up_prob = prediction_result.get("up_probability")

    b.metric(
        "Model UP Probability",
        f"{float(up_prob) * 100:.1f}%"
        if up_prob is not None else "N/A"
    )

    metrics = latest_model_metrics(coin, timeframe)

    test_accuracy = prediction_result.get("test_accuracy")
    if test_accuracy is None:
        test_accuracy = metrics.get("test_accuracy")

    c.metric(
        "Historical Test Accuracy",
        f"{float(test_accuracy) * 100:.1f}%"
        if test_accuracy is not None else "N/A"
    )

    baseline = prediction_result.get("baseline_accuracy")
    if baseline is None:
        baseline = metrics.get("baseline_accuracy")

    if baseline is not None:
        st.write(
            "Baseline accuracy:",
            f"{float(baseline) * 100:.1f}%"
        )

    st.caption(
        "Historical test accuracy and model probabilities "
        "do not guarantee future results."
    )

    st.line_chart(
        prediction_result["chart"].set_index("Time")["Close"]
    )


# =========================================
# PAPER TESTING HISTORY
# =========================================

st.divider()
st.header("🧪 Paper Testing Results")

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
        st.info("No predictions saved yet.")
    else:
        required_columns = [
            "created_at", "coin", "timeframe",
            "close_price", "prediction", "actual",
            "outcome_open", "outcome_close",
            "outcome_high", "outcome_low",
            "body_change_pct", "full_range_pct"
        ]

        for col in required_columns:
            if col not in history.columns:
                history[col] = np.nan

        resolved = history.dropna(
            subset=["actual", "prediction"]
        ).copy()

        if not resolved.empty:
            resolved["Correct"] = (
                resolved["prediction"].astype(int)
                == resolved["actual"].astype(int)
            )

            total = len(resolved)
            correct = int(resolved["Correct"].sum())

            a, b, c = st.columns(3)
            a.metric("Evaluated Predictions", total)
            b.metric("Correct", correct)
            c.metric(
                "Paper Accuracy",
                f"{correct / total * 100:.1f}%"
            )
        else:
            st.info(
                "Outcomes are pending until the next candle closes "
                "and is successfully evaluated."
            )

        history["Prediction"] = history["prediction"].map(
            {1: "UP", 0: "DOWN"}
        )

        history["Outcome"] = history["actual"].map(
            {1: "UP", 0: "DOWN"}
        ).fillna("Pending")

        def movement_label(value):
            if pd.isna(value):
                return "Pending"

            magnitude = abs(float(value))

            if magnitude < 0.1:
                return "Small move"
            elif magnitude < 0.5:
                return "Moderate move"

            return "Large move"

        history["Movement"] = history[
            "body_change_pct"
        ].apply(movement_label)

        def correctness(row):
            if pd.isna(row["actual"]) or pd.isna(row["prediction"]):
                return "Pending"

            return (
                "Yes"
                if int(row["prediction"]) == int(row["actual"])
                else "No"
            )

        history["Correct?"] = history.apply(
            correctness, axis=1
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
            "Outcome",
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
            "Movement labels describe candle body size only. "
            "They do not mean the prediction is safe. "
            "Full Range % measures the high-low range relative "
            "to the next candle's open."
        )

except Exception as e:
    st.error(f"Could not load Supabase history: {e}")
