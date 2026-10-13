import streamlit as st
import pandas as pd
import numpy as np
import ccxt
from datetime import datetime, timezone
from supabase import create_client
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Crypto Candle AI", page_icon="📈", layout="wide")
st.title("📈 Crypto Candle AI")
st.caption("Educational paper testing only. Predictions are uncertain; this is not a trading signal service.")

PAIRS = ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD"]
TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"]
DURATION = {
    "1m": pd.Timedelta(minutes=1), "5m": pd.Timedelta(minutes=5),
    "15m": pd.Timedelta(minutes=15), "1h": pd.Timedelta(hours=1),
    "4h": pd.Timedelta(hours=4), "1d": pd.Timedelta(days=1),
}

@st.cache_resource
def get_supabase():
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])

@st.cache_resource
def get_exchange():
    return ccxt.kraken({"enableRateLimit": True, "timeout": 20000})

try:
    supabase = get_supabase()
except Exception as e:
    st.error(f"Supabase connection failed: {e}")
    st.stop()

@st.cache_data(ttl=15, show_spinner=False)
def fetch_data(coin, timeframe):
    candles = get_exchange().fetch_ohlcv(coin, timeframe=timeframe, limit=1000)
    if not candles or len(candles) < 100:
        raise ValueError(f"Not enough candle data for {coin} {timeframe}.")
    df = pd.DataFrame(candles, columns=["Time", "Open", "High", "Low", "Close", "Volume"])
    df["Time"] = pd.to_datetime(df["Time"], unit="ms", utc=True)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Time", "Open", "High", "Low", "Close"])
    df = df.sort_values("Time").drop_duplicates("Time", keep="last").reset_index(drop=True)
    # Exclude the possibly unfinished latest candle.
    df = df.iloc[:-1].copy().reset_index(drop=True)
    if len(df) < 100:
        raise ValueError("Not enough completed candles.")
    return df

def features_for(df):
    f = pd.DataFrame(index=df.index)
    close = df["Close"]
    f["return_1"] = close.pct_change()
    f["return_3"] = close.pct_change(3)
    f["return_5"] = close.pct_change(5)
    f["body"] = (df["Close"] - df["Open"]) / df["Open"]
    f["range"] = (df["High"] - df["Low"]) / df["Open"]
    for n in [5, 10, 20]:
        f[f"sma_{n}"] = close / close.rolling(n).mean() - 1
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

def build_model():
    return RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1
    )

def latest_model_metrics(coin, timeframe):
    try:
        r = (supabase.table("model_runs").select("test_accuracy,baseline_accuracy")
             .eq("coin", coin).eq("timeframe", timeframe)
             .order("trained_at", desc=True).limit(1).execute())
        return r.data[0] if r.data else {}
    except Exception:
        return {}

def make_prediction(coin, timeframe):
    df = fetch_data(coin, timeframe)
    f = features_for(df)
    usable = f.dropna()
    if usable.empty:
        raise ValueError("Could not calculate latest candle features.")

    idx = usable.index[-1]
    latest = usable.iloc[[-1]]
    reference = df.loc[idx]
    ref_time = pd.Timestamp(reference["Time"]).tz_convert("UTC")
    target_time = ref_time + DURATION[timeframe]
    candle_time = ref_time.isoformat()
    target_candle_time = target_time.isoformat()
    close_price = float(reference["Close"])

    existing = (supabase.table("predictions")
                .select("id,prediction,up_probability,target_candle_time")
                .eq("coin", coin).eq("timeframe", timeframe)
                .eq("candle_time", candle_time).limit(1).execute())
    if existing.data:
        saved = existing.data[0]
        return {
            "is_new": False, "prediction": int(saved["prediction"]),
            "up_probability": saved.get("up_probability"), "chart": df,
            "candle_time": candle_time,
            "target_candle_time": saved.get("target_candle_time") or target_candle_time,
            "reference_close": close_price, "test_accuracy": None,
            "baseline_accuracy": None,
        }

    # Label row t with the direction of candle t+1. The latest row is unlabeled.
    target = (df["Close"].shift(-1) > df["Open"].shift(-1)).astype(float)
    target.iloc[-1] = np.nan
    training = f.copy()
    training["target"] = target
    training = training.dropna()
    if len(training) < 150:
        raise ValueError("Not enough labelled data for model training.")

    X = training.drop(columns=["target"])
    y = training["target"].astype(int)
    split = int(len(training) * 0.8)
    if split < 100 or len(training) - split < 20:
        raise ValueError("Not enough rows for chronological holdout testing.")

    test_model = build_model()
    test_model.fit(X.iloc[:split], y.iloc[:split])
    test_pred = test_model.predict(X.iloc[split:])
    test_accuracy = float(accuracy_score(y.iloc[split:], test_pred))
    baseline_label = int(y.iloc[:split].mode().iloc[0])
    baseline_accuracy = float(accuracy_score(
        y.iloc[split:], np.full(len(y.iloc[split:]), baseline_label)
    ))

    model = build_model()
    model.fit(X, y)
    prediction = int(model.predict(latest)[0])
    probs = model.predict_proba(latest)[0]
    classes = list(model.classes_)
    up_probability = float(probs[classes.index(1)]) if 1 in classes else 0.0

    payload = {
        "coin": coin, "timeframe": timeframe, "candle_time": candle_time,
        "target_candle_time": target_candle_time, "close_price": close_price,
        "prediction": prediction, "up_probability": up_probability,
    }
    # Insert only; don't overwrite an existing prediction.
    supabase.table("predictions").insert(payload).execute()
    try:
        supabase.table("model_runs").insert({
            "coin": coin, "timeframe": timeframe, "train_rows": int(len(X)),
            "test_accuracy": test_accuracy, "baseline_accuracy": baseline_accuracy,
            "accepted": False, "model_version": "random-forest-v2",
            "notes": "Chronological 80/20 holdout; next candle direction; paper testing."
        }).execute()
    except Exception as e:
        st.warning(f"Prediction saved, but model metrics were not saved: {e}")

    return {
        "is_new": True, "prediction": prediction,
        "up_probability": up_probability, "chart": df,
        "candle_time": candle_time, "target_candle_time": target_candle_time,
        "reference_close": close_price, "test_accuracy": test_accuracy,
        "baseline_accuracy": baseline_accuracy,
    }

def resolve_pending(coin=None, timeframe=None, limit=200):
    errors, updated = [], 0
    try:
        q = (supabase.table("predictions").select("*").is_("actual", "null")
             .order("created_at", desc=False).limit(limit))
        if coin is not None:
            q = q.eq("coin", coin)
        if timeframe is not None:
            q = q.eq("timeframe", timeframe)
        pending = q.execute().data or []
    except Exception as e:
        return 0, [f"Could not load pending predictions: {e}"]

    candle_cache = {}
    for row in pending:
        try:
            pair, tf = row["coin"], row["timeframe"]
            if tf not in DURATION:
                errors.append(f"{pair} {tf} record {row.get('id')}: unsupported timeframe")
                continue
            key = (pair, tf)
            if key not in candle_cache:
                candle_cache[key] = fetch_data(pair, tf)
            candles = candle_cache[key]

            ref_time = pd.to_datetime(row["candle_time"], utc=True)
            raw_target = row.get("target_candle_time")
            target_time = (pd.to_datetime(raw_target, utc=True)
                           if raw_target else ref_time + DURATION[tf])

            # Exact match only: never score a later candle by mistake.
            match = candles[candles["Time"] == target_time]
            if match.empty:
                continue  # target candle is not yet available as completed data

            candle = match.iloc[0]
            op, cl = float(candle["Open"]), float(candle["Close"])
            hi, lo = float(candle["High"]), float(candle["Low"])
            if not np.isfinite(op) or op <= 0:
                errors.append(f"{pair} {tf} record {row.get('id')}: invalid target open")
                continue

            update_data = {
                "target_candle_time": target_time.isoformat(),
                "actual": int(cl > op),
                "outcome_open": op, "outcome_close": cl,
                "outcome_high": hi, "outcome_low": lo,
                "body_change_pct": (cl - op) / op * 100,
                "full_range_pct": (hi - lo) / op * 100,
                "evaluated_at": datetime.now(timezone.utc).isoformat(),
            }
            result = (supabase.table("predictions").update(update_data)
                      .eq("id", row["id"]).is_("actual", "null").execute())
            if result.data:
                updated += 1
        except Exception as e:
            errors.append(f"{row.get('coin')} {row.get('timeframe')} record {row.get('id')}: {e}")
    return updated, errors

left, right = st.columns(2)
with left:
    selected_coin = st.selectbox("Crypto pair", PAIRS, key="selected_coin")
with right:
    selected_timeframe = st.selectbox("Timeframe", TIMEFRAMES, key="selected_timeframe")

st.caption("The live panels update every 30 seconds while the app is open. They do not run in the background after the app is closed.")

if st.button("Evaluate all pairs' pending outcomes"):
    with st.spinner("Checking pending outcomes..."):
        count, errors = resolve_pending()
    st.success(f"Updated outcomes: {count}")
    if errors:
        with st.expander(f"Warnings ({len(errors)})"):
            for msg in errors[:20]:
                st.warning(msg)

@st.fragment(run_every="30s")
def live_panel(coin, timeframe):
    st.subheader("Live paper-testing panel")
    st.caption("Last panel check (UTC): " + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))
    with st.spinner("Checking completed candles..."):
        resolved_count, outcome_errors = resolve_pending(coin, timeframe)
        prediction_result, prediction_error = None, None
        try:
            prediction_result = make_prediction(coin, timeframe)
        except Exception as e:
            prediction_error = str(e)

    if resolved_count:
        st.success(f"Newly evaluated outcomes: {resolved_count}")
    if outcome_errors:
        with st.expander(f"Outcome warnings ({len(outcome_errors)})"):
            for msg in outcome_errors[:10]:
                st.warning(msg)
    if prediction_error:
        st.error(f"Prediction check failed: {prediction_error}")
        return
    if not prediction_result:
        st.info("No prediction available yet.")
        return

    if prediction_result["is_new"]:
        st.success("New paper prediction saved.")
    else:
        st.info("This reference candle already has a saved prediction.")

    a, b, c = st.columns(3)
    a.metric("Next candle estimate", "UP" if prediction_result["prediction"] == 1 else "DOWN")
    p = prediction_result.get("up_probability")
    b.metric("Model UP probability", f"{float(p)*100:.1f}%" if p is not None else "N/A")

    metrics = latest_model_metrics(coin, timeframe)
    acc = prediction_result.get("test_accuracy")
    if acc is None:
        acc = metrics.get("test_accuracy")
    c.metric("Historical holdout accuracy", f"{float(acc)*100:.1f}%" if acc is not None else "N/A")

    baseline = prediction_result.get("baseline_accuracy")
    if baseline is None:
        baseline = metrics.get("baseline_accuracy")
    if baseline is not None:
        st.caption(f"Baseline holdout accuracy: {float(baseline)*100:.1f}%")

    st.write(f"Reference candle time (UTC): `{prediction_result['candle_time']}`")
    st.write(f"Target candle time (UTC): `{prediction_result['target_candle_time']}`")
    st.write(f"Reference close: `{prediction_result['reference_close']:.8g}`")
    st.line_chart(prediction_result["chart"].set_index("Time")["Close"])
    st.caption("Chart data comes from Kraken; another platform may use a different price feed.")

live_panel(selected_coin, selected_timeframe)

st.divider()
st.header("Paper testing history")

@st.fragment(run_every="20s")
def history_panel():
    try:
        response = (supabase.table("predictions").select("*")
                    .order("created_at", desc=True).limit(500).execute())
        history = pd.DataFrame(response.data or [])
        if history.empty:
            st.info("No predictions saved yet.")
            return

        needed = [
            "created_at", "candle_time", "target_candle_time", "coin", "timeframe",
            "close_price", "prediction", "actual", "up_probability",
            "outcome_open", "outcome_close", "outcome_high", "outcome_low",
            "body_change_pct", "full_range_pct",
        ]
        for col in needed:
            if col not in history.columns:
                history[col] = np.nan

        resolved = history.dropna(subset=["actual", "prediction"]).copy()
        if not resolved.empty:
            correct_mask = resolved["actual"].astype(int) == resolved["prediction"].astype(int)
            total, correct = len(resolved), int(correct_mask.sum())
            a, b, c = st.columns(3)
            a.metric("Evaluated predictions", total)
            b.metric("Correct", correct)
            c.metric("Overall paper accuracy", f"{correct/total*100:.1f}%")
        else:
            st.info("Outcomes are pending until their target candles are completed.")

        history["Prediction"] = history["prediction"].map({1: "UP", 0: "DOWN"})
        history["Outcome"] = history["actual"].map({1: "UP", 0: "DOWN"}).fillna("Pending")
        history["Correct?"] = "Pending"
        mask = history["actual"].notna() & history["prediction"].notna()
        history.loc[mask, "Correct?"] = np.where(
            history.loc[mask, "actual"].astype(int) == history.loc[mask, "prediction"].astype(int),
            "Yes", "No"
        )
        history["UP Probability %"] = pd.to_numeric(history["up_probability"], errors="coerce") * 100

        display = history[[
            "created_at", "candle_time", "target_candle_time", "coin", "timeframe",
            "close_price", "Prediction", "UP Probability %", "outcome_open",
            "outcome_close", "outcome_high", "outcome_low", "body_change_pct",
            "full_range_pct", "Outcome", "Correct?"
        ]].rename(columns={
            "created_at": "Created At (UTC)",
            "candle_time": "Reference Candle (UTC)",
            "target_candle_time": "Target Candle (UTC)",
            "coin": "Pair", "timeframe": "Timeframe", "close_price": "Reference Close",
            "outcome_open": "Target Open", "outcome_close": "Target Close",
            "outcome_high": "Target High", "outcome_low": "Target Low",
            "body_change_pct": "Body Change %", "full_range_pct": "Target Full Range %",
        })
        st.dataframe(display, use_container_width=True, hide_index=True)
        st.caption("Outcome label means target candle Close > target candle Open. Historical accuracy does not guarantee future results.")
    except Exception as e:
        st.error(f"Could not load history: {e}")

history_panel()
