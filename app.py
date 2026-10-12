
import streamlit as st
import pandas as pd
import ccxt

st.set_page_config(
    page_title="Crypto Candle AI",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Crypto Candle Prediction AI")
st.caption("Educational dashboard • Paper testing only")

exchange = ccxt.binance()

coin = st.selectbox(
    "Select Crypto Pair",
    ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]
)

timeframe = st.selectbox(
    "Select Candle Timeframe",
    ["1m", "5m", "15m", "1h", "4h", "1d"]
)

if st.button("🔄 Analyze Market"):
    try:
        candles = exchange.fetch_ohlcv(
            coin, timeframe=timeframe, limit=100
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

        df["Previous Close"] = df["Close"].shift(1)
        df["Return"] = df["Close"].pct_change()

        df = df.dropna()

        last = df.iloc[-1]
        previous = df.iloc[-2]

        direction = (
            "UP 🟢"
            if last["Close"] > last["Open"]
            else "DOWN 🔴"
        )

        st.subheader(f"{coin} • {timeframe}")
        st.metric("Latest Candle Direction", direction)
        st.metric("Latest Close Price", f'{last["Close"]:.6f}')

        st.info(
            "This is the direction of the latest candle, "
            "NOT an AI prediction of the next candle."
        )

        st.line_chart(df.set_index("Time")["Close"])

        st.subheader("Recent Market Data")
        st.dataframe(df.tail(10), use_container_width=True)

    except Exception as e:
        st.error(f"Could not fetch market data: {e}")
