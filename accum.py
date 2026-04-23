import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import numpy_financial as npf
from datetime import datetime

st.set_page_config(page_title="Accumulation Backtest", layout="wide")

st.title("💰 Accumulation Backtest (Rebal vs DCA)")

# =========================
# 세션 상태
# =========================
if "tickers" not in st.session_state:
    st.session_state.tickers = ["AAPL", "MSFT"]

if "weights" not in st.session_state:
    st.session_state.weights = [0.5, 0.5]

# =========================
# 입력
# =========================
st.sidebar.header("⚙️ Settings")

tickers_input = st.sidebar.text_input(
    "Tickers",
    ",".join(st.session_state.tickers)
)

weights_input = st.sidebar.text_input(
    "Weights",
    ",".join(map(str, st.session_state.weights))
)

monthly_cash = st.sidebar.number_input("Monthly Investment", value=1000)

start_date = st.sidebar.date_input("Start", datetime(2015,1,1))
end_date = st.sidebar.date_input("End", datetime.today())

run = st.sidebar.button("🚀 Run")

def yearly_twr(port, monthly_cash):
    df = pd.DataFrame({"value": port})
    df["year"] = df.index.year

    results = []

    for y in df["year"].unique():
        temp = df[df["year"] == y]["value"]

        if len(temp) < 2:
            continue

        returns = []
        for i in range(1, len(temp)):
            prev = temp.iloc[i-1]
            curr = temp.iloc[i]
            r = (curr - prev - monthly_cash) / prev
            returns.append(r)

        twr = np.prod([1 + r for r in returns]) - 1
        results.append((y, twr))

    return pd.DataFrame(results, columns=["Year", "Return"]).set_index("Year")

# =========================
# 데이터 로드
# =========================
@st.cache_data
def load_data(tickers, start, end):
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True)

    if raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        data = raw["Close"]
    else:
        data = raw[["Close"]]
        data.columns = tickers

    return data.dropna()

# =========================
# 시뮬레이션 (리밸런싱)
# =========================
def simulate_rebalance(data, weights, monthly_cash):
    prices = data.resample("ME").last()
    weights = np.array(weights)

    holdings = np.zeros(len(weights))
    port_values = []
    cashflows = []

    for i in range(len(prices)):
        p = prices.iloc[i]

        current_value = np.sum(holdings * p)
        total = current_value + monthly_cash

        target = total * weights
        diff = target - holdings * p
        buy = diff / p

        holdings += buy
        port_value = np.sum(holdings * p)

        port_values.append(port_value)
        cashflows.append(-monthly_cash)

    cashflows[-1] += port_values[-1]

    return pd.Series(port_values, index=prices.index), cashflows

# =========================
# 시뮬레이션 (DCA)
# =========================
def simulate_dca(data, weights, monthly_cash):
    prices = data.resample("ME").last()
    weights = np.array(weights)

    holdings = np.zeros(len(weights))
    port_values = []
    cashflows = []

    for i in range(len(prices)):
        p = prices.iloc[i]

        buy = (monthly_cash * weights) / p
        holdings += buy

        port_value = np.sum(holdings * p)

        port_values.append(port_value)
        cashflows.append(-monthly_cash)

    cashflows[-1] += port_values[-1]

    return pd.Series(port_values, index=prices.index), cashflows

# =========================
# IRR
# =========================
def calc_irr(cashflows):
    irr = npf.irr(cashflows)
    return irr * 12

# =========================
# TWR
# =========================
def calc_twr(port, monthly_cash):
    returns = []

    for i in range(1, len(port)):
        prev = port.iloc[i-1]
        curr = port.iloc[i]

        r = (curr - prev - monthly_cash) / prev
        returns.append(r)

    twr = np.prod([1 + r for r in returns]) - 1
    annual = (1 + twr) ** (12/len(returns)) - 1

    return annual

# =========================
# MDD
# =========================
def calc_mdd_twr(port, monthly_cash):
    returns = []

    for i in range(1, len(port)):
        prev = port.iloc[i-1]
        curr = port.iloc[i]

        # 현금 유입 제거
        r = (curr - prev - monthly_cash) / prev
        returns.append(r)

    cum = np.cumprod([1 + r for r in returns])
    cum = pd.Series(cum)

    dd = cum / cum.cummax() - 1
    return dd.min()

def drawdown_series(port, monthly_cash):
    returns = []

    for i in range(1, len(port)):
        prev = port.iloc[i-1]
        curr = port.iloc[i]
        r = (curr - prev - monthly_cash) / prev
        returns.append(r)

    cum = pd.Series(np.cumprod([1 + r for r in returns]), index=port.index[1:])
    dd = cum / cum.cummax() - 1
    return dd

# =========================
# 실행
# =========================
if run:

    tickers = [t.strip().upper() for t in tickers_input.split(",")]
    weights = np.array([float(w) for w in weights_input.split(",")])
    weights = weights / weights.sum()

    st.session_state.tickers = tickers
    st.session_state.weights = weights.tolist()

    data = load_data(tickers, start_date, end_date)

    if data.empty:
        st.error("데이터 없음")
        st.stop()

    rebal, cf_r = simulate_rebalance(data, weights, monthly_cash)
    dca, cf_d = simulate_dca(data, weights, monthly_cash)

    # =========================
    # 지표 계산
    # =========================
    irr_r = calc_irr(cf_r)
    irr_d = calc_irr(cf_d)

    twr_r = calc_twr(rebal, monthly_cash)
    twr_d = calc_twr(dca, monthly_cash)

    mdd_r = calc_mdd_twr(rebal, monthly_cash)
    mdd_d = calc_mdd_twr(dca, monthly_cash)

    # =========================
    # KPI
    # =========================
    st.markdown("## 📊 Performance")

    col1, col2, col3 = st.columns(3)

    col1.metric("Rebal IRR", f"{irr_r:.2%}")
    col2.metric("Rebal TWR", f"{twr_r:.2%}")
    col3.metric("Rebal MDD", f"{mdd_r:.2%}")

    col4, col5, col6 = st.columns(3)

    col4.metric("DCA IRR", f"{irr_d:.2%}")
    col5.metric("DCA TWR", f"{twr_d:.2%}")
    col6.metric("DCA MDD", f"{mdd_d:.2%}")

    # =========================
    # 그래프
    # =========================
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=rebal.index, y=rebal, name="Rebalance Accum"))
    fig.add_trace(go.Scatter(x=dca.index, y=dca, name="DCA"))

    fig.update_layout(
        template="plotly_dark",
        height=600,
        title="📈 Accumulation Comparison",
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

    # =========================
    # 연도별 성과
    # =========================
    st.markdown("## 📅 Yearly Returns (TWR)")

    yr_rebal = yearly_twr(rebal, monthly_cash).rename(columns={"Return": "Rebal"})
    yr_dca = yearly_twr(dca, monthly_cash).rename(columns={"Return": "DCA"})

    yr = pd.concat([yr_rebal, yr_dca], axis=1)

    st.dataframe(yr.style.format("{:.2%}"))

    st.markdown("## 📉 Drawdown (TWR)")

    dd_r = drawdown_series(rebal, monthly_cash)
    dd_d = drawdown_series(dca, monthly_cash)

    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=dd_r.index, y=dd_r, name="Rebal DD"))
    fig_dd.add_trace(go.Scatter(x=dd_d.index, y=dd_d, name="DCA DD"))

    fig_dd.update_layout(template="plotly_dark", height=400)

    st.plotly_chart(fig_dd, use_container_width=True)
