import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import numpy_financial as npf
from datetime import datetime

st.set_page_config(page_title="Portfolio Simulator", layout="wide")

# =========================
# 기본값
# =========================
DEFAULT_TICKERS = ["EWY","SPY","SCHD","QQQ","IEF","TLT","IAU","SLV","IGF","RWO"]
DEFAULT_WEIGHTS = [0.1,0.075,0.1,0.075,0.1,0.1,0.125,0.125,0.1,0.1]

# =========================
# Sidebar
# =========================
st.sidebar.header("⚙️ Settings")

st.sidebar.markdown("""
### 📌 티커 입력 방법
- 쉼표(,)로 구분해서 입력  
- 예: `SPY, QQQ, TLT`  
- 미국 ETF/주식: 그대로 입력  
- 한국 ETF: 일부는 지원 안 될 수 있음  

👉 데이터는 Yahoo Finance 기준
""")

strategy = st.sidebar.selectbox(
    "Strategy",
    ["Rebalance", "DCA"],  # 👉 디폴트 Rebalance
    index=0
)

tickers_input = st.sidebar.text_input("Tickers", ",".join(DEFAULT_TICKERS))
weights_input = st.sidebar.text_input("Weights", ",".join(map(str, DEFAULT_WEIGHTS)))

st.sidebar.markdown("👉 비중은 합계가 1이 아니어도 자동 정규화됨")

# 만원 단위
monthly_cash_input = st.sidebar.number_input("납입금액 (만원)", value=300)
monthly_cash = monthly_cash_input * 10000

start_date = st.sidebar.date_input("Start", datetime(2015,1,1))
end_date = st.sidebar.date_input("End", datetime.today())

freq = st.sidebar.selectbox(
    "납입 주기",
    ["Monthly","Quarterly","Semi-Annual","Annual"]
)

st.sidebar.markdown("""
👉 납입 주기:
- Monthly: 매월 투자
- Quarterly: 3개월마다 투자
""")

bank_rate = st.sidebar.number_input("기준 금리 (%)", value=3.0) / 100

run = st.sidebar.button("🚀 Run")

freq_map = {
    "Monthly":"ME",
    "Quarterly":"QE",
    "Semi-Annual":"2QE",
    "Annual":"YE"
}

# =========================
# 설명
# =========================
st.title("💰 투자 시뮬레이터")

st.markdown("""
정기적으로 투자했을 때 **내 돈이 어떻게 변하는지** 보여주는 도구입니다.

### 📊 핵심 지표
- **IRR**: 내가 실제로 번 연 수익률  
- **MDD**: 가장 크게 손실 난 순간  

👉 IRR은 높을수록 좋지만  
👉 MDD가 크면 실제 체감은 매우 힘들 수 있음  
""")

# =========================
# 데이터
# =========================
@st.cache_data
def load_data(tickers, start, end):
    return yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"].dropna()

# =========================
# 시뮬레이션
# =========================
def simulate(data, weights, cash, freq, strategy):

    prices = data.resample("ME").last()
    invest_dates = prices.resample(freq).last().index

    holdings = np.zeros(len(weights))
    port, cf = [], []

    for i in range(len(prices)):
        p = prices.iloc[i]
        date = prices.index[i]

        if date in invest_dates:
            value = np.sum(holdings * p)
            total = value + cash

            if strategy == "Rebalance":
                target = total * weights
                buy = (target - holdings*p)/p
            else:
                buy = (cash * weights)/p

            holdings += buy
            cf.append(-cash)
        else:
            cf.append(0)

        port.append(np.sum(holdings*p))

    cf[-1] += port[-1]
    return pd.Series(port, index=prices.index), cf

# =========================
# Metrics
# =========================
def irr(cf): return npf.irr(cf)*12
def mdd(port): return (port/port.cummax()-1).min()

# 👉 연도별 (현금 제거)
def yearly_returns(port, cash, freq):

    df = pd.DataFrame({"v":port})
    df["y"] = df.index.year

    results = []

    for y in df["y"].unique():
        temp = df[df["y"]==y]["v"]
        if len(temp) < 2:
            continue

        r=[]
        for i in range(1,len(temp)):
            rr=(temp.iloc[i]-temp.iloc[i-1]-cash)/temp.iloc[i-1]
            r.append(rr)

        results.append((y, np.prod([1+x for x in r])-1))

    return pd.DataFrame(results, columns=["Year","Return"]).set_index("Year")

# =========================
# Money
# =========================
def money(port, cf, rate):
    total = -sum([x for x in cf if x<0])
    final = port.iloc[-1]
    profit = final - total

    n = len(cf)
    fv_bank = npf.fv(rate/12, n, total/n*-1, 0)

    return total, final, profit, fv_bank

# =========================
# RUN
# =========================
if run:

    tickers=[x.strip().upper() for x in tickers_input.split(",")]
    weights=np.array([float(x) for x in weights_input.split(",")])
    weights=weights/weights.sum()

    data=load_data(tickers,start_date,end_date)

    if data.empty:
        st.error("데이터 없음 (티커 확인 필요)")
        st.stop()

    port, cf = simulate(data, weights, monthly_cash, freq_map[freq], strategy)

    irr_v = irr(cf)
    mdd_v = mdd(port)

    total, final, profit, bank = money(port, cf, bank_rate)

    # ================= KPI
    st.markdown("## 📊 결과 요약")

    c1,c2 = st.columns(2)
    c1.metric("IRR", f"{irr_v:.2%}")
    c2.metric("MDD", f"{mdd_v:.2%}")

    # ================= Money
    st.markdown("## 💰 투자 결과")

    c3,c4,c5,c6 = st.columns(4)
    c3.metric("총 투자금", f"{total:,.0f}원")
    c4.metric("최종 자산", f"{final:,.0f}원")
    c5.metric("총 수익", f"{profit:,.0f}원")
    c6.metric("은행 대비 초과수익", f"{final-bank:,.0f}원")

    # ================= Growth
    st.markdown("## 📈 자산 성장")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=port.index, y=port, name="Portfolio"))
    fig.update_layout(template="plotly_dark", height=500)
    st.plotly_chart(fig, use_container_width=True)

    # ================= Drawdown
    st.markdown("## 📉 손실 구간")

    dd = port/port.cummax()-1
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=dd.index, y=dd, name="Drawdown"))
    fig2.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig2, use_container_width=True)

    # ================= Yearly
    st.markdown("## 📅 연도별 수익률")

    yr = yearly_returns(port, monthly_cash, freq_map[freq])
    st.dataframe(yr.style.format("{:.2%}"))

# ================= 안내
st.markdown("""
---
💡 **활용 팁**

- IRR만 보지 말고 MDD도 함께 확인하세요  
- 납입 주기를 바꿔보면 투자 타이밍 효과를 볼 수 있습니다  
- 은행 금리를 올려보면 “투자할 가치가 있는지” 판단할 수 있습니다  

---

⚠️ 본 결과는 과거 데이터 기반이며 투자 판단 책임은 본인에게 있습니다.
""")
