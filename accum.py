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
- 쉼표(,)로 구분  
- 예: `SPY, QQQ, TLT`  

### 📊 주의
- Yahoo Finance 기준 티커 사용  
- 한국 ETF 일부는 조회 안 될 수 있음  
""")

strategy = st.sidebar.selectbox("Strategy", ["Rebalance", "DCA"], index=0)

tickers_input = st.sidebar.text_input("Tickers", ",".join(DEFAULT_TICKERS))
weights_input = st.sidebar.text_input("Weights", ",".join(map(str, DEFAULT_WEIGHTS)))

st.sidebar.markdown("👉 비중은 자동으로 합계 1로 맞춰집니다")

monthly_cash_input = st.sidebar.number_input("납입금액 (만원)", value=300)
monthly_cash = monthly_cash_input * 10000

st.sidebar.markdown("👉 예: 300 입력 = 매달 300만원 투자")

start_date = st.sidebar.date_input("Start", datetime(2015,1,1))
end_date = st.sidebar.date_input("End", datetime.today())

freq = st.sidebar.selectbox(
    "납입 주기",
    ["Monthly","Quarterly","Semi-Annual","Annual"]
)

st.sidebar.markdown("""
👉 납입 주기 설명  
- Monthly: 매월 투자  
- Quarterly: 3개월마다  
- Annual: 1년에 한 번  
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
# 상단 설명
# =========================
st.title("💰 투자 시뮬레이터")

st.markdown("""
이 도구는 **정기 투자(적립식)**를 했을 때  
👉 *내 돈이 어떻게 불어나는지* 보여줍니다.

---

## 📊 핵심 지표 설명

### ✔ IRR (연 수익률)
👉 내가 실제로 번 수익률  
👉 높을수록 좋음

### ✔ MDD (최대 낙폭)
👉 가장 크게 손실 난 순간  
👉 클수록 (절대값) 위험

---

## 💡 어떻게 해석하면 좋냐

- IRR이 높아도 MDD가 크면  
  → 실제로 버티기 어려운 전략  

- MDD가 낮으면  
  → 심리적으로 안정적인 투자  

👉 **둘을 같이 보는 게 핵심**
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
    port, cf, invest_flag = [], [], []

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
            invest_flag.append(1)
        else:
            cf.append(0)
            invest_flag.append(0)

        port.append(np.sum(holdings*p))

    cf[-1] += port[-1]

    return pd.Series(port, index=prices.index), cf, np.array(invest_flag)

# =========================
# Metrics
# =========================
def irr(cf): return npf.irr(cf)*12

def twr_returns(port, cash, invest_flag):
    r=[]
    for i in range(1,len(port)):
        prev = port.iloc[i-1]
        curr = port.iloc[i]
        adj_cash = cash if invest_flag[i]==1 else 0
        rr = (curr - prev - adj_cash)/prev
        r.append(rr)
    return np.array(r)

def mdd(port, cash, invest_flag):
    r = twr_returns(port, cash, invest_flag)
    cum = np.cumprod(1+r)
    dd = cum/np.maximum.accumulate(cum) - 1
    return dd.min()

# =========================
# Yearly
# =========================
def yearly_returns(port, cash, invest_flag):

    df = pd.DataFrame({"v":port,"flag":invest_flag})
    df["y"] = df.index.year

    out=[]

    for y in df["y"].unique():
        temp=df[df["y"]==y]
        if len(temp)<2: continue

        r=[]
        for i in range(1,len(temp)):
            prev=temp["v"].iloc[i-1]
            curr=temp["v"].iloc[i]
            adj=cash if temp["flag"].iloc[i]==1 else 0
            r.append((curr-prev-adj)/prev)

        out.append((y,np.prod([1+x for x in r])-1))

    return pd.DataFrame(out,columns=["Year","Return"]).set_index("Year")

# =========================
# Money
# =========================
def money(port, cf, rate):
    total = -sum([x for x in cf if x<0])
    final = port.iloc[-1]
    profit = final - total

    n=len(cf)
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
        st.error("❌ 데이터 없음 → 티커 확인 필요")
        st.stop()

    port, cf, flag = simulate(data, weights, monthly_cash, freq_map[freq], strategy)

    irr_v = irr(cf)
    mdd_v = mdd(port, monthly_cash, flag)

    total, final, profit, bank = money(port, cf, bank_rate)

    # ================= KPI
    st.markdown("## 📊 결과 요약")

    c1,c2 = st.columns(2)
    c1.metric("IRR (연 수익률)", f"{irr_v:.2%}")
    c2.metric("MDD (최대 손실)", f"{mdd_v:.2%}")

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
    st.plotly_chart(fig, use_container_width=True)

    # ================= Drawdown
    st.markdown("## 📉 손실 구간 (투자 기준)")

    dd = port/port.cummax()-1
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=dd.index, y=dd))
    st.plotly_chart(fig2, use_container_width=True)

    # ================= Yearly
    st.markdown("## 📅 연도별 수익률")

    yr = yearly_returns(port, monthly_cash, flag)
    st.dataframe(yr.style.format("{:.2%}"))

# ================= 안내
st.markdown("""
---
💡 **활용 팁**

- IRR이 높아도 MDD가 크면 실제로 버티기 어려움  
- 납입 주기를 바꾸면 시장 타이밍 효과 확인 가능  
- 금리를 올리면 “굳이 투자할 필요 있었나?” 판단 가능  

---

⚠️ 본 결과는 과거 데이터 기반이며 투자 판단 책임은 본인에게 있습니다.
""")
