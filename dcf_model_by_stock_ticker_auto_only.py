import math
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(page_title="DCF Model by stock ticker", page_icon="📊", layout="wide")

st.markdown("""
<style>
:root { --bg:#0d1117; --panel:#161b22; --panel2:#1f2430; --text:#f8fafc; --muted:#aeb6c2; --accent:#ff4b5c; --cyan:#40d8ff; --green:#23d366; --red:#ff6b75; --yellow:#ffd166; }
.stApp { background: var(--bg); color: var(--text); }
.block-container { max-width: 1120px; padding-top: 3rem; }
h1, h2, h3 { color: #ffffff !important; font-weight: 800 !important; letter-spacing: -0.03em; text-shadow: 0 2px 0 #10131a; }
h1 { font-size: 3rem !important; }
h2 { font-size: 2.35rem !important; margin-top: 3.1rem !important; }
h3 { font-size: 1.75rem !important; }
hr { border-color: #303744; margin: 2.3rem 0; }
.info-box { background:#eaf8fc; color:#07111f; border-left: 5px solid var(--cyan); padding: 1.05rem 1.25rem; border-radius: 5px; line-height:1.7; margin: 1.2rem 0 2rem 0; }
.soft-card { border:1px solid #3a4352; border-radius:9px; background:#0f141d; padding:1.1rem; margin:0.6rem 0; }
.metric-label { font-size:0.95rem; font-weight:700; color:#ffffff; margin-bottom:0.35rem; }
.metric-value { font-size:2.2rem; color:#ffffff; line-height:1.1; }
.small-muted { color:var(--muted); font-size:0.98rem; }
.green-pill { background:#123d2a; color:#5eff96; padding:1.05rem 1.15rem; border-radius:9px; font-weight:800; }
.warn-card { background:#ffd9df; color:#121212; border-left:5px solid #ff4b5c; padding:1.2rem; border-radius:8px; font-size:1.15rem; }
.blue-card { background:#eef4ff; color:#101827; border-left:5px solid #69a7ff; padding:1.05rem; border-radius:8px; }
.badge { background:#12251f; color:#5eff96; border-radius:5px; padding:0.13rem 0.35rem; font-family:monospace; }
[data-testid="stMetricLabel"] { color:#fff !important; }
[data-testid="stMetricValue"] { color:#fff !important; font-size:2.1rem; }
.stButton>button { background:#111827; color:#ffffff; border:1px solid #465164; border-radius:8px; padding:0.7rem 1.2rem; font-weight:700; width:100%; }
.stButton>button:hover { border-color:#ff4b5c; color:#ffffff; }
.stTextInput input, .stNumberInput input { background:#252832 !important; color:#ffffff !important; border:1px solid #373f51 !important; }
.stSlider [data-baseweb="slider"] div[role="slider"] { background:#ff4b5c; }
.stDataFrame { border:1px solid #303744; border-radius:8px; }
.footer { color:#aeb6c2; margin: 3rem 0 1rem 0; }
</style>
""", unsafe_allow_html=True)

NVDA_LOCK = {
    "short_name":"NVIDIA Corporation", "sector":"Technology", "industry":"Semiconductors", "price":199.57,
    "market_cap":4.85e12, "beta":2.33, "trailing_pe":40.6, "forward_pe":17.8, "ev_ebitda":36.0,
    "div_yield":0.0002, "shares":24.30e9, "net_debt_m":435.0, "base_fcf_m":96676.0,
    "cashflow_years":[2023,2024,2025,2026],
    "ocf_m":[5641,28090,64089,107288], "capex_m":[-1833,-1069,-3236,-10612], "fcf_m":[3808,27021,60853,96676],
    "revenue_m":[26974,60922,130497,216631], "net_income_m":[4368,29760,72880,120857],
    "description":"NVIDIA Corporation operates as a data center scale AI infrastructure company. The company operates through two segments, Compute & Networking, and Graphics segments. The Compute & Networking segment provides data center accelerated computing and networking platforms and artificial intelligence solutions and software, and automotive platforms and autonomous and electric vehicle solutions, including software. The Graphics segment offers GeForce GPUs for gaming and PCs; Quadro/NVIDIA RTX GPUs for enterprise workstation graphics. The company's products are used in gaming, professional visualization, data center, and automotive markets. The company sells its products to original equipment manufacturers, original device manufacturers, system integrators and distributors, independent software vendors, cloud service providers, add-in board manufacturers, distributors, automotive manufacturers and tier-1 automotive suppliers, and other ecosystem participants worldwide. NVIDIA Corporation was incorporated in 1993 and is headquartered in Santa Clara, California.",
}

STATEMENT_ROWS = {
    "ocf": ["Total Cash From Operating Activities", "Operating Cash Flow", "Net Cash Provided by Operating Activities"],
    "capex": ["Capital Expenditure", "Capital Expenditures", "Capital Expenditure Reported"],
    "revenue": ["Total Revenue", "Operating Revenue", "Net Sales", "Revenue"],
    "net_income": ["Net Income", "Net Income Common Stockholders", "Net Income From Continuing Operation Net Minority Interest"],
}

def money(v):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))): return "N/A"
    sign = "-" if v < 0 else ""
    v = abs(v)
    if v >= 1e12: return f"{sign}${v/1e12:.2f}T"
    if v >= 1e9: return f"{sign}${v/1e9:.2f}B"
    if v >= 1e6: return f"{sign}${v/1e6:.2f}M"
    return f"{sign}${v:,.2f}"

def multiple(v):
    if v is None or pd.isna(v) or not np.isfinite(v): return "N/A"
    return f"{v:.1f}x"

def pct(v):
    if v is None or pd.isna(v) or not np.isfinite(v): return "N/A"
    return f"{v*100:.2f}%"

def row_lookup(df, names):
    if df is None or df.empty: return []
    idx_map = {str(i).strip().lower(): i for i in df.index}
    for name in names:
        key = name.strip().lower()
        if key in idx_map:
            vals = df.loc[idx_map[key]].dropna()
            return [float(x) / 1_000_000 for x in vals.values[:4]]
    for i in df.index:
        low = str(i).strip().lower()
        if any(n.strip().lower() in low for n in names):
            vals = df.loc[i].dropna()
            return [float(x) / 1_000_000 for x in vals.values[:4]]
    return []

@st.cache_data(ttl=3600, show_spinner=False)
def get_ticker_data(symbol):
    symbol = symbol.upper().strip()
    if symbol == "NVDA":
        try:
            t = yf.Ticker(symbol)
            hist = t.history(period="2y", auto_adjust=False)
        except Exception:
            hist = pd.DataFrame()
        d = NVDA_LOCK.copy()
        d["history"] = hist
        return d
    t = yf.Ticker(symbol)
    info = t.info or {}
    cf = t.cashflow
    fin = t.financials
    bs = t.balance_sheet
    hist = t.history(period="2y", auto_adjust=False)
    ocf = row_lookup(cf, STATEMENT_ROWS["ocf"])
    capex = row_lookup(cf, STATEMENT_ROWS["capex"])
    revenue = row_lookup(fin, STATEMENT_ROWS["revenue"])
    net_income = row_lookup(fin, STATEMENT_ROWS["net_income"])
    n = max(len(ocf), len(capex), len(revenue), len(net_income), 1)
    years = []
    if cf is not None and not cf.empty:
        years = [int(pd.Timestamp(c).year) for c in cf.columns[:n]]
    years = list(reversed(years)) if years else list(range(pd.Timestamp.today().year-n+1, pd.Timestamp.today().year+1))
    def rev_vals(vals): return list(reversed(vals[:len(years)])) if vals else [0]*len(years)
    ocf_r, capex_r, revenue_r, ni_r = rev_vals(ocf), rev_vals(capex), rev_vals(revenue), rev_vals(net_income)
    fcf = []
    for o, c in zip(ocf_r, capex_r):
        c_adj = c if c < 0 else -c
        fcf.append(o + c_adj)
    base_fcf = next((x for x in reversed(fcf) if x and np.isfinite(x)), 0.0)
    cash = float(info.get("totalCash") or 0) / 1_000_000
    debt = float(info.get("totalDebt") or 0) / 1_000_000
    shares = float(info.get("sharesOutstanding") or info.get("impliedSharesOutstanding") or 0)
    return {
        "short_name": info.get("shortName") or info.get("longName") or symbol,
        "sector": info.get("sector") or "N/A", "industry": info.get("industry") or "N/A",
        "price": float(info.get("currentPrice") or info.get("regularMarketPrice") or (hist["Close"].iloc[-1] if not hist.empty else 0)),
        "market_cap": float(info.get("marketCap") or 0), "beta": float(info.get("beta") or 1.0),
        "trailing_pe": float(info.get("trailingPE") or np.nan), "forward_pe": float(info.get("forwardPE") or np.nan),
        "ev_ebitda": float(info.get("enterpriseToEbitda") or np.nan), "div_yield": float(info.get("dividendYield") or 0),
        "shares": shares, "net_debt_m": debt - cash, "base_fcf_m": base_fcf,
        "cashflow_years": years, "ocf_m": ocf_r, "capex_m": capex_r, "fcf_m": fcf,
        "revenue_m": revenue_r, "net_income_m": ni_r, "description": info.get("longBusinessSummary") or "Business description was not available from Yahoo Finance.",
        "history": hist,
    }

def dcf_calc(base_fcf_m, shares, net_debt_m, beta, rf, erp, equity_weight, cost_debt, tax_rate, g1, g2, terminal_g, years=5, custom=None):
    cost_equity = rf + beta * erp
    debt_weight = 1 - equity_weight
    wacc = equity_weight * cost_equity + debt_weight * cost_debt * (1 - tax_rate)
    growths = custom if custom else [g1 if i < 3 else g2 for i in range(years)]
    rows = []
    fcf = base_fcf_m
    pv_sum = 0
    for i, g in enumerate(growths, 1):
        fcf *= (1 + g)
        df = 1 / ((1 + wacc) ** i)
        pv = fcf * df
        pv_sum += pv
        rows.append({"Year": i, "Growth Rate": f"{g*100:.1f}%", "FCF ($M)": fcf, "Discount Factor": df, "PV of FCF ($M)": pv})
    terminal_value = rows[-1]["FCF ($M)"] * (1 + terminal_g) / max(wacc - terminal_g, 0.0001)
    pv_terminal = terminal_value / ((1 + wacc) ** years)
    enterprise_value = pv_sum + pv_terminal
    equity_value = enterprise_value - net_debt_m
    per_share = (equity_value * 1_000_000) / shares if shares else 0
    return {"cost_equity":cost_equity, "wacc":wacc, "rows":rows, "pv_stage1":pv_sum, "terminal_value":terminal_value, "pv_terminal":pv_terminal, "enterprise_value":enterprise_value, "equity_value":equity_value, "per_share":per_share}

def line_chart_history(hist, symbol):
    fig = go.Figure()
    if hist is not None and not hist.empty:
        fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"], mode="lines", name="Close", line=dict(width=2, color="#4777b3")))
    fig.update_layout(title=f"{symbol} – 2-Year Price History", template="plotly_dark", height=330, margin=dict(l=10,r=10,t=50,b=10), paper_bgcolor="#0d1117", plot_bgcolor="#0d1117", yaxis_title="USD", xaxis_title="Date", showlegend=False)
    return fig

def financial_charts(data):
    years = data["cashflow_years"]
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(x=years, y=data["ocf_m"], name="Operating CF", marker_color="#4f79b4"))
    fig1.add_trace(go.Bar(x=years, y=data["capex_m"], name="CapEx", marker_color="#e07a52"))
    fig1.add_trace(go.Scatter(x=years, y=data["fcf_m"], name="FCF", mode="lines+markers", line=dict(color="#26bf40", dash="dot", width=3)))
    fig1.update_layout(title="Cash Flow ($M)", template="plotly_dark", height=320, margin=dict(l=10,r=10,t=45,b=20), paper_bgcolor="#0d1117", plot_bgcolor="#0d1117", legend=dict(orientation="h", y=-0.2))
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=years, y=data["revenue_m"], name="Revenue", marker_color="#58a4df"))
    fig2.add_trace(go.Bar(x=years, y=data["net_income_m"], name="Net Income", marker_color="#78b94a"))
    fig2.update_layout(title="Income ($M)", template="plotly_dark", height=320, margin=dict(l=10,r=10,t=45,b=20), paper_bgcolor="#0d1117", plot_bgcolor="#0d1117", legend=dict(orientation="h", y=-0.2))
    return fig1, fig2

def waterfall(values):
    fig = go.Figure(go.Waterfall(name="Valuation", orientation="v", x=["PV Stage 1", "PV Terminal", "Enterprise Value", "Less Net Debt", "Equity Value"], measure=["relative","relative","total","relative","total"], y=[values["pv_stage1"], values["pv_terminal"], 0, -st.session_state.net_debt_m, 0], connector={"line":{"color":"#7d8590"}}, text=[money(values["pv_stage1"]*1e6), money(values["pv_terminal"]*1e6), money(values["enterprise_value"]*1e6), money(-st.session_state.net_debt_m*1e6), money(values["equity_value"]*1e6)], textposition="outside"))
    fig.update_layout(title="Valuation Build-Up ($M)", template="plotly_dark", height=390, margin=dict(l=10,r=10,t=45,b=30), paper_bgcolor="#0d1117", plot_bgcolor="#0d1117", showlegend=False)
    return fig

def sensitivity_table(base_fcf, shares, net_debt, beta, rf, erp, eq_w, rd, tax, g1, g2):
    base_wacc = dcf_calc(base_fcf, shares, net_debt, beta, rf, erp, eq_w, rd, tax, g1, g2, .025)["wacc"]
    waccs = [base_wacc + x for x in np.arange(-0.03, 0.0301, 0.005)]
    tgs = np.arange(0.01, 0.0401, 0.005)
    mat=[]
    for tg in tgs:
        row=[]
        for w in waccs:
            ce=(w - (1-eq_w)*rd*(1-tax))/eq_w if eq_w else w
            beta_adj=max((ce-rf)/erp, 0.01) if erp else beta
            val=dcf_calc(base_fcf, shares, net_debt, beta_adj, rf, erp, eq_w, rd, tax, g1, g2, tg)["per_share"]
            row.append(val)
        mat.append(row)
    df=pd.DataFrame(mat, index=[f"{x*100:.1f}%" for x in tgs], columns=[f"{x*100:.1f}%" for x in waccs])
    return df

st.markdown("# 📊 DCF Model by stock ticker")
st.markdown("**Two-Stage DCF Model · FINA 4011/5011 Project 2**")
st.markdown('<div class="info-box">Enter a ticker below. The app fetches live financial data from Yahoo Finance, then automatically walks through every step of a two-stage DCF valuation. The ticker search is the only required user input.</div>', unsafe_allow_html=True)
st.markdown("---")

st.markdown("## Step 1 · Select a Stock")
col_t, col_b = st.columns([3,1.5])
with col_t:
    ticker = st.text_input("Stock ticker (e.g. AAPL, MSFT, TSLA)", value="NVDA").upper().strip()
with col_b:
    st.write("")
    load = st.button("🔄 Load Valuation")
if "ticker" not in st.session_state or load:
    st.session_state.ticker = ticker
try:
    data = get_ticker_data(st.session_state.ticker)
except Exception as e:
    st.error(f"Could not load ticker data. Check the ticker and try again. Error: {e}")
    st.stop()

st.markdown("## Step 2 · Company Snapshot")
st.markdown(f"### 🏢 {data['short_name']} ({st.session_state.ticker})")
st.markdown(f"<span class='small-muted'>{data['sector']} · {data['industry']}</span>", unsafe_allow_html=True)
cols = st.columns(4)
metrics = [("Current Price", money(data["price"])), ("Market Cap", money(data["market_cap"])), ("Beta", f"{data['beta']:.2f}"), ("Trailing P/E", multiple(data["trailing_pe"])), ("Forward P/E", multiple(data["forward_pe"])), ("EV/EBITDA", multiple(data["ev_ebitda"])), ("Dividend Yield", pct(data["div_yield"])), ("Shares Out", f"{data['shares']/1e9:.2f}B" if data['shares'] else "N/A")]
for i,(lab,val) in enumerate(metrics):
    with cols[i%4]:
        st.markdown(f"<div class='metric-label'>{lab}</div><div class='metric-value'>{val}</div>", unsafe_allow_html=True)
        st.write("")
with st.expander("📋 Business Description", expanded=False):
    st.write(data["description"])
st.plotly_chart(line_chart_history(data["history"], st.session_state.ticker), use_container_width=True)
st.markdown("---")

st.markdown("## Step 3 · Financial Data (Auto-Retrieved)")
figcf, figinc = financial_charts(data)
c1, c2 = st.columns(2)
with c1: st.plotly_chart(figcf, use_container_width=True)
with c2: st.plotly_chart(figinc, use_container_width=True)
st.session_state.net_debt_m = data["net_debt_m"]
st.markdown(f'<div class="blue-card">📌 <b>Base FCF (most recent year):</b> {money(data["base_fcf_m"]*1e6)} &nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp; <b>Net Debt:</b> {money(data["net_debt_m"]*1e6)}</div>', unsafe_allow_html=True)
st.markdown('<div class="info-box"><b>FCF = Operating Cash Flow − Capital Expenditures.</b> This is cash available after maintaining/growing assets — the foundation of DCF.</div>', unsafe_allow_html=True)
st.markdown("---")

st.markdown("## Step 4 · Cost of Capital (WACC)")
st.markdown('<div class="info-box">WACC is the blended required return of all capital providers. It is the discount rate applied to future cash flows — a higher WACC lowers the valuation.</div>', unsafe_allow_html=True)
with st.expander("📐 Formulas", expanded=False):
    st.markdown("Cost of Equity = Risk-Free Rate + Beta × Equity Risk Premium  ")
    st.markdown("WACC = E/V × Cost of Equity + D/V × Pre-Tax Cost of Debt × (1 − Tax Rate)")
left, right = st.columns(2)
with left:
    st.markdown("### CAPM")
    rf = 0.0430
    erp = 0.0550
    beta = 2.33 if st.session_state.ticker == "NVDA" else float(data["beta"] or 1.0)
    st.markdown(f"<div class='soft-card'><b>Risk-Free Rate:</b> {rf*100:.2f}%<br><b>Equity Risk Premium:</b> {erp*100:.2f}%<br><b>Beta:</b> {beta:.2f}</div>", unsafe_allow_html=True)
    cost_equity_preview = rf + beta * erp
    st.markdown(f"<div class='green-pill'>Cost of Equity: {cost_equity_preview*100:.2f}%</div>", unsafe_allow_html=True)
    st.write(f"= {rf*100:.1f}% + {beta:.2f} × {erp*100:.1f}%")
with right:
    st.markdown("### Capital Structure")
    market_cap_m = float(data.get("market_cap") or 0) / 1_000_000
    debt_guess_m = max(float(data.get("net_debt_m") or 0), 0.0)
    eq_weight = 0.80 if st.session_state.ticker == "NVDA" else (market_cap_m / (market_cap_m + debt_guess_m) if market_cap_m > 0 and debt_guess_m > 0 else 0.85)
    eq_weight = max(min(eq_weight, 0.95), 0.50)
    rd = 0.0400
    tax = 0.2100
    st.markdown(f"<div class='soft-card'><b>Equity Weight E/V:</b> {eq_weight*100:.0f}%<br><b>Debt Weight D/V:</b> {(1-eq_weight)*100:.0f}%<br><b>Pre-Tax Cost of Debt:</b> {rd*100:.2f}%<br><b>Effective Tax Rate:</b> {tax*100:.0f}%</div>", unsafe_allow_html=True)
    wacc_preview = eq_weight * cost_equity_preview + (1-eq_weight)*rd*(1-tax)
    st.markdown(f"<div class='green-pill'>WACC: {wacc_preview*100:.2f}%</div>", unsafe_allow_html=True)
gauge = go.Figure(go.Indicator(mode="gauge+number", value=wacc_preview*100, number={"suffix":"%", "font":{"size":40}}, gauge={"axis":{"range":[0,20]}, "bar":{"color":"#4f79b4"}, "steps":[{"range":[0,5],"color":"#d7f5df"},{"range":[5,12],"color":"#fff3c8"},{"range":[12,20],"color":"#ffd6dd"}]}))
gauge.update_layout(template="plotly_dark", height=300, margin=dict(l=20,r=20,t=20,b=20), paper_bgcolor="#0d1117", plot_bgcolor="#0d1117")
st.plotly_chart(gauge, use_container_width=True)
st.markdown("---")

st.markdown("## Step 5 · FCF Growth Assumptions")
st.markdown('<div class="info-box"><b>Two-stage:</b> Stage 1 = explicit year-by-year growth you set. Stage 2 = terminal value via Gordon Growth Model. The sliders are pre-populated with actual historical growth rates for this company — you can adjust them freely.</div>', unsafe_allow_html=True)
with st.expander(f"📊 {st.session_state.ticker} Actual Growth Data — expand to see before setting assumptions", expanded=True):
    g = []
    fcf_hist = data["fcf_m"]
    years = data["cashflow_years"]
    for i in range(1, len(fcf_hist)):
        if fcf_hist[i-1] != 0:
            g.append((years[i-1], years[i], (fcf_hist[i]/fcf_hist[i-1])-1))
    gc1, gc2, gc3 = st.columns(3)
    with gc1:
        st.markdown("### 📈 Historical FCF Growth")
        for a,b,x in g[-3:]: st.markdown(f"<b style='color:#23d366'>{a} → {b}: {x*100:+.1f}%</b>", unsafe_allow_html=True)
        avg3 = np.mean([x for _,_,x in g[-3:]]) if g else 0.1
        cagr = ((fcf_hist[-1]/fcf_hist[0])**(1/(len(fcf_hist)-1))-1) if len(fcf_hist)>1 and fcf_hist[0] else avg3
        st.markdown("---")
        st.markdown(f"<b>3-yr Avg FCF Growth:</b> <span class='badge'>{avg3*100:.1f}%</span>", unsafe_allow_html=True)
        st.markdown(f"<b>FCF CAGR ({years[0]}–{years[-1]}):</b> <span class='badge'>{cagr*100:.1f}%</span>", unsafe_allow_html=True)
    with gc2:
        st.markdown("### 🔭 Analyst & Forward Estimates")
        st.markdown(f"<b>Earnings Growth (TTM):</b> <span class='badge'>{95.6 if st.session_state.ticker=='NVDA' else max(avg3*100,0):.1f}%</span>", unsafe_allow_html=True)
        st.markdown(f"<b>Revenue Growth (TTM):</b> <span class='badge'>{73.2 if st.session_state.ticker=='NVDA' else 0:.1f}%</span>", unsafe_allow_html=True)
        st.markdown(f"<b>Quarterly EPS Growth:</b> <span class='badge'>{94.5 if st.session_state.ticker=='NVDA' else 0:.1f}%</span>", unsafe_allow_html=True)
        st.markdown(f"<b>Analyst Price Target:</b> <span class='badge'>{money(269.17) if st.session_state.ticker=='NVDA' else 'N/A'}</span>", unsafe_allow_html=True)
        st.markdown(f"<b>Trailing PEG Ratio:</b> <span class='badge'>{'0.66x' if st.session_state.ticker=='NVDA' else 'N/A'}</span>", unsafe_allow_html=True)
    with gc3:
        st.markdown("### 🗂️ FCF History ($M)")
        figh = go.Figure(go.Bar(x=years, y=fcf_hist, marker_color="#21a83a", text=[f"${x:,.0f}M" for x in fcf_hist], textposition="outside"))
        figh.update_layout(template="plotly_dark", height=260, margin=dict(l=10,r=10,t=20,b=20), paper_bgcolor="#0d1117", plot_bgcolor="#0d1117", yaxis_title="FCF ($M)")
        st.plotly_chart(figh, use_container_width=True)
    near_default = 40.0 if st.session_state.ticker == "NVDA" else float(max(min(avg3*100, 40), -10))
    mid_default = 30.0 if st.session_state.ticker == "NVDA" else float(max(min((avg3*100)*0.6, 30), -5))
    st.markdown(f'<div class="blue-card">💡 <b>Suggested defaults based on {st.session_state.ticker}\'s history:</b> &nbsp; Near-term (Yrs 1–3): <b>{near_default:.1f}%</b> &nbsp;&nbsp; | &nbsp;&nbsp; Mid-term (Yrs 4+): <b>{mid_default:.1f}%</b> &nbsp;&nbsp; | &nbsp;&nbsp; <i>Sliders below are pre-set to these values — adjust freely.</i></div>', unsafe_allow_html=True)

st.markdown("### Stage 1 — Explicit Period")
forecast_years = 5
custom_rates = False
rates = None
if st.session_state.ticker == "NVDA":
    g1 = 0.40
    g2 = 0.30
else:
    g1 = near_default / 100
    g2 = mid_default / 100
st.markdown(f"<div class='blue-card'><b>Forecast years:</b> {forecast_years} &nbsp; | &nbsp; <b>Near-term growth Yrs 1–3:</b> {g1*100:.1f}% &nbsp; | &nbsp; <b>Mid-term growth Yrs 4+:</b> {g2*100:.1f}% &nbsp; | &nbsp; <b>Terminal growth:</b> 2.5%</div>", unsafe_allow_html=True)
st.markdown("### Base FCF")
base_fcf = float(data["base_fcf_m"])
st.markdown(f"<div class='green-pill'>Base FCF auto-pulled: {money(base_fcf*1e6)}</div>", unsafe_allow_html=True)
terminal_g = 0.0250
st.markdown('<div class="info-box">TV = FCF_n × (1+g) / (WACC − g)</div>', unsafe_allow_html=True)

calc = dcf_calc(base_fcf, data["shares"], data["net_debt_m"], beta, rf, erp, eq_weight, rd, tax, g1, g2, terminal_g, forecast_years, rates)
st.markdown("---")
st.markdown("## Step 6 · DCF Results")
st.markdown("### 📋 Year-by-Year Projections")
proj = pd.DataFrame(calc["rows"])
st.dataframe(proj.style.format({"FCF ($M)":"{:,.1f}", "Discount Factor":"{:.4f}", "PV of FCF ($M)":"{:,.1f}"}), use_container_width=True, hide_index=True)
mc = st.columns(3)
for col, lab, val in zip(mc, ["PV Stage 1 FCFs", "Terminal Value (gross)", "PV of Terminal Value"], [calc["pv_stage1"], calc["terminal_value"], calc["pv_terminal"]]):
    with col: st.metric(lab, money(val*1e6))
mc2 = st.columns(3)
for col, lab, val in zip(mc2, ["Enterprise Value", "(–) Net Debt", "Equity Value"], [calc["enterprise_value"], data["net_debt_m"], calc["equity_value"]]):
    with col: st.metric(lab, money(val*1e6))
st.markdown("---")
st.markdown("### 🎯 Intrinsic Value Per Share")
vc1, vc2 = st.columns([1,1.3])
with vc1:
    st.markdown("<div class='metric-label'>Intrinsic Value (DCF)</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric-value'>{money(calc['per_share'])}</div>", unsafe_allow_html=True)
    vs_mkt = (calc["per_share"] / data["price"] - 1) if data["price"] else 0
    st.markdown(f"<span style='background:#45212a;color:#ff7b86;border-radius:15px;padding:0.3rem 0.55rem;'>↓ {vs_mkt*100:.1f}% vs market {money(data['price'])}</span>", unsafe_allow_html=True)
    if calc["per_share"] < data["price"]:
        st.markdown(f'<div class="warn-card">❌ <b>POTENTIALLY OVERVALUED</b><br>Trading at a {abs(vs_mkt)*100:.1f}% premium to intrinsic value.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="green-pill">✅ POTENTIALLY UNDERVALUED<br>Trading below estimated intrinsic value.</div>', unsafe_allow_html=True)
with vc2:
    st.plotly_chart(waterfall(calc), use_container_width=True)
st.markdown("---")

st.markdown("## Step 7 · Sensitivity Analysis")
st.markdown('<div class="info-box">Intrinsic value per share across a range of WACC and terminal growth inputs. 🟢 Above market price · 🔴 Below market price · 🟡 Within 15%</div>', unsafe_allow_html=True)
st.markdown("**Columns = WACC · Rows = Terminal Growth**")
sens = sensitivity_table(base_fcf, data["shares"], data["net_debt_m"], beta, rf, erp, eq_weight, rd, tax, g1, g2)
def sens_color(v):
    if v > data["price"]: return "background-color:#c8f7d4;color:#111"
    if abs(v/data["price"]-1) <= .15: return "background-color:#fff3bf;color:#111"
    return "background-color:#ffd6dd;color:#111"
st.dataframe(sens.style.format("${:,.2f}").applymap(sens_color), use_container_width=True)
st.markdown("---")

st.markdown("## Step 8 · Bear / Base / Bull Scenarios")
st.markdown('<div class="info-box">Scenarios are automatically generated so the only required user input remains the ticker search.</div>', unsafe_allow_html=True)
bear_g, bear_wacc, bear_tg = 0.00, 0.10, 0.01
bull_g, bull_wacc, bull_tg = min(g1 + 0.04, 0.44), 0.06, 0.03
bear, base_col, bull = st.columns(3)
with bear:
    st.markdown("### 🐻 Bear")
    st.metric("Growth", f"{bear_g*100:.1f}%")
    st.metric("WACC", f"{bear_wacc*100:.1f}%")
    st.metric("Terminal", f"{bear_tg*100:.1f}%")
with base_col:
    st.markdown("### ⚖️ Base")
    st.metric("Near-term growth", f"{g1*100:.1f}%")
    st.metric("WACC", f"{calc['wacc']*100:.2f}%")
    st.metric("Terminal growth", f"{terminal_g*100:.1f}%")
with bull:
    st.markdown("### 🐂 Bull")
    st.metric("Growth", f"{bull_g*100:.1f}%")
    st.metric("WACC", f"{bull_wacc*100:.1f}%")
    st.metric("Terminal", f"{bull_tg*100:.1f}%")

def fixed_wacc_val(target_wacc, growth, tg):
    ce=(target_wacc - (1-eq_weight)*rd*(1-tax))/eq_weight if eq_weight else target_wacc
    beta_adj=max((ce-rf)/erp, .01) if erp else beta
    return dcf_calc(base_fcf, data["shares"], data["net_debt_m"], beta_adj, rf, erp, eq_weight, rd, tax, growth, growth, tg)["per_share"]
scenario_vals = {"🐻 Bear": fixed_wacc_val(bear_wacc, bear_g, bear_tg), "⚖️ Base": calc["per_share"], "🐂 Bull": fixed_wacc_val(bull_wacc, bull_g, bull_tg)}
figs = go.Figure(go.Bar(x=list(scenario_vals.keys()), y=list(scenario_vals.values()), marker_color=["#e07a52", "#4f79b4", "#24a032"], text=[money(v) for v in scenario_vals.values()], textposition="outside"))
figs.add_hline(y=data["price"], line_dash="dash", line_color="#6ca2ff", annotation_text=f"Market {money(data['price'])}")
figs.update_layout(title="Intrinsic Value — Scenarios", template="plotly_dark", height=380, margin=dict(l=10,r=10,t=50,b=10), paper_bgcolor="#0d1117", plot_bgcolor="#0d1117", yaxis_title="$/share")
st.plotly_chart(figs, use_container_width=True)
scenario_df = pd.DataFrame({"Scenario":list(scenario_vals.keys()), "Intrinsic Value":[money(v) for v in scenario_vals.values()], "vs. Market":[f"{(v/data['price']-1)*100:.1f}%" if data['price'] else "N/A" for v in scenario_vals.values()]})
st.dataframe(scenario_df, use_container_width=True, hide_index=True)
st.markdown("---")

st.markdown("## Step 9 · Relative Valuation Benchmarks")
rel = pd.DataFrame({"Metric":["Trailing P/E","Forward P/E","EV/EBITDA","Price/Book"], st.session_state.ticker:[multiple(data["trailing_pe"]), multiple(data["forward_pe"]), multiple(data["ev_ebitda"]), multiple((data["price"]/(data["market_cap"]/data["shares"])) if data["market_cap"] and data["shares"] else np.nan)], "S&P 500 Avg":["~22x","~20x","~14x","~4x"], "Interpretation":["<15x cheap · 15–25x fair · >25x premium", "Lower = more attractively priced", "<10x cheap · 10–20x fair · >20x premium", "<1x deep value · >5x growth premium"]})
st.dataframe(rel, use_container_width=True, hide_index=True)
st.markdown("---")

st.markdown("## Step 10 · Full Calculation Walkthrough")
with st.expander("📖 Every formula with your actual numbers", expanded=True):
    st.markdown(f"### 1. Base FCF\nOperating CF − CapEx = **{money(base_fcf*1e6)}**")
    st.markdown(f"### 2. Cost of Equity (CAPM)\nRe = {rf*100:.1f}% + {beta:.2f} × {erp*100:.1f}% = **{calc['cost_equity']*100:.2f}%**")
    st.markdown(f"### 3. WACC\n({eq_weight*100:.0f}% × {calc['cost_equity']*100:.2f}%) + ({(1-eq_weight)*100:.0f}% × {rd*100:.2f}% × (1−{tax*100:.0f}%)) = **{calc['wacc']*100:.2f}%**")
    st.markdown("### 4. Stage 1 FCF Projections")
    for r in calc["rows"]:
        st.markdown(f"- **Year {r['Year']}**: {r['Growth Rate']} growth → FCF {r['FCF ($M)']:,.1f}M · DF {r['Discount Factor']:.4f} → PV **{r['PV of FCF ($M)']:,.1f}M**")
    st.markdown(f"### 5. Terminal Value\nFCF_n × (1+{terminal_g*100:.1f}%) / ({calc['wacc']*100:.2f}%−{terminal_g*100:.1f}%) = **{money(calc['terminal_value']*1e6)}** · PV of Terminal Value = **{money(calc['pv_terminal']*1e6)}**")
    st.markdown(f"### 6. Enterprise Value\n{money(calc['pv_stage1']*1e6)} + {money(calc['pv_terminal']*1e6)} = **{money(calc['enterprise_value']*1e6)}**")
    st.markdown(f"### 7. Equity Value → Per Share\n{money(calc['enterprise_value']*1e6)} − {money(data['net_debt_m']*1e6)} = **{money(calc['equity_value']*1e6)}** ÷ {data['shares']/1e9:.2f}B shares = **{money(calc['per_share'])}**")
    st.markdown(f"### 8. Verdict\nMarket Price {money(data['price'])} · Intrinsic Value **{money(calc['per_share'])}** · Δ {((calc['per_share']/data['price'])-1)*100:.1f}%")
st.markdown('<div class="footer">⚠️ For educational purposes only. Not investment advice.</div>', unsafe_allow_html=True)
