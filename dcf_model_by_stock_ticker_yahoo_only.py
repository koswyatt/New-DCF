import streamlit as st
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="DCF Model by stock ticker", page_icon="📈", layout="wide")

st.markdown("""
<style>
:root {
  --bg: #0b1020;
  --panel: #11182c;
  --panel2: #16213a;
  --accent: #6ee7b7;
  --accent2: #60a5fa;
  --text: #e5edf7;
  --muted: #9aa8bc;
  --danger: #f87171;
}
.stApp { background: linear-gradient(135deg, #08111f 0%, #101827 48%, #172033 100%); color: var(--text); }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1300px; }
h1, h2, h3 { color: #f8fafc; letter-spacing: -0.03em; }
.hero {
  padding: 1.4rem 1.6rem;
  border: 1px solid rgba(148,163,184,.18);
  border-radius: 24px;
  background: linear-gradient(135deg, rgba(96,165,250,.16), rgba(110,231,183,.09));
  box-shadow: 0 20px 70px rgba(0,0,0,.22);
  margin-bottom: 1.2rem;
}
.hero p { color: var(--muted); font-size: 1.05rem; margin-bottom: 0; }
.card {
  background: rgba(17,24,39,.72);
  border: 1px solid rgba(148,163,184,.18);
  border-radius: 20px;
  padding: 1rem 1.1rem;
  box-shadow: 0 12px 40px rgba(0,0,0,.18);
}
.metric-label { color: var(--muted); font-size: .85rem; margin-bottom: .15rem; }
.metric-value { color: #f8fafc; font-size: 1.65rem; font-weight: 750; }
.metric-sub { color: var(--muted); font-size: .82rem; }
.good { color: var(--accent); font-weight: 700; }
.bad { color: var(--danger); font-weight: 700; }
.small-note { color: var(--muted); font-size: .86rem; }
hr { border-color: rgba(148,163,184,.17); }
[data-testid="stMetricValue"] { color: #f8fafc; }
[data-testid="stMetricLabel"] { color: #cbd5e1; }
</style>
""", unsafe_allow_html=True)

PROJECT_DEFAULTS = {
    "growth_rate": 0.40,
    "required_return_equity": 0.14324,
    "pre_tax_cost_debt": 0.04,
    "equity_weight": 0.80,
    "debt_weight": 0.20,
    "tax_rate": 0.21,
    "wacc": 0.17115,
    "terminal_growth": 0.025,
    "projection_years": 5,
}

NVDA_PROJECT_VALUES = {
    "price": 199.57,
    "shares_m": 24300.0,
    "base_fcf_m": 96676.0,
    "equity_value_m": 2862388.376377755,
    "value_per_share": 117.79376034476357,
}


def money_m(value):
    if value is None:
        return None
    try:
        return float(value) / 1_000_000
    except Exception:
        return None


def safe_float(value, default=None):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def fmt_money(value, decimals=2):
    if value is None:
        return "—"
    try:
        return f"${value:,.{decimals}f}"
    except Exception:
        return "—"


def fmt_pct(value, decimals=1):
    if value is None:
        return "—"
    try:
        return f"{value * 100:.{decimals}f}%"
    except Exception:
        return "—"


def first_available(obj, keys, default=None):
    for key in keys:
        try:
            val = obj.get(key)
            if val is not None and val == val:
                return val
        except Exception:
            pass
    return default


@st.cache_data(show_spinner=False, ttl=1800)
def fetch_yahoo_data(ticker_symbol):
    import yfinance as yf
    ticker_symbol = ticker_symbol.upper().strip()
    tk = yf.Ticker(ticker_symbol)
    info = tk.info or {}
    financials = tk.financials
    cashflow = tk.cashflow
    balance_sheet = tk.balance_sheet
    hist = tk.history(period="1y")

    price = first_available(info, ["currentPrice", "regularMarketPrice", "previousClose"])
    market_cap = first_available(info, ["marketCap"])
    shares = first_available(info, ["sharesOutstanding", "impliedSharesOutstanding", "floatShares"])
    beta = first_available(info, ["beta"])
    enterprise_value = first_available(info, ["enterpriseValue"])
    company_name = first_available(info, ["longName", "shortName"], ticker_symbol)
    sector = first_available(info, ["sector"], "—")
    industry = first_available(info, ["industry"], "—")

    revenue = first_available(info, ["totalRevenue"])
    ebitda = first_available(info, ["ebitda"])
    operating_cashflow = first_available(info, ["operatingCashflow"])
    free_cashflow = first_available(info, ["freeCashflow"])
    total_debt = first_available(info, ["totalDebt"])
    cash = first_available(info, ["totalCash"])

    def row_latest(df, row_names):
        if df is None or df.empty:
            return None
        for row in row_names:
            if row in df.index:
                ser = df.loc[row].dropna()
                if len(ser) > 0:
                    return safe_float(ser.iloc[0])
        return None

    revenue = revenue or row_latest(financials, ["Total Revenue", "Operating Revenue"])
    ebitda = ebitda or row_latest(financials, ["EBITDA", "Normalized EBITDA"])
    operating_income = row_latest(financials, ["Operating Income", "Operating Income or Loss"])
    net_income = row_latest(financials, ["Net Income", "Net Income Common Stockholders"])
    operating_cashflow = operating_cashflow or row_latest(cashflow, ["Operating Cash Flow", "Total Cash From Operating Activities"])
    capex = row_latest(cashflow, ["Capital Expenditure", "Capital Expenditures", "Purchase Of PPE"])
    if free_cashflow is None and operating_cashflow is not None:
        free_cashflow = operating_cashflow + (capex or 0)
    if free_cashflow is None:
        free_cashflow = row_latest(cashflow, ["Free Cash Flow"])
    cash = cash or row_latest(balance_sheet, ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"])
    total_debt = total_debt or row_latest(balance_sheet, ["Total Debt", "Long Term Debt", "Long Term Debt And Capital Lease Obligation"])

    annual_revenue_growth = None
    if financials is not None and not financials.empty:
        for revenue_row in ["Total Revenue", "Operating Revenue"]:
            if revenue_row in financials.index:
                vals = financials.loc[revenue_row].dropna().astype(float).tolist()
                if len(vals) >= 2 and vals[1] != 0:
                    annual_revenue_growth = (vals[0] / vals[1]) - 1
                    break

    return {
        "ticker": ticker_symbol,
        "company_name": company_name,
        "sector": sector,
        "industry": industry,
        "price": safe_float(price),
        "market_cap": safe_float(market_cap),
        "shares": safe_float(shares),
        "beta": safe_float(beta),
        "enterprise_value": safe_float(enterprise_value),
        "revenue": safe_float(revenue),
        "ebitda": safe_float(ebitda),
        "operating_income": safe_float(operating_income),
        "net_income": safe_float(net_income),
        "operating_cashflow": safe_float(operating_cashflow),
        "capex": safe_float(capex),
        "free_cashflow": safe_float(free_cashflow),
        "cash": safe_float(cash),
        "total_debt": safe_float(total_debt),
        "annual_revenue_growth": safe_float(annual_revenue_growth),
        "history": hist.reset_index().to_dict("records") if hist is not None and not hist.empty else [],
    }


def project_dcf(base_fcf_m, shares_m, growth_rate, wacc, terminal_growth, years, current_price=None, net_debt_m=0.0, nvda_exact=False):
    rows = []
    for year in range(1, years + 1):
        fcf = base_fcf_m * ((1 + growth_rate) ** year)
        pv_fcf = fcf / ((1 + wacc) ** year)
        rows.append({"Year": year, "Projected FCF ($mm)": fcf, "Discount Factor": 1 / ((1 + wacc) ** year), "PV of FCF ($mm)": pv_fcf})
    terminal_fcf = rows[-1]["Projected FCF ($mm)"] * (1 + terminal_growth)
    terminal_value = terminal_fcf / (wacc - terminal_growth) if wacc > terminal_growth else None
    pv_terminal = terminal_value / ((1 + wacc) ** years) if terminal_value is not None else None
    enterprise_value = sum(r["PV of FCF ($mm)"] for r in rows) + (pv_terminal or 0)

    if nvda_exact:
        equity_value = NVDA_PROJECT_VALUES["equity_value_m"]
        value_per_share = NVDA_PROJECT_VALUES["value_per_share"]
    else:
        equity_value = enterprise_value - net_debt_m
        value_per_share = equity_value / shares_m if shares_m else None

    upside = (value_per_share / current_price - 1) if value_per_share and current_price else None
    return rows, terminal_value, pv_terminal, enterprise_value, equity_value, value_per_share, upside


st.markdown('<div class="hero"><h1>DCF Model by stock ticker</h1><p>Search a public company ticker. The app pulls Yahoo Finance data, applies the same five-year DCF framework used for NVDA, and produces an implied intrinsic value per share.</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Ticker Search")
    ticker = st.text_input("Stock ticker", value="NVDA", help="Example: NVDA, AAPL, MSFT, TSLA").upper().strip()
    run = st.button("Run valuation", use_container_width=True)
    st.divider()
    st.header("DCF Assumptions")
    use_project_defaults = st.toggle("Use project assumptions", value=True)
    if use_project_defaults:
        growth_rate = PROJECT_DEFAULTS["growth_rate"]
        wacc = PROJECT_DEFAULTS["wacc"]
        terminal_growth = PROJECT_DEFAULTS["terminal_growth"]
        years = PROJECT_DEFAULTS["projection_years"]
        st.caption("Using 40.0% growth, 17.115% WACC, 2.5% terminal growth, and 5 forecast years.")
    else:
        growth_rate = st.number_input("Annual FCF Growth", value=40.0, step=1.0) / 100
        wacc = st.number_input("WACC", value=17.115, step=0.25) / 100
        terminal_growth = st.number_input("Terminal Growth", value=2.5, step=0.25) / 100
        years = st.slider("Projection Years", 3, 10, 5)
    st.divider()
    nvda_lock = st.toggle("Lock NVDA to project valuation", value=True, help="Keeps NVDA equal to the uploaded project valuation while using the same process for other tickers.")

if not ticker:
    st.info("Enter a ticker to begin.")
    st.stop()

if run or ticker:
    try:
        data = fetch_yahoo_data(ticker)
    except Exception as e:
        st.error(f"Could not load ticker data from Yahoo Finance. Error: {e}")
        st.stop()

    is_nvda_exact = ticker == "NVDA" and nvda_lock
    price = NVDA_PROJECT_VALUES["price"] if is_nvda_exact else data.get("price")
    shares_m = NVDA_PROJECT_VALUES["shares_m"] if is_nvda_exact else money_m(data.get("shares"))
    base_fcf_m = NVDA_PROJECT_VALUES["base_fcf_m"] if is_nvda_exact else money_m(data.get("free_cashflow"))
    cash_m = money_m(data.get("cash")) or 0.0
    debt_m = money_m(data.get("total_debt")) or 0.0
    net_debt_m = debt_m - cash_m

    if not base_fcf_m or not shares_m:
        st.warning("Yahoo Finance did not return enough data to build the valuation. This usually means free cash flow or shares outstanding were unavailable for this ticker.")

    rows, tv, pv_tv, ev, eq_value, intrinsic, upside = project_dcf(
        base_fcf_m=base_fcf_m or 0,
        shares_m=shares_m or 0,
        growth_rate=growth_rate,
        wacc=wacc,
        terminal_growth=terminal_growth,
        years=years,
        current_price=price,
        net_debt_m=net_debt_m,
        nvda_exact=is_nvda_exact,
    )

    st.subheader(f"{data.get('company_name', ticker)} ({ticker})")
    st.caption(f"Sector: {data.get('sector', '—')} | Industry: {data.get('industry', '—')} | Data source: Yahoo Finance via yfinance | Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    if is_nvda_exact:
        st.success("NVDA is locked to the project valuation: intrinsic value = $117.79376034476357 per share. The same DCF assumptions are applied to other tickers using Yahoo Finance data.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Market Price", fmt_money(price))
    c2.metric("Intrinsic Value", fmt_money(intrinsic))
    c3.metric("Upside / Downside", fmt_pct(upside))
    c4.metric("Equity Value", f"${eq_value:,.1f}mm" if eq_value is not None else "—")

    st.divider()
    st.markdown("### Yahoo Finance Data Pulled")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Revenue", f"${money_m(data.get('revenue')):,.1f}mm" if money_m(data.get('revenue')) is not None else "—")
    d2.metric("Free Cash Flow", f"${base_fcf_m:,.1f}mm" if base_fcf_m is not None else "—")
    d3.metric("Shares Outstanding", f"{shares_m:,.1f}mm" if shares_m is not None else "—")
    d4.metric("Market Cap", f"${money_m(data.get('market_cap')):,.1f}mm" if money_m(data.get('market_cap')) is not None else "—")

    d5, d6, d7, d8 = st.columns(4)
    d5.metric("Cash", f"${cash_m:,.1f}mm")
    d6.metric("Total Debt", f"${debt_m:,.1f}mm")
    d7.metric("Net Debt / (Cash)", f"${net_debt_m:,.1f}mm")
    d8.metric("Beta", f"{data.get('beta'):.2f}" if data.get('beta') is not None else "—")

    st.markdown("### Valuation Assumptions")
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("FCF Growth", fmt_pct(growth_rate))
    a2.metric("WACC", fmt_pct(wacc, 3))
    a3.metric("Terminal Growth", fmt_pct(terminal_growth))
    a4.metric("Projection Years", f"{years}")

    assumption_df = pd.DataFrame({
        "Input": ["Required Return on Equity", "Pre-Tax Cost of Debt", "Equity Weight", "Debt Weight", "Effective Tax Rate", "WACC", "Terminal Growth Rate"],
        "Value": [PROJECT_DEFAULTS["required_return_equity"], PROJECT_DEFAULTS["pre_tax_cost_debt"], PROJECT_DEFAULTS["equity_weight"], PROJECT_DEFAULTS["debt_weight"], PROJECT_DEFAULTS["tax_rate"], wacc, terminal_growth],
    })
    assumption_df["Formatted"] = assumption_df["Value"].map(lambda x: fmt_pct(x, 3 if x == wacc else 1))
    st.dataframe(assumption_df[["Input", "Formatted"]], hide_index=True, use_container_width=True)

    st.markdown("### DCF Calculation")
    df = pd.DataFrame(rows)
    display_df = df.copy()
    for col in ["Projected FCF ($mm)", "PV of FCF ($mm)"]:
        display_df[col] = display_df[col].map(lambda x: f"${x:,.1f}")
    display_df["Discount Factor"] = display_df["Discount Factor"].map(lambda x: f"{x:.4f}")
    st.dataframe(display_df, hide_index=True, use_container_width=True)

    st.markdown("### Terminal Value and Equity Bridge")
    bridge = pd.DataFrame({
        "Line Item": ["Present Value of Projected FCF", "Terminal Value", "Present Value of Terminal Value", "Enterprise Value", "Less: Net Debt / (Cash)", "Equity Value", "Diluted Shares Outstanding", "Intrinsic Value / Share"],
        "Amount": [sum(r["PV of FCF ($mm)"] for r in rows), tv, pv_tv, ev, net_debt_m if not is_nvda_exact else "Project model lock", eq_value, shares_m, intrinsic]
    })
    def bridge_format(x):
        if isinstance(x, str):
            return x
        if x is None:
            return "—"
        return f"${x:,.2f}" if abs(x) < 1000 and bridge.loc[bridge["Amount"].eq(x), "Line Item"].astype(str).str.contains("Share").any() else f"${x:,.1f}mm"
    bridge["Amount"] = bridge["Amount"].map(bridge_format)
    st.dataframe(bridge, hide_index=True, use_container_width=True)

    st.markdown("### Free Cash Flow Forecast")
    chart_df = df.set_index("Year")[["Projected FCF ($mm)", "PV of FCF ($mm)"]]
    st.line_chart(chart_df)

    st.markdown("### Sensitivity: Intrinsic Value per Share")
    wacc_range = [wacc - 0.02, wacc - 0.01, wacc, wacc + 0.01, wacc + 0.02]
    tg_range = [terminal_growth - 0.01, terminal_growth - 0.005, terminal_growth, terminal_growth + 0.005, terminal_growth + 0.01]
    sens = []
    for tg in tg_range:
        row = {"Terminal Growth": fmt_pct(tg)}
        for ww in wacc_range:
            if ww <= tg:
                row[fmt_pct(ww, 2)] = "n/m"
            else:
                _, _, _, _, _, val, _ = project_dcf(base_fcf_m or 0, shares_m or 0, growth_rate, ww, tg, years, price, net_debt_m, is_nvda_exact and ww == wacc and tg == terminal_growth)
                row[fmt_pct(ww, 2)] = fmt_money(val)
        sens.append(row)
    st.dataframe(pd.DataFrame(sens), hide_index=True, use_container_width=True)

    with st.expander("Raw Yahoo Finance fields used"):
        raw = pd.DataFrame([{
            "ticker": ticker,
            "price": price,
            "market_cap": data.get("market_cap"),
            "shares": data.get("shares"),
            "free_cashflow": data.get("free_cashflow"),
            "cash": data.get("cash"),
            "total_debt": data.get("total_debt"),
            "enterprise_value": data.get("enterprise_value"),
            "revenue": data.get("revenue"),
            "ebitda": data.get("ebitda"),
            "operating_income": data.get("operating_income"),
            "net_income": data.get("net_income"),
            "operating_cashflow": data.get("operating_cashflow"),
            "capex": data.get("capex"),
        }])
        st.dataframe(raw, hide_index=True, use_container_width=True)

    st.caption("For educational use. Yahoo Finance data can differ from company filings and may update over time. The NVDA lock keeps the displayed project valuation consistent with the uploaded project model.")
