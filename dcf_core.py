
import os
import io
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

import streamlit as st

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet

# =====================
# Debug helper
# =====================
def d(msg):
    st.write(f"**[DEBUG]** {msg}")

# =====================
# Data helpers
# =====================
def fetch_financial_data(ticker, want_points=5):
    """
    Returns a pd.Series of FCF (most recent first) with up to `want_points` points.
    Tries annual cashflow first, then quarterly. Includes debug messages.
    """
    try:
        t = yf.Ticker(ticker)
        d("Fetching cashflows (annual)")
        cashflow = t.cashflow
        if cashflow is None or cashflow.empty:
            d("Annual cashflow empty -> try quarterly")
            cashflow = t.quarterly_cashflow

        if cashflow is None or cashflow.empty:
            d("Both annual and quarterly cashflows empty")
            return None

        idx = cashflow.index.tolist()
        d(f"Cashflow rows: {idx}")

        # CFO
        cfo_row_names = [
            "Total Cash From Operating Activities",
            "Operating Cash Flow",
        ]
        cfo = None
        for nm in cfo_row_names:
            if nm in cashflow.index:
                cfo = cashflow.loc[nm]
                break
        if cfo is None:
            d("CFO row not found in cashflow")
            return None

        # CapEx
        capex_row_names = [
            "Capital Expenditures",
            "Investment In Property, Plant, and Equipment"
        ]
        capex = None
        for nm in capex_row_names:
            if nm in cashflow.index:
                capex = cashflow.loc[nm]
                break
        if capex is None:
            d("CapEx row not found in cashflow")
            return None

        fcf = (cfo + capex).dropna()
        if fcf.empty:
            d("FCF computed but empty after dropna")
            return None

        # Sort most-recent first (columns are datetimes or periods)
        try:
            fcf = fcf.sort_index(ascending=False)
        except Exception:
            pass

        d(f"FCF head: {fcf.head().to_dict()}")
        return fcf.head(want_points)
    except Exception as e:
        d(f"Error in fetch_financial_data: {e}")
        return None

def safe_get_shares_outstanding(ticker_symbol: str):
    t = yf.Ticker(ticker_symbol)
    so = None
    try:
        fi = getattr(t, "fast_info", None)
        if fi is not None:
            so = getattr(fi, "shares_outstanding", None) if not isinstance(fi, dict) else fi.get("shares_outstanding", None)
    except Exception:
        so = None
    if not so:
        try:
            info = t.info
            so = info.get("sharesOutstanding", None)
        except Exception:
            so = None
    return so

def safe_get_current_price(ticker_symbol: str):
    t = yf.Ticker(ticker_symbol)
    price = None
    try:
        fi = getattr(t, "fast_info", None)
        if fi is not None:
            price = getattr(fi, "last_price", None) if not isinstance(fi, dict) else fi.get("last_price", None)
    except Exception:
        price = None
    if not price:
        try:
            hist = t.history(period="1d")
            if not hist.empty:
                price = float(hist["Close"].iloc[-1])
        except Exception:
            price = None
    return price

def safe_get_net_debt_from_yfinance(ticker_symbol: str):
    t = yf.Ticker(ticker_symbol)
    bs = t.balance_sheet
    if bs is None or bs.empty:
        return 0.0

    def _get_first_available(df, names):
        for n in names:
            if n in df.index:
                s = df.loc[n].dropna()
                if not s.empty:
                    return float(s.iloc[0])
        return None

    total_debt = _get_first_available(bs, ["Total Debt", "Long Term Debt", "Short Long Term Debt"])
    cash = _get_first_available(bs, ["Cash And Cash Equivalents", "Cash"])

    if total_debt is None:
        total_debt = 0.0
    if cash is None:
        cash = 0.0
    return float(total_debt - cash)

def fetch_beta_and_marketcap(ticker_symbol: str):
    t = yf.Ticker(ticker_symbol)
    beta = None
    mktcap = None
    try:
        info = t.info
        beta = info.get("beta", None) or info.get("beta3Year", None)
        mktcap = info.get("marketCap", None)
    except Exception:
        pass
    return beta, mktcap

def estimate_tax_rate(ticker_symbol: str):
    t = yf.Ticker(ticker_symbol)
    fin = t.financials
    if fin is None or fin.empty:
        return 0.21
    tax_names = ["Income Tax Expense", "Provision for Income Taxes"]
    pretax_names = ["Pretax Income", "Earnings Before Tax"]
    tax = None
    pretax = None
    for n in tax_names:
        if n in fin.index:
            s = fin.loc[n].dropna()
            if not s.empty:
                tax = float(s.iloc[0]); break
    for n in pretax_names:
        if n in fin.index:
            s = fin.loc[n].dropna()
            if not s.empty:
                pretax = float(s.iloc[0]); break
    if tax is None or pretax is None or pretax == 0:
        return 0.21
    return max(0.0, min(0.35, abs(tax / pretax)))

def fetch_risk_free_rate_us10y():
    try:
        t = yf.Ticker("^TNX")
        hist = t.history(period="5d")
        if hist is not None and not hist.empty:
            last = float(hist["Close"].iloc[-1])
            return last / 100.0
    except Exception:
        pass
    return 0.04

def fetch_interest_expense_and_debt(ticker_symbol: str):
    t = yf.Ticker(ticker_symbol)
    fin = t.financials
    bs = t.balance_sheet
    interest_expense = None
    total_debt = None
    if fin is not None and not fin.empty:
        for cand in ["Interest Expense", "Interest Expense Non Operating"]:
            if cand in fin.index:
                s = fin.loc[cand].dropna()
                if not s.empty:
                    interest_expense = float(abs(s.iloc[0])); break
    if bs is not None and not bs.empty:
        for cand in ["Total Debt", "Long Term Debt", "Short Long Term Debt"]:
            if cand in bs.index:
                s = bs.loc[cand].dropna()
                if not s.empty:
                    total_debt = float(s.iloc[0]); break
    return interest_expense, total_debt

# =====================
# Growth + DCF
# =====================
def project_shares(shares_outstanding: float, buyback_rate: float, years: int) -> list:
    shares = []
    s = shares_outstanding
    for _ in range(int(years)):
        s = s * (1 - buyback_rate)
        shares.append(s)
    return shares

def dcf_valuation_multistage(last_fcf, growth_rates, discount_rate, terminal_growth,
                             net_debt, shares_outstanding, share_path=None, per_share_method="current"):
    years = len(growth_rates)
    if years == 0:
        raise ValueError("At least one projection year required.")
    if discount_rate <= terminal_growth:
        raise ValueError("Discount rate must be greater than terminal growth.")
    if shares_outstanding <= 0:
        raise ValueError("Shares outstanding must be > 0.")

    projected_fcfs = []
    fcf = last_fcf
    for g in growth_rates:
        fcf = fcf * (1 + g)
        projected_fcfs.append(fcf)

    pv_fcfs_each = [cf / ((1 + discount_rate) ** i) for i, cf in enumerate(projected_fcfs, start=1)]
    pv_fcfs = float(np.sum(pv_fcfs_each))

    terminal_value = projected_fcfs[-1] * (1 + terminal_growth) / (discount_rate - terminal_growth)
    pv_terminal = float(terminal_value / ((1 + discount_rate) ** years))

    enterprise_value = float(pv_fcfs + pv_terminal)
    equity_value = float(enterprise_value - net_debt)

    if per_share_method == "buyback-adjusted" and share_path and len(share_path) == years:
        avg_shares = float(np.mean(share_path))
        denom = avg_shares if avg_shares > 0 else shares_outstanding
    else:
        denom = shares_outstanding
    fair_value_per_share = float(equity_value / denom)

    return {
        "projected_fcfs": projected_fcfs,
        "pv_fcfs_each": pv_fcfs_each,
        "pv_fcfs": pv_fcfs,
        "terminal_value": float(terminal_value),
        "pv_terminal": pv_terminal,
        "enterprise_value": enterprise_value,
        "equity_value": equity_value,
        "fair_value_per_share": fair_value_per_share,
        "denominator_shares": denom
    }

# =====================
# WACC
# =====================
def compute_wacc(ticker: str, risk_free_rate: float = None, market_risk_premium: float = 0.055):
    rf = risk_free_rate if risk_free_rate is not None else fetch_risk_free_rate_us10y()
    beta, mktcap = fetch_beta_and_marketcap(ticker)
    interest_expense, total_debt = fetch_interest_expense_and_debt(ticker)
    tax_rate = estimate_tax_rate(ticker)

    beta = beta if beta is not None else 1.0
    total_debt = float(total_debt) if total_debt is not None else 0.0
    mktcap = float(mktcap) if mktcap is not None else 0.0

    cost_of_equity = rf + beta * market_risk_premium

    if total_debt and interest_expense is not None and total_debt > 0:
        cost_of_debt = max(0.0, min(0.15, float(interest_expense) / float(total_debt)))
    else:
        cost_of_debt = 0.05

    E = max(0.0, mktcap)
    D = max(0.0, total_debt)
    V = max(1.0, E + D)
    w_e = E / V
    w_d = D / V

    wacc = w_e * cost_of_equity + w_d * cost_of_debt * (1 - tax_rate)

    return {
        "risk_free_rate": rf,
        "market_risk_premium": market_risk_premium,
        "beta": beta,
        "cost_of_equity": cost_of_equity,
        "cost_of_debt": cost_of_debt,
        "tax_rate": tax_rate,
        "weights_equity": w_e,
        "weights_debt": w_d,
        "market_cap": E,
        "total_debt": D,
        "wacc": wacc
    }

def wacc_color(wacc_value: float):
    if wacc_value < 0.10:
        return "green"
    if wacc_value <= 0.15:
        return "orange"
    return "red"

# =====================
# Plot & PDF
# =====================
def plot_and_save_projection(years, projected_fcfs, pv_each, outpath):
    plt.figure()
    plt.plot(years, projected_fcfs, label="Projected FCF")
    plt.plot(years, pv_each, label="Present Value of FCF")
    plt.xlabel("Year")
    plt.ylabel("Amount")
    plt.title("Projected FCFs and Present Values")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def generate_pdf_report(filepath, company_label, inputs, results, chart_path, wacc_breakdown=None):
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("DCF Valuation Report", styles['Title']))
    story.append(Paragraph(company_label, styles['Heading2']))
    story.append(Spacer(1, 12))

    inputs_table_data = [
        ["Projection Years", str(inputs.get("projection_years"))],
        ["Discount Rate", f"{inputs.get('discount_rate'):.2%}"],
        ["Terminal Growth", f"{inputs.get('terminal_growth'):.2%}"],
        ["Net Debt", f"${inputs.get('net_debt'):,.0f}"],
        ["Shares (denominator)", f"{int(results.get('denominator_shares')):,} ({inputs.get('per_share_method')})"],
        ["Buyback Rate", f"{inputs.get('buyback_rate'):.2%}"],
        ["Growth Notes", inputs.get("growth_notes","Three-stage or fade")]
    ]
    t = Table(inputs_table_data, hAlign='LEFT')
    t.setStyle(TableStyle([('GRID',(0,0),(-1,-1),0.5,colors.grey)]))
    story.append(Paragraph("Assumptions", styles['Heading3']))
    story.append(t)
    story.append(Spacer(1, 12))

    res_table_data = [
        ["PV of Projected FCFs", f"${results['pv_fcfs']:,.0f}"],
        ["PV of Terminal Value", f"${results['pv_terminal']:,.0f}"],
        ["Enterprise Value", f"${results['enterprise_value']:,.0f}"],
        ["Equity Value", f"${results['equity_value']:,.0f}"],
        ["Fair Value per Share", f"${results['fair_value_per_share']:,.2f}"]
    ]
    t2 = Table(res_table_data, hAlign='LEFT')
    t2.setStyle(TableStyle([('GRID',(0,0),(-1,-1),0.5,colors.grey)]))
    story.append(Paragraph("Results", styles['Heading3']))
    story.append(t2)
    story.append(Spacer(1, 12))

    if chart_path and os.path.exists(chart_path):
        story.append(Paragraph("Projected FCFs Chart", styles['Heading3']))
        story.append(RLImage(chart_path, width=480, height=300))
        story.append(Spacer(1, 12))

    if wacc_breakdown:
        wb = wacc_breakdown
        story.append(Paragraph("WACC Calculation Breakdown", styles['Heading3']))
        wtab = [
            ["Risk-free rate (US 10Y)", f"{wb.get('risk_free_rate',0):.2%}"],
            ["Market risk premium", f"{wb.get('market_risk_premium',0):.2%}"],
            ["Beta", f"{wb.get('beta',0):.2f}"],
            ["Cost of equity (CAPM)", f"{wb.get('cost_of_equity',0):.2%}"],
            ["Cost of debt (pre-tax)", f"{wb.get('cost_of_debt',0):.2%}"],
            ["Tax rate", f"{wb.get('tax_rate',0):.2%}"],
            ["Weights (Equity / Debt)", f"{wb.get('weights_equity',0):.1%} / {wb.get('weights_debt',0):.1%}"],
            ["Market cap / Total debt", f"${wb.get('market_cap',0):,.0f} / ${wb.get('total_debt',0):,.0f}"],
            ["Final WACC", f"{wb.get('wacc',0):.2%}"]
        ]
        tw = Table(wtab, hAlign='LEFT')
        tw.setStyle(TableStyle([('GRID',(0,0),(-1,-1),0.5,colors.grey)]))
        story.append(tw)
        story.append(Spacer(1, 12))

    story.append(Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
    SimpleDocTemplate(filepath, pagesize=A4).build(story)

# =====================
# Main App
# =====================
def run_dcf_app():
    if "wacc_data" not in st.session_state:
        st.session_state.wacc_data = None
    if "mrp" not in st.session_state:
        st.session_state.mrp = 0.055
    if "rf" not in st.session_state:
        st.session_state.rf = fetch_risk_free_rate_us10y()

    st.title("💸 DCF + WACC Valuation Tool")

    tab_dcf, tab_batch, tab_wacc = st.tabs(["DCF Calculator", "Batch Reports", "WACC Helper"])

    # ------------- DCF TAB -------------
    with tab_dcf:
        with st.sidebar:
            st.header("Controls")
            ticker = st.text_input("Ticker", value="AAPL").strip()
            use_manual_inputs = st.checkbox("Enter FCF manually (override auto fetch)", value=False)

            discount_rate = st.slider("Discount Rate (WACC / required return)",
                                      0.05, 0.20, 0.10, 0.005, key="discount_rate")
            if st.button("Fill from WACC"):
                if st.session_state.wacc_data:
                    st.session_state.discount_rate = float(st.session_state.wacc_data.get("wacc", st.session_state.discount_rate))
                    st.success(f"Filled: {st.session_state.discount_rate:.2%}")
                else:
                    st.warning("No WACC yet. Use the WACC tab.")

            terminal_growth = st.slider("Terminal Growth", 0.00, 0.05, 0.02, 0.005)
            projection_years = st.slider("Projection Years", 3, 20, 10)

            st.header("Growth (Three-Stage)")
            stage1_years = st.slider("Stage 1 Years", 0, 10, 5)
            stage1_growth = st.slider("Stage 1 Growth", -0.20, 0.30, 0.12, 0.005)
            stage2_years = st.slider("Stage 2 Years", 0, 10, 3)
            stage2_growth = st.slider("Stage 2 Growth", -0.10, 0.20, 0.07, 0.005)
            mature_growth = st.slider("Stage 3 (Mature) Growth", -0.02, 0.10, 0.03, 0.005)

            st.header("Buybacks (Share Count)")
            buyback_rate = st.slider("Annual buyback rate (negative = dilution)", -0.10, 0.10, 0.02, 0.005)
            per_share_method = st.selectbox("Per-share method", ["current", "buyback-adjusted"], index=0)

            run = st.button("Run Valuation")

        # auto fetch
        past_fcfs = None
        shares_outstanding = None
        current_price = None
        net_debt = None
        fetch_error = None

        if not use_manual_inputs and ticker:
            try:
                past_fcfs = fetch_financial_data(ticker, want_points=5)
                shares_outstanding = safe_get_shares_outstanding(ticker)
                current_price = safe_get_current_price(ticker)
                net_debt = safe_get_net_debt_from_yfinance(ticker)
            except Exception as e:
                fetch_error = str(e)

        st.subheader("Inputs")
        if fetch_error and not use_manual_inputs:
            st.warning(f"Auto-fetch issue: {fetch_error}")

        if use_manual_inputs or past_fcfs is None:
            st.write("Enter up to 5 most-recent years of **FCF** (most recent first).")
            default_fcfs = [500, 450, 400, 350, 300]
            fcf_values = [st.number_input(f"FCF Year {i+1} (most recent first)", value=float(default_fcfs[i])) for i in range(5)]
            past_fcfs = pd.Series(fcf_values, index=[f"Y-{i}" for i in range(5)], dtype=float)
            shares_outstanding = st.number_input("Shares Outstanding", value=float(100_000_000))
            current_price = st.number_input("Current Share Price", value=150.0)
            net_debt = st.number_input("Net Debt", value=0.0)
        else:
            st.write("Auto-fetched:")
            st.dataframe(pd.DataFrame({"FCF (most recent → older)": past_fcfs.astype(float).values},
                                      index=[f"Y-{i}" for i in range(len(past_fcfs))]))
            st.write(f"**Shares Outstanding**: {shares_outstanding:,}" if shares_outstanding else "**Shares Outstanding**: not found")
            st.write(f"**Current Price**: {current_price}" if current_price is not None else "**Current Price**: not found")
            st.write(f"**Net Debt**: {net_debt:,}" if net_debt is not None else "**Net Debt**: not found")

        with st.expander("Optional overrides"):
            shares_outstanding = st.number_input("Override Shares Outstanding", value=float(shares_outstanding) if shares_outstanding else float(100_000_000))
            net_debt = st.number_input("Override Net Debt", value=float(net_debt) if net_debt is not None else 0.0)

        # build growth curve
        base_growths = [stage1_growth]*int(stage1_years) + [stage2_growth]*int(stage2_years)
        growth_curve = base_growths + [mature_growth]*max(0, int(projection_years)-len(base_growths))
        growth_curve = growth_curve[:int(projection_years)]

        shares_path = project_shares(float(shares_outstanding), float(buyback_rate), int(projection_years)) if per_share_method == "buyback-adjusted" else None

        if run:
            if past_fcfs is None or len(past_fcfs) < 1:
                st.error("Need at least one year of FCF for projection base.")
            elif shares_outstanding is None or shares_outstanding <= 0:
                st.error("Shares outstanding must be provided and > 0.")
            elif discount_rate <= terminal_growth:
                st.error("Discount rate must be greater than terminal growth.")
            else:
                try:
                    last_fcf = float(past_fcfs.iloc[0])
                    results = dcf_valuation_multistage(
                        last_fcf=last_fcf,
                        growth_rates=growth_curve,
                        discount_rate=float(discount_rate),
                        terminal_growth=float(terminal_growth),
                        net_debt=float(net_debt),
                        shares_outstanding=float(shares_outstanding),
                        share_path=shares_path,
                        per_share_method=per_share_method
                    )

                    fv_per_share = results["fair_value_per_share"]
                    st.success(f"**Fair Value per Share:** ${fv_per_share:,.2f}  •  Denominator: {int(results['denominator_shares']):,} shares")
                    if current_price:
                        mos = (fv_per_share - current_price) / current_price
                        st.write(f"**Margin of Safety vs Current Price ({current_price}):** {mos:.2%}")

                    # Chart
                    proj_years = np.arange(1, len(growth_curve) + 1)
                    projected_fcfs = results["projected_fcfs"]
                    pv_each = results["pv_fcfs_each"]
                    chart_path = f"{ticker}_dcf_chart.png"
                    plot_and_save_projection(proj_years, projected_fcfs, pv_each, chart_path)
                    st.image(chart_path, caption="Projected FCFs and Present Values")

                    # PDF
                    st.subheader("📄 Export")
                    pdf_name = f"{ticker}_DCF_Report.pdf"
                    inputs = dict(
                        projection_years=len(growth_curve),
                        discount_rate=float(discount_rate),
                        terminal_growth=float(terminal_growth),
                        net_debt=float(net_debt),
                        per_share_method=per_share_method,
                        buyback_rate=float(buyback_rate),
                        growth_notes=f"Stage1 {stage1_growth:.1%} x {stage1_years} | Stage2 {stage2_growth:.1%} x {stage2_years} | Stage3 {mature_growth:.1%}"
                    )
                    wacc_breakdown = st.session_state.wacc_data if st.session_state.wacc_data and st.session_state.wacc_data.get("ticker") == ticker else None
                    generate_pdf_report(pdf_name, f"{ticker} DCF Valuation", inputs, results, chart_path, wacc_breakdown=wacc_breakdown)
                    with open(pdf_name, "rb") as f:
                        st.download_button("Download PDF Report", data=f, file_name=pdf_name, mime="application/pdf")
                except Exception as e:
                    st.error(f"Valuation failed: {e}")

    # ------------- BATCH TAB -------------
    with tab_batch:
        st.write("Generate **multiple PDF reports**. Paste tickers separated by commas, spaces, or new lines.")
        batch_input = st.text_area("Tickers", value="AAPL\nMSFT\nGOOGL")
        use_wacc = st.checkbox("Auto-calc discount rate using WACC helper (per ticker)", value=True)
        if use_wacc:
            st.caption("Uses your Market Risk Premium setting below. Risk-free uses US 10Y (^TNX).")
        discount_rate_b = st.slider("Fallback Discount Rate", 0.05, 0.20, 0.10, 0.005, key="drb_fallback")
        terminal_growth_b = st.slider("Terminal Growth", 0.00, 0.05, 0.02, 0.005, key="tgb")
        projection_years_b = st.slider("Projection Years", 3, 20, 10, key="pyb")

        stage1_years_b = st.slider("Stage 1 Years", 0, 10, 5, key="s1yb")
        stage1_growth_b = st.slider("Stage 1 Growth", -0.20, 0.30, 0.12, 0.005, key="s1gb")
        stage2_years_b = st.slider("Stage 2 Years", 0, 10, 3, key="s2yb")
        stage2_growth_b = st.slider("Stage 2 Growth", -0.10, 0.20, 0.07, 0.005, key="s2gb")
        stage3_growth_b = st.slider("Stage 3 (Mature) Growth", -0.02, 0.10, 0.03, 0.005, key="s3gb")

        buyback_rate_b = st.slider("Annual buyback rate", -0.10, 0.10, 0.02, 0.005, key="bbrb")
        per_share_method_b = st.selectbox("Per-share method", ["current", "buyback-adjusted"], index=0, key="psmb")

        run_batch = st.button("Generate Batch PDFs")

        if run_batch:
            import zipfile
            tickers = [t.strip() for t in batch_input.replace(",", " ").split() if t.strip()]
            if not tickers:
                st.error("Please enter at least one ticker.")
            else:
                mem_zip = io.BytesIO()
                with zipfile.ZipFile(mem_zip, "w", zipfile.ZIP_DEFLATED) as zf:
                    for tck in tickers:
                        try:
                            past_fcfs = fetch_financial_data(tck, want_points=5)
                            shares_outstanding = safe_get_shares_outstanding(tck) or 100_000_000
                            net_debt = safe_get_net_debt_from_yfinance(tck) or 0.0

                            base_growths = [stage1_growth_b]*int(stage1_years_b) + [stage2_growth_b]*int(stage2_years_b)
                            growth_curve = base_growths + [stage3_growth_b] * max(0, int(projection_years_b) - len(base_growths))
                            growth_curve = growth_curve[:int(projection_years_b)]

                            shares_path = project_shares(float(shares_outstanding), float(buyback_rate_b), int(projection_years_b)) if per_share_method_b == "buyback-adjusted" else None

                            wacc_data = None
                            if use_wacc:
                                try:
                                    wacc_data = compute_wacc(tck, risk_free_rate=None, market_risk_premium=st.session_state.mrp)
                                    dr_use = float(wacc_data["wacc"])
                                except Exception:
                                    dr_use = float(discount_rate_b)
                            else:
                                dr_use = float(discount_rate_b)

                            results = dcf_valuation_multistage(
                                last_fcf=float(past_fcfs.iloc[0]),
                                growth_rates=growth_curve,
                                discount_rate=float(dr_use),
                                terminal_growth=float(terminal_growth_b),
                                net_debt=float(net_debt),
                                shares_outstanding=float(shares_outstanding),
                                share_path=shares_path,
                                per_share_method=per_share_method_b
                            )

                            # Chart
                            proj_years = np.arange(1, len(growth_curve) + 1)
                            chart_path = f"{tck}_dcf_chart.png"
                            plt.figure()
                            plt.plot(proj_years, results["projected_fcfs"], label="Projected FCF")
                            plt.plot(proj_years, results["pv_fcfs_each"], label="Present Value of FCF")
                            plt.xlabel("Year"); plt.ylabel("Amount"); plt.title("Projected FCFs and Present Values"); plt.legend(); plt.tight_layout()
                            plt.savefig(chart_path); plt.close()

                            # PDF
                            pdf_name = f"{tck}_DCF_Report.pdf"
                            inputs = dict(
                                projection_years=len(growth_curve),
                                discount_rate=float(dr_use),
                                terminal_growth=float(terminal_growth_b),
                                net_debt=float(net_debt),
                                per_share_method=per_share_method_b,
                                buyback_rate=float(buyback_rate_b),
                                growth_notes=f"Stage1 {stage1_growth_b:.1%} x {stage1_years_b} | Stage2 {stage2_growth_b:.1%} x {stage2_years_b} | Stage3 {stage3_growth_b:.1%}"
                            )
                            generate_pdf_report(pdf_name, f"{tck} DCF Valuation", inputs, results, chart_path, wacc_breakdown=wacc_data)
                            with open(pdf_name, "rb") as pf:
                                zf.writestr(pdf_name, pf.read())
                        except Exception as e:
                            zf.writestr(f"{tck}_ERROR.txt", str(e))

                mem_zip.seek(0)
                st.download_button("Download ZIP of PDFs", data=mem_zip, file_name="dcf_reports.zip", mime="application/zip")

    # ------------- WACC TAB -------------
    with tab_wacc:
        st.subheader("WACC Helper (auto)")
        ticker_w = st.text_input("Ticker for WACC", value="AAPL").strip()

        colA, colB = st.columns(2)
        with colA:
            rf_in = st.number_input("Risk-free rate (US 10Y)", value=float(st.session_state.rf), format="%.4f")
        with colB:
            mrp_in = st.number_input("Market risk premium", value=float(st.session_state.mrp), format="%.4f")

        try:
            wacc_data = compute_wacc(ticker_w, risk_free_rate=rf_in, market_risk_premium=mrp_in)
            wacc_data["ticker"] = ticker_w
            st.session_state.wacc_data = wacc_data
            st.session_state.rf = rf_in
            st.session_state.mrp = mrp_in

            col1, col2 = st.columns([2,1])
            with col1:
                df = pd.DataFrame({
                    "Metric": [
                        "Risk-free rate (US 10Y)","Market risk premium","Beta",
                        "Cost of equity (CAPM)","Cost of debt (pre-tax)","Tax rate",
                        "Weights - Equity","Weights - Debt","Market Cap","Total Debt","Final WACC"
                    ],
                    "Value": [
                        f"{wacc_data['risk_free_rate']:.2%}", f"{wacc_data['market_risk_premium']:.2%}", f"{wacc_data['beta']:.2f}",
                        f"{wacc_data['cost_of_equity']:.2%}", f"{wacc_data['cost_of_debt']:.2%}", f"{wacc_data['tax_rate']:.2%}",
                        f"{wacc_data['weights_equity']:.1%}", f"{wacc_data['weights_debt']:.1%}", f"${wacc_data['market_cap']:,.0f}", f"${wacc_data['total_debt']:,.0f}", f"{wacc_data['wacc']:.2%}"
                    ]
                })
                st.dataframe(df, use_container_width=True)
            with col2:
                color = wacc_color(wacc_data["wacc"])
                st.markdown(f"### WACC: <span style='color:{color};font-weight:700'>{wacc_data['wacc']:.2%}</span>", unsafe_allow_html=True)
                if st.button("Use in DCF"):
                    st.session_state.discount_rate = float(wacc_data["wacc"])
                    st.success("Filled DCF discount rate from WACC. Go to DCF tab.")
        except Exception as e:
            st.error(f"WACC calculation failed: {e}")
