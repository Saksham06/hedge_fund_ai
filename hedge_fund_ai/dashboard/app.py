"""
Elite Dashboard — 8 tabs including AI Q&A.
"""
import json
import os
import time

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="AI Hedge Fund", layout="wide", page_icon="📈")
st.markdown("""
<style>
  div[data-testid="metric-container"] { background:#111827; border-radius:8px; padding:.6rem 1rem; }
  .verdict-good  { background:#14532d; color:#86efac; border-radius:6px; padding:.5rem 1rem; font-weight:600; }
  .verdict-warn  { background:#713f12; color:#fde68a; border-radius:6px; padding:.5rem 1rem; font-weight:600; }
  .verdict-bad   { background:#7f1d1d; color:#fca5a5; border-radius:6px; padding:.5rem 1rem; font-weight:600; }
  .claim-pass    { color:#86efac; }
  .claim-fail    { color:#fca5a5; }
  .qa-bubble     { background:#1e293b; border-radius:8px; padding:1rem; margin:.5rem 0; }
  .qa-answer     { background:#0f172a; border-radius:8px; padding:1rem; margin:.5rem 0; border-left:3px solid #6366f1; }
</style>
""", unsafe_allow_html=True)

_STATE = os.path.join(os.path.dirname(__file__), "..", "state")

@st.cache_data(ttl=60)
def load_perf():
    try:
        from hedge_fund_ai.state.performance import load_performance
        return load_performance() or []
    except Exception:
        return []

@st.cache_data(ttl=300)
def load_equity():
    try:
        from hedge_fund_ai.state.backtest import load_equity_curve
        return load_equity_curve()
    except Exception:
        return None

@st.cache_data(ttl=60)
def load_positions():
    try:
        from hedge_fund_ai.state.positions import load_prev_portfolio
        return load_prev_portfolio() or []
    except Exception:
        return []

@st.cache_data(ttl=120)
def load_ic_history():
    try:
        with open(os.path.join(_STATE, "ic_history.json")) as f:
            return json.load(f)
    except Exception:
        return {}

@st.cache_data(ttl=120)
def load_audit():
    try:
        with open(os.path.join(_STATE, "audit_log.json")) as f:
            return json.load(f)
    except Exception:
        return []

@st.cache_data(ttl=120)
def load_signal_log():
    try:
        with open(os.path.join(_STATE, "signal_log.json")) as f:
            return json.load(f)
    except Exception:
        return []

@st.cache_data(ttl=60)
def load_risk_state():
    try:
        with open(os.path.join(_STATE, "risk_state.json")) as f:
            return json.load(f)
    except Exception:
        return {"entry_prices": {}, "halted_until": None}

@st.cache_data(ttl=120)
def load_metrics():
    try:
        with open(os.path.join(_STATE, "last_metrics.json")) as f:
            return json.load(f)
    except Exception:
        return {}

@st.cache_data(ttl=120)
def load_query_audit():
    """Load Q&A query audit log from SQLite."""
    try:
        from hedge_fund_ai.llm.metrics_cache import _conn
        with _conn() as c:
            rows = c.execute("""
                SELECT timestamp, question_type, question_text,
                       confidence, validation_pass, response_length
                FROM query_audit ORDER BY timestamp DESC LIMIT 100
            """).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️  Controls")
    lookback  = st.selectbox("Window", ["All","252d","126d","63d","21d"])
    show_raw  = st.toggle("Show raw data", False)
    st.divider()
    if st.button("🔄 Refresh"):
        st.cache_data.clear()
        st.rerun()
    st.caption("Live: 60s | Backtest: 5min")

st.title("📈 Elite AI Hedge Fund")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📊 Overview","🔁 Backtest","💼 Portfolio",
    "🧠 Signal IC","📐 Factors","🛡️ Risk",
    "🤖 AI Q&A","📋 Audit"
])
overview_tab,bt_tab,port_tab,ic_tab,factor_tab,risk_tab,qa_tab,audit_tab = tabs

# ══════════════════════════════════════════════════════════════════════
# TAB 1: OVERVIEW
# ══════════════════════════════════════════════════════════════════════
with overview_tab:
    perf = load_perf()
    if not perf:
        st.info("No data yet — run `python -m hedge_fund_ai.run` first.")
    else:
        df = pd.DataFrame(perf)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        df["ret"]            = df["portfolio_value"].pct_change().fillna(0)
        df["cum_ret"]        = (1+df["ret"]).cumprod()-1
        df["drawdown"]       = df["portfolio_value"]/df["portfolio_value"].cummax()-1
        df["rolling_sharpe"] = (df["ret"].rolling(20).mean()/(df["ret"].rolling(20).std()+1e-9))*np.sqrt(252)

        lb = {"252d":252,"126d":126,"63d":63,"21d":21}
        view = df.tail(lb[lookback]) if lookback!="All" else df

        ret=df["ret"]; latest=float(df["portfolio_value"].iloc[-1]); start=float(df["portfolio_value"].iloc[0])
        total_ret=latest/start-1; max_dd=float(df["drawdown"].min())
        sharpe=float(ret.mean()/(ret.std()+1e-9)*np.sqrt(252))
        sortino=float(ret.mean()/(ret[ret<0].std()+1e-9)*np.sqrt(252))
        cagr=float((1+total_ret)**(252/max(len(df),1))-1)
        win_rate=float((ret>0).mean()); daily=float(ret.iloc[-1]) if len(ret)>1 else 0

        c1,c2,c3,c4,c5,c6,c7=st.columns(7)
        c1.metric("Value",   f"${latest:,.0f}", f"{total_ret*100:+.2f}%")
        c2.metric("Sharpe",  f"{sharpe:.2f}")
        c3.metric("Sortino", f"{sortino:.2f}")
        c4.metric("Max DD",  f"{max_dd*100:.1f}%")
        c5.metric("CAGR",    f"{cagr*100:.1f}%")
        c6.metric("WinRate", f"{win_rate*100:.1f}%")
        c7.metric("Today",   f"{daily*100:+.2f}%")
        st.divider()
        l,r=st.columns(2)
        with l:
            st.subheader("NAV"); st.line_chart(view.set_index("timestamp")["portfolio_value"],height=240)
            st.subheader("Drawdown"); st.area_chart(view.set_index("timestamp")["drawdown"],height=200)
        with r:
            st.subheader("Cumulative Return"); st.area_chart(view.set_index("timestamp")["cum_ret"],height=240)
            st.subheader("Rolling 20d Sharpe"); st.line_chart(view.set_index("timestamp")["rolling_sharpe"],height=200)

# ══════════════════════════════════════════════════════════════════════
# TAB 2: BACKTEST
# ══════════════════════════════════════════════════════════════════════
with bt_tab:
    equity_data=load_equity(); metrics=load_metrics()
    if not equity_data:
        st.info("No backtest curve yet.")
    else:
        eq=pd.Series(equity_data); r=eq.pct_change().dropna(); dd=eq/eq.cummax()-1
        bt_sharpe=float(r.mean()/(r.std()+1e-9)*np.sqrt(252))
        bt_sortino=float(r.mean()/(r[r<0].std()+1e-9)*np.sqrt(252))
        bt_total=float(eq.iloc[-1]/eq.iloc[0]-1); bt_dd=float(dd.min())
        years=len(eq)/252; bt_cagr=float((1+bt_total)**(1/max(years,.01))-1)
        bt_wr=float((r>0).mean())
        c1,c2,c3,c4,c5,c6=st.columns(6)
        c1.metric("Sharpe",  f"{bt_sharpe:.2f}"); c2.metric("Sortino",f"{bt_sortino:.2f}")
        c3.metric("Max DD",  f"{bt_dd*100:.1f}%"); c4.metric("Return",f"{bt_total*100:.1f}%")
        c5.metric("CAGR",    f"{bt_cagr*100:.1f}%"); c6.metric("WinRate",f"{bt_wr*100:.1f}%")

        # Benchmark comparison
        if metrics:
            st.divider()
            bc1,bc2,bc3=st.columns(3)
            bc1.metric("vs SPY",    f"{metrics.get('excess_vs_spy',0)*100:+.1f}%")
            bc2.metric("vs EW",     f"{metrics.get('excess_vs_ew',0)*100:+.1f}%")
            bc3.metric("Alpha Ann", f"{metrics.get('alpha_ann',0)*100:+.1f}%")

        # Placebo + Monte Carlo
        placebo=metrics.get("placebo_test",{}); mc=metrics.get("monte_carlo",{})
        if placebo or mc:
            st.divider()
            st.subheader("🎲 Signal Validation")
            pc1,pc2=st.columns(2)
            with pc1:
                st.write("**Placebo Test**")
                if placebo:
                    p=placebo.get("p_value",1); color="verdict-good" if p<.05 else ("verdict-warn" if p<.1 else "verdict-bad")
                    st.markdown(f'<div class="{color}">{placebo.get("verdict","")}</div>',unsafe_allow_html=True)
                    st.metric("p-value",f"{p:.3f}"); st.metric("Random Sharpe",placebo.get("random_mean","?"))
                else:
                    st.info("Not run yet")
            with pc2:
                st.write("**Monte Carlo (500 sims)**")
                if mc:
                    v=mc.get("verdict",""); color="verdict-good" if "ROBUST" in v else ("verdict-warn" if "ACCEPTABLE" in v else "verdict-bad")
                    st.markdown(f'<div class="{color}">{v}</div>',unsafe_allow_html=True)
                    st.metric("P(Sharpe>1)",f"{mc.get('p_sharpe_gt_1',0):.0%}")
                    st.metric("5th pct CAGR",f"{mc.get('cagr_p5',0)*100:.1f}%")
                else:
                    st.info("Not run yet")

        l,r=st.columns(2)
        with l: st.subheader("Equity"); st.line_chart(eq,height=260)
        with r: st.subheader("Drawdown"); st.area_chart(dd,height=260)

# ══════════════════════════════════════════════════════════════════════
# TAB 3: PORTFOLIO
# ══════════════════════════════════════════════════════════════════════
with port_tab:
    positions=load_positions()
    if not positions:
        st.info("No positions yet.")
    else:
        pos_df=pd.DataFrame(positions)
        pos_df["weight_pct"]=pd.to_numeric(pos_df.get("weight",0),errors="coerce").fillna(0)*100
        pos_df=pos_df.sort_values("weight_pct",ascending=False)
        c1,c2,c3=st.columns(3)
        c1.metric("Positions",len(pos_df))
        c2.metric("Gross Exposure",f"{pos_df['weight_pct'].sum():.1f}%")
        c3.metric("Sectors",pos_df["sector"].nunique() if "sector" in pos_df.columns else "—")
        st.dataframe(pos_df[["ticker","weight_pct","sector","price"]].rename(columns={"weight_pct":"Weight %"}),use_container_width=True,hide_index=True)
        l,r=st.columns(2)
        with l: st.subheader("Weight"); st.bar_chart(pos_df.set_index("ticker")["weight_pct"],height=260)
        with r:
            if "sector" in pos_df.columns:
                st.subheader("Sectors"); st.bar_chart(pos_df.groupby("sector")["weight_pct"].sum().sort_values(ascending=False),height=260)

# ══════════════════════════════════════════════════════════════════════
# TAB 4: SIGNAL IC
# ══════════════════════════════════════════════════════════════════════
with ic_tab:
    st.subheader("🔬 Live Signal Validation")
    sl=load_signal_log(); filled=[r for r in sl if r.get("filled") and r.get("ic") is not None]
    unfilled=[r for r in sl if not r.get("filled")]
    if filled:
        ics=[r["ic"] for r in filled]; mean_ic=float(np.mean(ics)); std_ic=float(np.std(ics))+1e-9
        ic_ir=mean_ic/std_ic; hit_rate=float(np.mean([ic>0 for ic in ics]))
        if len(filled)>=12:
            if mean_ic>0.05 and ic_ir>0.5 and hit_rate>0.55: vc,vt="verdict-good",f"✅ DEPLOY — IC={mean_ic:.3f} IR={ic_ir:.2f} hit={hit_rate*100:.0f}%"
            elif mean_ic>0: vc,vt="verdict-warn",f"⚠️  PAPER — IC={mean_ic:.3f}"
            else: vc,vt="verdict-bad",f"❌ NO ALPHA — IC={mean_ic:.3f}"
        else: vc,vt="verdict-warn",f"🕐 EARLY — {len(filled)}/12 periods"
        st.markdown(f'<div class="{vc}">{vt}</div>',unsafe_allow_html=True)
        st.divider()
        c1,c2,c3,c4,c5=st.columns(5)
        c1.metric("Mean IC",f"{mean_ic:.4f}"); c2.metric("IC-IR",f"{ic_ir:.3f}")
        c3.metric("Hit Rate",f"{hit_rate*100:.1f}%"); c4.metric("Filled",len(filled)); c5.metric("Pending",len(unfilled))
        st.subheader("IC Over Time")
        st.line_chart(pd.Series([r["ic"] for r in filled],index=[r["signal_date"] for r in filled]),height=200)
    else:
        st.warning(f"0 periods filled. {len(unfilled)} signals logged — IC fills after 21 trading days.")

# ══════════════════════════════════════════════════════════════════════
# TAB 5: FACTOR ANALYTICS
# ══════════════════════════════════════════════════════════════════════
with factor_tab:
    ic_data=load_ic_history()
    if not ic_data:
        st.info("No IC history yet.")
    else:
        rows=[]
        for factor,vals in sorted(ic_data.items()):
            arr=np.array(vals[-12:],dtype=float)
            rows.append({"Factor":factor,"Mean IC":round(float(np.mean(arr)),4),"Std":round(float(np.std(arr)),4),
                         "IC-IR":round(float(np.mean(arr)/(np.std(arr)+1e-6)),3),"Hit%":f"{np.mean(arr>0)*100:.0f}%","n":len(arr)})
        ic_df=pd.DataFrame(rows).sort_values("IC-IR",ascending=False)
        st.dataframe(ic_df,use_container_width=True,hide_index=True)
        st.subheader("Mean IC by Factor"); st.bar_chart(ic_df.set_index("Factor")["Mean IC"].sort_values(),height=280)
        sel=st.selectbox("Factor time series",sorted(ic_data.keys()))
        if sel:
            s=pd.Series(ic_data[sel]); st.line_chart(s,height=180)
            st.caption(f"Mean={s.mean():.3f} IR={s.mean()/(s.std()+1e-6):.2f}")

# ══════════════════════════════════════════════════════════════════════
# TAB 6: RISK CONTROLS
# ══════════════════════════════════════════════════════════════════════
with risk_tab:
    st.subheader("🛡️ Risk State")
    rs=load_risk_state(); halted=rs.get("halted_until"); reason=rs.get("halt_reason","")
    if halted:
        halt_ts=pd.Timestamp(halted,tz="UTC")
        if pd.Timestamp.now(tz="UTC")<halt_ts:
            st.markdown(f'<div class="verdict-bad">🚨 TRADING HALTED — {reason}<br>Until: {halted}</div>',unsafe_allow_html=True)
        else: st.success("✅ Trading active (halt expired)")
    else: st.success("✅ Trading active")
    ep=rs.get("entry_prices",{})
    if ep:
        st.subheader("Stop-Loss Tracker")
        st.dataframe(pd.DataFrame([{"Ticker":t,"Entry":f"${v['price']:.2f}","Date":v.get("date","")} for t,v in ep.items()]),use_container_width=True,hide_index=True)
    with st.expander("📋 Thresholds"):
        st.markdown("""
| Control | Threshold | Action |
|---|---|---|
| Position stop | -8% from entry | Close immediately |
| Daily loss | -3% portfolio | Halt 24h |
| Max drawdown | -20% from peak | Halt 30 days |
""")

# ══════════════════════════════════════════════════════════════════════
# TAB 7: AI Q&A  ← Phase 4 integration
# ══════════════════════════════════════════════════════════════════════
with qa_tab:
    st.subheader("🤖 Ask the AI Fund Manager")
    st.caption("Grounded in live metrics — every number is verified against the metrics cache.")

    # Session ID per browser session
    if "qa_session" not in st.session_state:
        import uuid
        st.session_state.qa_session = str(uuid.uuid4())[:8]

    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []

    # Example questions
    with st.expander("💡 Example questions"):
        example_cols = st.columns(2)
        examples = [
            "Why did the portfolio underperform last week?",
            "Which factor is working best right now?",
            "What is our current drawdown and risk exposure?",
            "Should we deploy real capital yet?",
            "What happened on the last rebalance?",
            "Which factors have been auto-disabled?",
        ]
        for i, ex in enumerate(examples):
            col = example_cols[i % 2]
            if col.button(ex, key=f"ex_{i}", use_container_width=True):
                st.session_state.qa_pending = ex

    # Input
    question = st.chat_input("Ask about P&L, factors, risk, or deployment readiness...")
    if "qa_pending" in st.session_state:
        question = st.session_state.pop("qa_pending")

    if question:
        with st.spinner("Thinking..."):
            try:
                from hedge_fund_ai.llm.qa_engine import answer_question
                result = answer_question(
                    question=question,
                    session_id=st.session_state.qa_session,
                )
                st.session_state.qa_history.append({
                    "question": question,
                    "result":   result,
                })
            except Exception as e:
                st.session_state.qa_history.append({
                    "question": question,
                    "result":   {
                        "response": f"Error: {e}",
                        "question_type": "error",
                        "confidence": 0.0,
                        "validation_passed": False,
                        "claim_checks": [],
                        "cache_hit": False,
                        "elapsed_ms": 0,
                    },
                })

    # Display conversation history (newest first)
    for turn in reversed(st.session_state.qa_history[-10:]):
        q   = turn["question"]
        res = turn["result"]

        # User bubble
        st.markdown(f'<div class="qa-bubble">👤 {q}</div>', unsafe_allow_html=True)

        # Answer bubble
        with st.container():
            st.markdown(f'<div class="qa-answer">', unsafe_allow_html=True)

            # Metadata row
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.caption(f"Type: `{res.get('question_type','?')}`")
            mc2.caption(f"Confidence: {res.get('confidence',0):.0%}")
            mc3.caption(f"✅ Valid" if res.get("validation_passed") else "⚠️ Unverified")
            mc4.caption(f"{'⚡ Cached' if res.get('cache_hit') else f'{res.get(\"elapsed_ms\",0):.0f}ms'}")

            # Response
            st.markdown(res.get("response", "No response"))

            # Claim checks (expandable)
            checks = [c for c in res.get("claim_checks", []) if c.get("actual") is not None]
            if checks:
                with st.expander(f"🔍 Claim verification ({len(checks)} checked)"):
                    for c in checks:
                        icon = "✅" if c["passed"] else "❌"
                        delta_str = f" Δ{c['delta_pct']:.1f}%" if c.get("delta_pct") is not None else ""
                        st.caption(f"{icon} `{c['claim']}`: LLM={c.get('claimed','?')} actual={c.get('actual','?')}{delta_str}")

            st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.qa_history:
        if st.button("🗑️ Clear conversation"):
            st.session_state.qa_history = []
            from hedge_fund_ai.llm.qa_engine import _memory
            _memory.clear(st.session_state.qa_session)
            st.rerun()

# ══════════════════════════════════════════════════════════════════════
# TAB 8: AUDIT LOG
# ══════════════════════════════════════════════════════════════════════
with audit_tab:
    audit_data=load_audit()
    if audit_data:
        st.subheader(f"Rebalance Audit — {len(audit_data)} periods")
        rows=[]
        for r in audit_data:
            rows.append({"Date":r.get("date"),"Regime":r.get("regime"),"Univ":r.get("universe_size"),
                         "Pos":len(r.get("final_weights",{})),"RawTurn":r.get("raw_turnover"),
                         "FiltTurn":r.get("filtered_turnover"),"Cost bps":round(r.get("total_cost_frac",0)*10000,2),
                         "DD Scale":r.get("dd_scale"),
                         "Period Ret":f"{r.get('period_return',0)*100:+.3f}%" if r.get("period_return") is not None else "—"})
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
        if show_raw:
            sel=st.selectbox("Inspect",  [r["date"] for r in audit_data])
            sel_r=next((r for r in audit_data if r["date"]==sel),None)
            if sel_r: st.json(sel_r)

    # Query audit log (Phase 5)
    st.divider()
    st.subheader("🔍 Q&A Query Audit (Compliance)")
    qlog=load_query_audit()
    if qlog:
        qdf=pd.DataFrame(qlog)
        qdf["validation_pass"]=qdf["validation_pass"].map({1:"✅",0:"❌"})
        st.dataframe(qdf,use_container_width=True,hide_index=True)
        st.caption(f"{len(qlog)} queries logged. Exported monthly for compliance review.")
    else:
        st.info("No Q&A queries logged yet — start asking questions in the AI Q&A tab.")
