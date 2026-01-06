from __future__ import annotations

import time
import json
from typing import Dict, Optional, List, Tuple

import streamlit as st
import pandas as pd
import altair as alt
import requests

from polymarket_api import (
    normalize_address,
    resolve_market,
    data_api_get_positions,
    clob_get_mid_price,
    PMError,
)

from analytics import (
    trades_to_df,
    format_money,
    cashflows,
    net_shares_by_outcome,
    avg_buy_price_by_outcome,
    max_exposure,
    pair_pnl_open_estimate,
    build_time_axis,
    running_vwap,
    max_profit_loss_curves,
)

st.set_page_config(page_title="Polymarket Wallet Analyzer", layout="wide")
st.title("Polymarket Wallet Trade Analyzer — Report")

TRADES_URL = "https://data-api.polymarket.com/trades"
PRICE_RESOLUTION_THRESHOLD = 0.5  # like your script


# ----------------------------
# Navigation / Session
# ----------------------------
if "page" not in st.session_state:
    st.session_state.page = "market"  # market | profile

# Optional query-param navigation (so cards can deep-link into the market analyzer)
try:
    q = st.query_params  # streamlit >= 1.32
    if "page" in q and str(q.get("page", "")).strip():
        st.session_state.page = str(q.get("page")).strip()
    if "wallet" in q and str(q.get("wallet", "")).strip():
        st.session_state.prefill_wallet = str(q.get("wallet")).strip()
    if "market" in q and str(q.get("market", "")).strip():
        st.session_state.prefill_market = str(q.get("market")).strip()
except Exception:
    # Back-compat (older streamlit)
    try:
        qp = st.experimental_get_query_params()
        if "page" in qp and qp["page"]:
            st.session_state.page = str(qp["page"][0]).strip()
        if "wallet" in qp and qp["wallet"]:
            st.session_state.prefill_wallet = str(qp["wallet"][0]).strip()
        if "market" in qp and qp["market"]:
            st.session_state.prefill_market = str(qp["market"][0]).strip()
    except Exception:
        pass

with st.sidebar:
    st.markdown("### Views")
    cnav1, cnav2 = st.columns(2)
    with cnav1:
        if st.button("Market Analyzer", use_container_width=True):
            st.session_state.page = "market"
            try:
                st.query_params["page"] = "market"
            except Exception:
                try:
                    st.experimental_set_query_params(page="market")
                except Exception:
                    pass
            st.rerun()
    with cnav2:
        if st.button("Trader Profile", use_container_width=True):
            st.session_state.page = "profile"
            try:
                st.query_params["page"] = "profile"
            except Exception:
                try:
                    st.experimental_set_query_params(page="profile")
                except Exception:
                    pass
            st.rerun()


# ----------------------------
# Helpers
# ----------------------------
def norm_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.strip().lower() in ("true", "1", "yes")
    if isinstance(x, (int, float)):
        return x != 0
    return False


def compute_state(market: dict) -> Tuple[bool, bool]:
    status = str(market.get("status") or "").lower()
    resolved = norm_bool(market.get("resolved")) or norm_bool(market.get("isResolved"))
    closed_raw = norm_bool(market.get("closed"))
    active_raw = norm_bool(market.get("active"))

    is_closed = closed_raw or resolved or status in ("closed", "resolved", "finalized", "settled")
    is_active = active_raw and (not is_closed)
    return is_active, is_closed


def market_start_end_ts(market: dict) -> Tuple[Optional[int], Optional[int]]:
    for a, b in [("startTime", "endTime"), ("startTimestamp", "endTimestamp"), ("start", "end")]:
        s = market.get(a)
        e = market.get(b)
        try:
            if s is not None and e is not None:
                return int(s), int(e)
        except Exception:
            pass
    return None, None


def parse_outcomes(market: dict) -> List[str]:
    out = market.get("outcomes")
    if out is None:
        return []
    if isinstance(out, list):
        if len(out) == 0:
            return []
        if isinstance(out[0], str):
            return [str(x) for x in out]
        if isinstance(out[0], dict):
            res = []
            for d in out:
                res.append(str(d.get("name") or d.get("title") or d.get("outcome") or ""))
            return [x for x in res if x]
        return [str(x) for x in out]
    if isinstance(out, str):
        s = out.strip()
        try:
            arr = json.loads(s)
            if isinstance(arr, list):
                return [str(x) for x in arr]
        except Exception:
            pass
        if "," in s:
            return [p.strip() for p in s.split(",") if p.strip()]
    return []


def resolve_winner_label(market: dict) -> Optional[str]:
    outcomes = parse_outcomes(market)
    for k in ["resolvedOutcome", "winningOutcome", "resolution", "result", "winner"]:
        v = market.get(k)
        if isinstance(v, str) and v.strip():
            try:
                idx = int(v)
                if 0 <= idx < len(outcomes):
                    return outcomes[idx]
            except Exception:
                pass
            return v.strip()
    for k in ["resolvedOutcomeIndex", "winningOutcomeIndex", "resultIndex"]:
        v = market.get(k)
        if v is None:
            continue
        try:
            idx = int(v)
            if 0 <= idx < len(outcomes):
                return outcomes[idx]
        except Exception:
            pass
    return None


def winner_to_outcome_label(winner: Optional[str]) -> Optional[str]:
    """
    Convert a "winner" string into our df outcome labels: "Up" / "Down".
    Accepts YES/NO, Up/Down, etc.
    """
    if not winner:
        return None
    w = str(winner).strip().lower()
    if w in ("up", "yes", "true", "1"):
        return "Up"
    if w in ("down", "no", "false", "0"):
        return "Down"
    if "up" in w or "yes" in w:
        return "Up"
    if "down" in w or "no" in w:
        return "Down"
    return None


# =========================================================
# ✅ Trades fetch EXACTLY like your working script, robust:
# - 500 per page (server cap)
# - offset jumps by 500
# - stop only on empty batch
# - dedupe by transactionHash
# =========================================================
def fetch_trades_complete(condition_id: str, user_address: str) -> List[dict]:
    all_trades: List[dict] = []
    seen: set[str] = set()

    limit = 500
    offset = 0

    while True:
        params = {
            "limit": limit,
            "offset": offset,
            "takerOnly": "false",
            "market": condition_id,
            "user": user_address,
        }

        resp = requests.get(TRADES_URL, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()

        if isinstance(data, dict):
            batch = data.get("trades", [])
        elif isinstance(data, list):
            batch = data
        else:
            batch = []

        if not batch:
            break

        new_count = 0
        for t in batch:
            tx = t.get("transactionHash")
            if not tx:
                continue
            if tx in seen:
                continue
            seen.add(tx)
            all_trades.append(t)
            new_count += 1

        offset += limit

        # Safety: if backend repeats pages endlessly
        if new_count == 0:
            break

    return all_trades


def mid_prices_for_assets(df: pd.DataFrame, cap: int = 30) -> Dict[str, Optional[float]]:
    mids: Dict[str, Optional[float]] = {}
    assets = [str(a) for a in df["asset"].dropna().unique().tolist()][:cap]
    for a in assets:
        mids[a] = clob_get_mid_price(a)
        time.sleep(0.03)
    return mids


# ----------------------------
# Charts
# ----------------------------
def add_series_line(df_line: pd.DataFrame, tooltip_label: str):
    line = (
        alt.Chart(df_line)
        .mark_line(strokeWidth=4)
        .encode(
            x=alt.X("min_from_start:Q", scale=alt.Scale(domain=[0, 15])),
            y=alt.Y("value:Q"),
            color=alt.Color("series:N", legend=None),
        )
    )

    hover_pts = (
        alt.Chart(df_line)
        .mark_point(size=260, opacity=0)
        .encode(
            x="min_from_start:Q",
            y="value:Q",
            color=alt.Color("series:N", legend=None),
            tooltip=[
                alt.Tooltip("min_from_start:Q", title="Minute"),
                alt.Tooltip("value:Q", title=tooltip_label),
            ],
        )
    )
    return alt.layer(line, hover_pts)


def price_chart(df: pd.DataFrame, show_points: bool, only_buys_points: bool,
                show_vwap_all_buys: bool, show_vwap_all_trades: bool,
                show_vwap_up: bool, show_vwap_down: bool):

    layers = []

    dpts = df.copy()
    if only_buys_points:
        dpts = dpts[dpts["side"] == "BUY"].copy()

    def _color(o: str) -> str:
        if str(o).lower() == "up":
            return "#22c55e"
        if str(o).lower() == "down":
            return "#3b82f6"
        return "#9ca3af"

    def _shape(o: str) -> str:
        if str(o).lower() == "up":
            return "triangle-up"
        if str(o).lower() == "down":
            return "triangle-down"
        return "circle"

    dpts["color"] = dpts["outcome"].apply(_color)
    dpts["shape"] = dpts["outcome"].apply(_shape)

    if show_points:
        points = (
            alt.Chart(dpts)
            .mark_point(filled=True, size=70)
            .encode(
                x=alt.X("min_from_start:Q", title="Minutes (0 → 15)", scale=alt.Scale(domain=[0, 15])),
                y=alt.Y("price:Q", title="Price", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("color:N", scale=None, legend=None),
                shape=alt.Shape("shape:N", legend=None),
                tooltip=[
                    alt.Tooltip("min_from_start:Q", title="Minute"),
                    alt.Tooltip("side:N", title="Side"),
                    alt.Tooltip("outcome:N", title="Outcome"),
                    alt.Tooltip("price:Q", title="Price"),
                    alt.Tooltip("size:Q", title="Size"),
                    alt.Tooltip("notional:Q", title="Notional"),
                ],
            )
        )
        layers.append(points)

    def add_vwap(name: str, v: pd.DataFrame):
        if v.empty:
            return
        tmp = v.rename(columns={"vwap": "value"}).copy()
        tmp["series"] = name
        layers.append(add_series_line(tmp[["min_from_start", "value", "series"]], name))

    if show_vwap_all_buys:
        add_vwap("VWAP ALL (buys)", running_vwap(df, outcome=None, only_buys=True))
    if show_vwap_all_trades:
        add_vwap("VWAP ALL (all trades)", running_vwap(df, outcome=None, only_buys=False))
    if show_vwap_up:
        add_vwap("VWAP UP", running_vwap(df, outcome="Up", only_buys=True))
    if show_vwap_down:
        add_vwap("VWAP DOWN", running_vwap(df, outcome="Down", only_buys=True))

    if not layers:
        return None

    ch = alt.layer(*layers).properties(height=420).interactive()
    ch = ch.encode(
        color=alt.Color(
            "series:N",
            scale=alt.Scale(
                domain=["VWAP ALL (buys)", "VWAP ALL (all trades)", "VWAP UP", "VWAP DOWN"],
                range=["#ffffff", "#a3a3a3", "#22c55e", "#3b82f6"],
            ),
            legend=alt.Legend(title="Lines"),
        )
    )
    return ch


def pnl_chart(df: pd.DataFrame, show_max_profit: bool, show_max_loss: bool):
    curves = max_profit_loss_curves(df)
    if curves.empty:
        return None

    layers = []
    if show_max_profit:
        tmp = curves[["min_from_start", "max_profit"]].rename(columns={"max_profit": "value"}).copy()
        tmp["series"] = "Max Profit"
        layers.append(add_series_line(tmp[["min_from_start", "value", "series"]], "Max Profit ($)"))
    if show_max_loss:
        tmp = curves[["min_from_start", "max_loss"]].rename(columns={"max_loss": "value"}).copy()
        tmp["series"] = "Max Loss"
        layers.append(add_series_line(tmp[["min_from_start", "value", "series"]], "Max Loss ($)"))

    if not layers:
        return None

    ch = alt.layer(*layers).properties(height=220).interactive()
    ch = ch.encode(
        color=alt.Color(
            "series:N",
            scale=alt.Scale(domain=["Max Profit", "Max Loss"], range=["#f59e0b", "#ef4444"]),
            legend=alt.Legend(title="PnL Bands"),
        )
    )
    ch = ch.encode(y=alt.Y("value:Q", title="PnL ($)"))
    return ch


# ----------------------------
# Exposure USD & Shares charts (kept as requested)
# ----------------------------
def exposure_curves_net(df: pd.DataFrame) -> pd.DataFrame:
    d = df.sort_values("timestamp", ascending=True).copy()
    up = 0.0
    down = 0.0
    rows = []
    for _, r in d.iterrows():
        side = str(r["side"]).upper()
        outcome = str(r["outcome"]).lower()
        notional = float(r["notional"])
        delta = notional if side == "BUY" else -notional if side == "SELL" else 0.0
        if outcome == "up":
            up += delta
        elif outcome == "down":
            down += delta
        t = float(r["min_from_start"])
        rows.append({"min_from_start": t, "value": up + down, "series": "Exposure TOTAL"})
        rows.append({"min_from_start": t, "value": up, "series": "Exposure UP"})
        rows.append({"min_from_start": t, "value": down, "series": "Exposure DOWN"})
    return pd.DataFrame(rows)


def shares_curves_net(df: pd.DataFrame) -> pd.DataFrame:
    d = df.sort_values("timestamp", ascending=True).copy()
    up = 0.0
    down = 0.0
    rows = []
    for _, r in d.iterrows():
        side = str(r["side"]).upper()
        outcome = str(r["outcome"]).lower()
        size = float(r["size"])
        delta = size if side == "BUY" else -size if side == "SELL" else 0.0
        if outcome == "up":
            up += delta
        elif outcome == "down":
            down += delta
        t = float(r["min_from_start"])
        rows.append({"min_from_start": t, "value": up + down, "series": "Shares TOTAL"})
        rows.append({"min_from_start": t, "value": up, "series": "Shares UP"})
        rows.append({"min_from_start": t, "value": down, "series": "Shares DOWN"})
    return pd.DataFrame(rows)


def multi_line_chart(df_lines: pd.DataFrame, title: str, y_title: str, domain: List[str], colors: List[str]):
    base = (
        alt.Chart(df_lines)
        .mark_line(strokeWidth=4)
        .encode(
            x=alt.X("min_from_start:Q", title="Minutes (0 → 15)", scale=alt.Scale(domain=[0, 15])),
            y=alt.Y("value:Q", title=y_title),
            color=alt.Color("series:N", scale=alt.Scale(domain=domain, range=colors),
                            legend=alt.Legend(title=title)),
            tooltip=[
                alt.Tooltip("min_from_start:Q", title="Minute"),
                alt.Tooltip("series:N", title="Series"),
                alt.Tooltip("value:Q", title=y_title),
            ],
        )
        .properties(height=260, title=title)
        .interactive()
    )
    hover = (
        alt.Chart(df_lines)
        .mark_point(size=260, opacity=0)
        .encode(
            x="min_from_start:Q",
            y="value:Q",
            color=alt.Color("series:N", scale=alt.Scale(domain=domain, range=colors), legend=None),
            tooltip=[
                alt.Tooltip("min_from_start:Q", title="Minute"),
                alt.Tooltip("series:N", title="Series"),
                alt.Tooltip("value:Q", title=y_title),
            ],
        )
    )
    return alt.layer(base, hover)


# ----------------------------
# FINAL P&L (infer close like your script)
# ----------------------------
def infer_closed_winner_from_last_trade(df: pd.DataFrame, threshold: float = PRICE_RESOLUTION_THRESHOLD) -> Optional[str]:
    """
    Like your CLI script:
    - take latest trade (max timestamp)
    - if outcome is up/down and price >= threshold => resolved toward that outcome
      else => opposite
    Returns "Up" or "Down" or None
    """
    if df.empty:
        return None
    last = df.loc[df["timestamp"].idxmax()]
    outcome = str(last.get("outcome", "")).lower()
    price = float(last.get("price", 0.0))

    if outcome not in ("up", "down"):
        return None

    if price >= threshold:
        return "Up" if outcome == "up" else "Down"
    else:
        return "Down" if outcome == "up" else "Up"


def pair_final_pnl_inferred(df: pd.DataFrame, winner_outcome: str) -> float:
    """
    FINAL PnL at resolution:
    - net_spent = total BUY notional - total SELL notional  (cash out)
    - final_value = net_shares(winner_outcome) * 1.0
    - pnl = final_value - net_spent
    """
    spent, received, _net_cf = cashflows(df)
    net_spent = float(spent - received)

    nsh = net_shares_by_outcome(df)
    win_shares = float(nsh.get(winner_outcome, 0.0))
    final_value = win_shares * 1.0
    return final_value - net_spent


# ============================
# Trader Profile View
# ============================
def _seconds_until_next_quarter() -> int:
    """Seconds until next :00/:15/:30/:45 boundary (UTC)."""
    try:
        now = time.time()
        sec = int(now)
        # next boundary in unix seconds
        step = 15 * 60
        next_ts = ((sec // step) + 1) * step
        return max(1, int(next_ts - sec))
    except Exception:
        return 900


def fetch_trades_all_user(user_address: str, max_pages: int = 60) -> List[dict]:
    """Best-effort: fetches user's trades across markets from Data-API."""
    all_trades: List[dict] = []
    seen: set[str] = set()

    limit = 500
    offset = 0
    pages = 0

    while True:
        params = {
            "limit": limit,
            "offset": offset,
            "takerOnly": "false",
            "user": user_address,
        }
        resp = requests.get(TRADES_URL, params=params, timeout=25)
        resp.raise_for_status()
        data = resp.json()

        if isinstance(data, dict):
            batch = data.get("trades", [])
        elif isinstance(data, list):
            batch = data
        else:
            batch = []

        if not batch:
            break

        new_count = 0
        for t in batch:
            tx = t.get("transactionHash")
            # If tx is missing, still include but do not dedupe
            if tx and tx in seen:
                continue
            if tx:
                seen.add(tx)
            all_trades.append(t)
            new_count += 1

        offset += limit
        pages += 1

        # Stop if backend repeats pages endlessly OR we reached our cap.
        if new_count == 0 or pages >= max_pages:
            break

    return all_trades


def _trade_market_key(t: dict) -> Optional[str]:
    for k in ("market", "conditionId", "condition_id", "condition", "marketId"):
        v = t.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


@st.cache_data(show_spinner=False, ttl=60 * 15)
def _market_meta_for_condition(condition_or_slug: str) -> dict:
    try:
        return resolve_market(condition_or_slug)
    except Exception:
        # If the key is a conditionId, try that. If it's not, just return empty.
        try:
            if str(condition_or_slug).startswith("0x") and len(str(condition_or_slug)) == 66:
                return resolve_market(str(condition_or_slug))
        except Exception:
            pass
        return {}


def _filter_df_by_range(df: pd.DataFrame, rng: str) -> pd.DataFrame:
    if df.empty:
        return df
    now = int(time.time())
    if rng == "1D":
        cutoff = now - 86400
        return df[df["timestamp"] >= cutoff].copy()
    if rng == "1W":
        cutoff = now - 7 * 86400
        return df[df["timestamp"] >= cutoff].copy()
    if rng == "1M":
        cutoff = now - 30 * 86400
        return df[df["timestamp"] >= cutoff].copy()
    return df.copy()


def _pnl_series_from_trades(df: pd.DataFrame, mids_by_asset: Dict[str, Optional[float]]) -> pd.DataFrame:
    """Equity curve proxy: net cashflow + mark-to-market at CURRENT mids, updated on each trade."""
    if df.empty:
        return pd.DataFrame(columns=["time_utc", "pnl"])

    d = df.sort_values("timestamp", ascending=True).copy()
    net_cf = 0.0
    token_net: Dict[str, float] = {}
    rows = []

    for _, r in d.iterrows():
        side = str(r.get("side") or "").upper()
        token = str(r.get("asset") or "")
        notional = float(r.get("notional") or 0.0)
        size = float(r.get("size") or 0.0)

        if side == "BUY":
            net_cf -= notional
            token_net[token] = token_net.get(token, 0.0) + size
        elif side == "SELL":
            net_cf += notional
            token_net[token] = token_net.get(token, 0.0) - size

        m2m = 0.0
        for tok, sh in token_net.items():
            mid = mids_by_asset.get(tok)
            if mid is None:
                continue
            m2m += float(sh) * float(mid)

        rows.append({
            "time_utc": r.get("time_utc"),
            "pnl": float(net_cf + m2m),
        })

    return pd.DataFrame(rows).dropna()


def _render_trade_card(summary: dict, key: str):
    """A compact 'card' for a paired position in a single market."""
    title = summary.get("title") or "Market"
    sub = summary.get("subtitle") or ""
    pnl = float(summary.get("pnl", 0.0))
    cost = float(summary.get("cost", 0.0))
    worst = float(summary.get("worst", 0.0))
    best = float(summary.get("best", 0.0))
    avg = float(summary.get("avg", 0.0))
    med = float(summary.get("median", 0.0))
    imbalance = summary.get("imbalance") or "—"
    flips = int(summary.get("flips", 0))

    # Card UI
    st.markdown(
        """
        <style>
        .pm-card {background: #f8fafc; border: 1px solid #e5e7eb; border-radius: 16px; padding: 16px;}
        .pm-title {font-size: 22px; font-weight: 700; margin-bottom: 2px;}
        .pm-sub {color: #6b7280; margin-bottom: 10px;}
        .pm-row {display:flex; justify-content:space-between; align-items:baseline; gap:12px; flex-wrap:wrap;}
        .pm-big {font-size: 34px; font-weight: 800;}
        .pm-pill {display:inline-block; padding: 6px 10px; border-radius: 999px; font-weight: 700; font-size: 13px; border: 1px solid #e5e7eb; background: white;}
        .pm-metrics {display:flex; gap: 18px; flex-wrap:wrap; margin-top: 12px; color:#111827;}
        .pm-metric {min-width: 120px;}
        .pm-label {color:#6b7280; font-size: 13px; margin-bottom: 2px;}
        .pm-val {font-weight: 800; font-size: 18px;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    pnl_str = format_money(pnl)
    cost_str = format_money(cost)

    st.markdown(
        f"""
        <div class="pm-card">
          <div class="pm-row">
            <div>
              <div class="pm-title">{title}</div>
              <div class="pm-sub">{sub}</div>
              <div class="pm-pill">Cost: {cost_str}</div>
            </div>
            <div class="pm-big">{pnl_str}</div>
          </div>
          <div class="pm-metrics">
            <div class="pm-metric"><div class="pm-label">Worst</div><div class="pm-val">{format_money(worst)}</div></div>
            <div class="pm-metric"><div class="pm-label">Best</div><div class="pm-val">{format_money(best)}</div></div>
            <div class="pm-metric"><div class="pm-label">Avg</div><div class="pm-val">{format_money(avg)}</div></div>
            <div class="pm-metric"><div class="pm-label">Median</div><div class="pm-val">{format_money(med)}</div></div>
            <div class="pm-metric"><div class="pm-label">Imbalance</div><div class="pm-val">{imbalance}</div></div>
            <div class="pm-metric"><div class="pm-label">Flips</div><div class="pm-val">{flips}</div></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    b1, b2 = st.columns([1.1, 1.9])
    with b1:
        if st.button("Open market analysis", key=f"go_{key}", use_container_width=True):
            st.session_state.page = "market"
            st.session_state.prefill_wallet = summary.get("wallet") or ""
            st.session_state.prefill_market = summary.get("market") or ""
            try:
                st.query_params["page"] = "market"
                st.query_params["wallet"] = st.session_state.prefill_wallet
                st.query_params["market"] = st.session_state.prefill_market
            except Exception:
                try:
                    st.experimental_set_query_params(page="market", wallet=st.session_state.prefill_wallet, market=st.session_state.prefill_market)
                except Exception:
                    pass
            st.rerun()
    with b2:
        # Best-effort popup (Streamlit 1.32+)
        try:
            @st.dialog("Trade Pair Details")
            def _dlg():
                st.write(f"**Market:** {title}")
                st.write(f"**Market key:** {summary.get('market')}")
                st.write(f"**Wallet:** {summary.get('wallet')}")
                st.write(f"**Cost:** {cost_str}")
                st.write(f"**PnL (open est):** {pnl_str}")
                st.write("---")
                st.dataframe(summary.get("df_trades"), use_container_width=True, hide_index=True)

            if st.button("View details (popup)", key=f"dlg_{key}", use_container_width=True):
                _dlg()
        except Exception:
            # Fallback: expander
            with st.expander("View details"):
                st.dataframe(summary.get("df_trades"), use_container_width=True, hide_index=True)


def render_trader_profile_page():
    st.header("Trader Profile")

    # Auto-refresh aligned to :00/:15/:30/:45
    next_in = _seconds_until_next_quarter()
    try:
        st.autorefresh(interval=next_in * 1000, key="profile_autorefresh")
    except Exception:
        try:
            from streamlit_autorefresh import st_autorefresh

            st_autorefresh(interval=next_in * 1000, key="profile_autorefresh")
        except Exception:
            pass

    col1, col2, col3 = st.columns([2.2, 1.2, 2.6])
    with col1:
        wallet_in = st.text_input(
            "Trader wallet (0x...)",
            value=str(getattr(st.session_state, "prefill_wallet", "")) or "",
            placeholder="0x1234...",
            key="profile_wallet",
        )
    with col2:
        try:
            rng = st.segmented_control("Range", options=["ALL", "1M", "1W", "1D"], default="1W")
        except Exception:
            rng = st.selectbox("Range", ["ALL", "1M", "1W", "1D"], index=2)
    with col3:
        st.caption("Auto-refreshes on quarter-hours (UTC): :00 / :15 / :30 / :45")
        st.caption(f"Next refresh in ~{next_in}s")

    if not (wallet_in or "").strip():
        st.info("Paste a wallet to analyze.")
        st.stop()

    try:
        wallet = normalize_address(wallet_in)
    except PMError as e:
        st.error(str(e))
        st.stop()

    with st.spinner("Fetching trader trades…"):
        try:
            trades = fetch_trades_all_user(wallet)
        except Exception as e:
            st.error(f"Error fetching trades: {e}")
            st.stop()

    df_all = trades_to_df(trades)
    if df_all.empty:
        st.warning("No trades found for that wallet.")
        st.stop()

    # timestamp ms vs sec fix
    if df_all["timestamp"].max() > 2_000_000_000_000:
        df_all["timestamp"] = (df_all["timestamp"] / 1000).astype(int)
        df_all["time_utc"] = pd.to_datetime(df_all["timestamp"], unit="s", utc=True)

    # Attach a market key column (conditionId-like) for grouping
    df_all["market_key"] = [
        _trade_market_key(t) for t in trades
    ]
    df_all = df_all.dropna(subset=["market_key"]).copy()

    df = _filter_df_by_range(df_all, rng)

    # Mid prices (cap to keep it fast)
    with st.spinner("Fetching mid prices (for open PnL estimates)…"):
        mids = mid_prices_for_assets(df, cap=60)

    # Total PnL (sum of per-market open estimates)
    per_market = []
    for mk, g in df.groupby("market_key"):
        pnl_open, _ = pair_pnl_open_estimate(g, mids)
        spent, received, _ = cashflows(g)
        per_market.append({"market_key": mk, "pnl": float(pnl_open), "cost": float(spent), "trades": int(len(g))})
    pm_df = pd.DataFrame(per_market).sort_values("pnl", ascending=False) if per_market else pd.DataFrame(columns=["market_key", "pnl", "cost", "trades"])

    total_pnl = float(pm_df["pnl"].sum()) if not pm_df.empty else 0.0
    st.subheader("Total PnL")
    st.metric("TOTAL PnL (open est)", format_money(total_pnl))

    # PnL curve (proxy)
    curve = _pnl_series_from_trades(df, mids)
    if not curve.empty:
        ch = (
            alt.Chart(curve)
            .mark_line(strokeWidth=3)
            .encode(
                x=alt.X("time_utc:T", title="Time (UTC)"),
                y=alt.Y("pnl:Q", title="PnL ($)"),
                tooltip=[alt.Tooltip("time_utc:T", title="Time"), alt.Tooltip("pnl:Q", title="PnL ($)")],
            )
            .properties(height=260)
            .interactive()
        )
        st.altair_chart(ch, use_container_width=True)
    else:
        st.info("Not enough data to draw a PnL curve for that range.")

    st.write("---")
    st.subheader("Trade Pairs (by market)")

    # Build paired market cards
    cards = []
    for mk, g in df.groupby("market_key"):
        meta = _market_meta_for_condition(str(mk))
        title = meta.get("question") or meta.get("title") or str(mk)
        # Subtitle: attempt to include slug + window
        sub = meta.get("slug") or ""
        # Cost and PnL
        spent, received, _ = cashflows(g)
        pnl_open, _ = pair_pnl_open_estimate(g, mids)

        # Simple "worst/best" proxies using max_profit_loss_curves (within 15m-style markets this is meaningful)
        g_tmp = g.copy()
        # If market has start/end, we can normalize; otherwise, keep raw timestamps.
        m_start, m_end = market_start_end_ts(meta) if meta else (None, None)
        g_tmp = build_time_axis(g_tmp, m_start, m_end)
        curves = max_profit_loss_curves(g_tmp)
        worst = float(curves["max_loss"].min()) if not curves.empty else float(pnl_open)
        best = float(curves["max_profit"].max()) if not curves.empty else float(pnl_open)

        # Avg/Median: using pnl snapshots (proxy)
        snaps = _pnl_series_from_trades(g_tmp, mids)
        avg = float(snaps["pnl"].mean()) if not snaps.empty else float(pnl_open)
        med = float(snaps["pnl"].median()) if not snaps.empty else float(pnl_open)

        # Imbalance: which side has more net shares
        nsh = net_shares_by_outcome(g)
        up_sh = float(nsh.get("Up", 0.0))
        down_sh = float(nsh.get("Down", 0.0))
        if abs(up_sh) > abs(down_sh):
            imb = "UP"
        elif abs(down_sh) > abs(up_sh):
            imb = "DOWN"
        else:
            imb = "EVEN"

        # Flips: number of side changes (BUY/SELL) on the most traded outcome
        flips = 0
        try:
            s = g.sort_values("timestamp", ascending=True)["side"].astype(str).tolist()
            for i in range(1, len(s)):
                if s[i] != s[i - 1]:
                    flips += 1
        except Exception:
            flips = 0

        df_trades_show = g.sort_values("timestamp", ascending=False).copy()
        df_trades_show["time_utc"] = df_trades_show["time_utc"].astype(str)
        df_trades_show = df_trades_show[["time_utc", "side", "outcome", "price", "size", "notional", "asset", "transactionHash"]]

        cards.append({
            "wallet": wallet,
            "market": str(mk),
            "title": title,
            "subtitle": sub,
            "pnl": float(pnl_open),
            "cost": float(spent),
            "worst": float(worst),
            "best": float(best),
            "avg": float(avg),
            "median": float(med),
            "imbalance": imb,
            "flips": int(flips),
            "df_trades": df_trades_show,
        })

    # Sort by most recent activity (best-effort)
    cards = sorted(cards, key=lambda x: float(df[df["market_key"] == x["market"]]["timestamp"].max()) if not df.empty else 0.0, reverse=True)

    if not cards:
        st.info("No markets found for that range.")
        st.stop()

    for i, c in enumerate(cards):
        _render_trade_card(c, key=f"card_{i}")
        st.write("")


# ============================
# UI
# ============================
if str(getattr(st.session_state, "page", "market")) == "profile":
    render_trader_profile_page()
    st.stop()

colA, colB = st.columns([2, 3])
with colA:
    wallet_in = st.text_input(
        "Wallet (0x...)",
        value=str(getattr(st.session_state, "prefill_wallet", "")) or "",
        placeholder="0x1234...",
    )
with colB:
    market_in = st.text_input(
        "Market (slug / conditionId / URL)",
        value=str(getattr(st.session_state, "prefill_market", "")) or "",
    )

st.divider()

if not wallet_in.strip() or not market_in.strip():
    st.info("Fill **Wallet** and **Market** to begin.")
    st.stop()

try:
    wallet = normalize_address(wallet_in)
    market = resolve_market(market_in)
except PMError as e:
    st.error(str(e))
    st.stop()

condition_id = (market.get("conditionId") or "").strip()
if not condition_id:
    st.error("Could not resolve conditionId from market metadata.")
    st.stop()

is_active, is_closed = compute_state(market)
winner_meta_raw = resolve_winner_label(market)
winner_meta = winner_to_outcome_label(winner_meta_raw)
m_start, m_end = market_start_end_ts(market)

h1, h2, h3, h4 = st.columns([2.6, 1.2, 1.0, 1.0])
with h1:
    st.subheader(market.get("question") or market.get("title") or "Market")
    st.caption(f"Slug: {market.get('slug')}")
with h2:
    st.metric("ConditionId", condition_id[:18] + "…")
with h3:
    st.metric("Active", str(is_active))
with h4:
    st.metric("Closed", str(is_closed))

with st.spinner("Fetching trades (complete, paginated)…"):
    try:
        trades = fetch_trades_complete(condition_id, wallet)
    except Exception as e:
        st.error(f"Error fetching trades: {e}")
        st.stop()

df = trades_to_df(trades)
if df.empty:
    st.warning("No trades found for that wallet in this market.")
    st.stop()

# timestamp ms vs sec fix
if df["timestamp"].max() > 2_000_000_000_000:
    df["timestamp"] = (df["timestamp"] / 1000).astype(int)

# build minutes axis
df = build_time_axis(df, m_start, m_end)

# re-anchor if needed so last trade lands in minute 15 window
max_min = float(df["min_from_start"].max()) if len(df) else 0.0
if max_min > 15.0:
    t1 = int(df["timestamp"].max())
    t0 = t1 - 900
    df["t0"] = t0
    df["t1"] = t1
    df["sec_from_start"] = (df["timestamp"] - t0).clip(lower=0, upper=900)
    df["min_from_start"] = df["sec_from_start"] / 60.0

st.caption(f"Fetched trades: {len(df)}")

# Metrics core
spent, received, net_cf = cashflows(df)
exposure = max_exposure(df)
net_sh_trades = net_shares_by_outcome(df)
avg_buy = avg_buy_price_by_outcome(df)

with st.spinner("Fetching mid prices (OPEN estimates)…"):
    mids = mid_prices_for_assets(df, cap=30)

pair_open_pnl, _m2m = pair_pnl_open_estimate(df, mids)

# FINAL PnL block (infer if meta missing)
st.write("## Pair Final P&L (Resolution)")
infer_winner = infer_closed_winner_from_last_trade(df, threshold=PRICE_RESOLUTION_THRESHOLD)

winner_used = winner_meta if winner_meta else infer_winner
pnl_final = None
if winner_used in ("Up", "Down"):
    pnl_final = pair_final_pnl_inferred(df, winner_used)

p1, p2, p3, p4 = st.columns([1.2, 1.4, 1.4, 2.0])
with p1:
    st.metric("Market State", "CLOSED" if is_closed else "OPEN")
with p2:
    st.metric("Winner (metadata)", winner_meta or "Unknown")
with p3:
    st.metric("Winner (inferred)", infer_winner or "Unknown")
with p4:
    if pnl_final is not None:
        st.metric("PAIR FINAL P&L (inferred/metadata)", format_money(float(pnl_final)))
    else:
        st.metric("PAIR P&L (OPEN • EST)", format_money(float(pair_open_pnl)))

st.write("---")

c1, c2, c3, c4 = st.columns(4)
c1.metric("TOTAL TRADES", f"{len(df)}")
c2.metric("VOLUME", format_money(float(df["notional"].sum())))
if pnl_final is not None:
    c3.metric("PAIR FINAL P&L", format_money(float(pnl_final)))
else:
    c3.metric("PAIR P&L (OPEN • EST)", format_money(float(pair_open_pnl)))
c4.metric("MAX EXPOSURE", format_money(exposure))

s1, s2, s3, s4 = st.columns(4)
s1.metric("FINAL UP SHARES (from trades)", f"{net_sh_trades.get('Up', 0.0):,.2f}")
s2.metric("FINAL DOWN SHARES (from trades)", f"{net_sh_trades.get('Down', 0.0):,.2f}")
s3.metric("AVG UP BUY PRICE", "—" if avg_buy.get("Up") is None else f"{avg_buy['Up']:.3f}")
s4.metric("AVG DOWN BUY PRICE", "—" if avg_buy.get("Down") is None else f"{avg_buy['Down']:.3f}")

positions = data_api_get_positions(wallet, condition_id)
if positions is not None and len(positions) > 0:
    st.write("### Positions (best-effort from Data-API)")
    st.dataframe(pd.DataFrame(positions), use_container_width=True, hide_index=True)

# Controls
st.write("### Price Chart")

k1, k2, k3, k4 = st.columns(4)
with k1:
    show_points = st.checkbox("Show trades (points)", value=True)
with k2:
    only_buys_points = st.checkbox("Points: only BUY", value=True)
with k3:
    show_vwap_all_buys = st.checkbox("Line: VWAP ALL (buys)", value=True)
with k4:
    show_vwap_all_trades = st.checkbox("Line: VWAP ALL (all trades)", value=False)

l1, l2, l3, l4 = st.columns(4)
with l1:
    show_vwap_up = st.checkbox("Line: VWAP UP", value=True)
with l2:
    show_vwap_down = st.checkbox("Line: VWAP DOWN", value=True)
with l3:
    show_max_profit = st.checkbox("PnL band: Max Profit", value=True)
with l4:
    show_max_loss = st.checkbox("PnL band: Max Loss", value=True)

top = price_chart(
    df,
    show_points=show_points,
    only_buys_points=only_buys_points,
    show_vwap_all_buys=show_vwap_all_buys,
    show_vwap_all_trades=show_vwap_all_trades,
    show_vwap_up=show_vwap_up,
    show_vwap_down=show_vwap_down,
)
bottom = pnl_chart(df, show_max_profit=show_max_profit, show_max_loss=show_max_loss)

if top is not None:
    st.altair_chart(top, use_container_width=True)
if bottom is not None:
    st.altair_chart(bottom, use_container_width=True)

# ✅ YOUR REQUESTED EXTRA CHARTS (kept)
st.write("### Exposure (USD)")
exp_df = exposure_curves_net(df)
exp_chart = multi_line_chart(
    exp_df,
    title="Exposure in USD over time",
    y_title="Exposure ($)",
    domain=["Exposure TOTAL", "Exposure UP", "Exposure DOWN"],
    colors=["#ffffff", "#22c55e", "#3b82f6"],
)
st.altair_chart(exp_chart, use_container_width=True)

st.write("### Shares (Net)")
sh_df = shares_curves_net(df)
sh_chart = multi_line_chart(
    sh_df,
    title="Shares over time",
    y_title="Shares",
    domain=["Shares TOTAL", "Shares UP", "Shares DOWN"],
    colors=["#ffffff", "#22c55e", "#3b82f6"],
)
st.altair_chart(sh_chart, use_container_width=True)

# Trades table
st.write("### Trades (latest first)")
df_show = df.sort_values("timestamp", ascending=False).copy()
df_show["time_utc"] = df_show["time_utc"].astype(str)
st.dataframe(
    df_show[["time_utc", "min_from_start", "side", "outcome", "price", "size", "notional", "asset", "transactionHash"]],
    use_container_width=True,
    hide_index=True,
)
