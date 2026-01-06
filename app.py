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

# -----------------------
# Top navigation (no sidebar)
# -----------------------
def _get_qp():
    try:
        return dict(st.query_params)
    except Exception:
        return st.experimental_get_query_params()

def _set_qp(**kwargs):
    try:
        st.query_params.update(kwargs)
    except Exception:
        st.experimental_set_query_params(**kwargs)

_qp = _get_qp()
_page = (_qp.get("page", ["analyzer"])[0] if isinstance(_qp.get("page"), list) else _qp.get("page")) or "analyzer"

navA, navB, navC = st.columns([1, 1, 8])
with navA:
    if st.button("Market", key="nav_market"):
        _set_qp(page="analyzer")
        st.rerun()
with navB:
    if st.button("Trader", key="nav_trader"):
        _set_qp(page="trader")
        st.rerun()


# -----------------------
# Trader Profile view
# -----------------------
def _now_utc_ts() -> int:
    return int(time.time())

def _start_ts_for_range(rng: str, now_ts: int) -> Optional[int]:
    rng = (rng or "ALL").upper()
    if rng == "1D":
        return now_ts - 86400
    if rng == "1W":
        return now_ts - 7 * 86400
    if rng == "1M":
        return now_ts - 30 * 86400
    return None

def _ms_until_next_quarter_hour(now_ts: Optional[int] = None) -> int:
    now_ts = int(now_ts or _now_utc_ts())
    q = 15 * 60
    next_ts = ((now_ts // q) + 1) * q
    return max(1000, int((next_ts - now_ts) * 1000))

@st.cache_data(show_spinner=False, ttl=60)
def _positions_for_user(user_addr: str) -> List[dict]:
    return data_api_get_positions(user_addr)

@st.cache_data(show_spinner=False, ttl=60)
def _market_for_condition(condition_id: str) -> dict:
    return resolve_market(condition_id)

def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None

def _mid_by_assets_for_market(market: dict) -> Dict[str, Optional[float]]:
    mid_by_asset: Dict[str, Optional[float]] = {}
    outcomes = market.get("outcomes") or market.get("tokens") or []
    # Try multiple shapes:
    for o in outcomes:
        asset = o.get("tokenId") or o.get("assetId") or o.get("id")
        if asset is None:
            continue
        asset = str(asset)
        # mid price via CLOB
        mid = None
        try:
            mid = clob_get_mid_price(asset)
        except Exception:
            mid = None
        mid_by_asset[asset] = _safe_float(mid)
    return mid_by_asset

def _group_markets_from_positions(pos: List[dict]) -> List[str]:
    # Prefer conditionId keys
    ids: List[str] = []
    for p in pos or []:
        cid = p.get("conditionId") or p.get("condition_id") or p.get("market") or p.get("marketId")
        if cid:
            ids.append(str(cid))
    # de-dupe stable
    seen=set()
    out=[]
    for cid in ids:
        if cid not in seen:
            seen.add(cid)
            out.append(cid)
    return out

def _df_for_market_wallet(user_addr: str, condition_id: str) -> pd.DataFrame:
    trades = fetch_trades_complete(user_addr, condition_id)
    df = trades_to_df(trades)
    return df

def _pnl_series_open(df: pd.DataFrame, mid_by_asset: Dict[str, Optional[float]]) -> pd.DataFrame:
    d = df.sort_values("timestamp", ascending=True).copy()
    net_cf = 0.0
    token_net: Dict[str, float] = {}
    rows=[]
    for _, r in d.iterrows():
        side = str(r.get("side") or "").upper()
        notional = float(r.get("notional") or 0.0)
        asset = str(r.get("asset") or "")
        size = float(r.get("size") or 0.0)
        if side == "BUY":
            net_cf -= notional
            token_net[asset] = token_net.get(asset, 0.0) + size
        elif side == "SELL":
            net_cf += notional
            token_net[asset] = token_net.get(asset, 0.0) - size
        m2m=0.0
        for a, sh in token_net.items():
            mid = mid_by_asset.get(a)
            if mid is None:
                continue
            m2m += float(sh) * float(mid)
        rows.append({"timestamp": int(r["timestamp"]), "pnl": float(net_cf + m2m)})
    if not rows:
        return pd.DataFrame(columns=["timestamp","pnl"])
    return pd.DataFrame(rows)

def render_trader_profile() -> None:
    st.header("Trader Profile")

    topA, topB, topC = st.columns([2, 1, 2])
    with topA:
        wallet_raw = st.text_input("Trader wallet (0x...)", value=str(_get_qp().get("wallet", [""])[0] if isinstance(_get_qp().get("wallet"), list) else _get_qp().get("wallet","")), placeholder="0x1234...")
    with topB:
        rng = st.selectbox("Range", ["ALL", "1M", "1W", "1D"], index=0, key="profile_range")
    with topC:
        st.write("")
        st.write("")

    if not wallet_raw.strip():
        st.info("Paste a wallet to analyze.")
        return

    try:
        wallet = normalize_address(wallet_raw)
    except Exception:
        st.error("Invalid wallet format.")
        return

    # Aligned refresh
    ms = _ms_until_next_quarter_hour()
    try:
        st.autorefresh(interval=ms, key=f"trader_refresh_{wallet}_{rng}")
    except Exception:
        # If older Streamlit
        pass

    now_ts = _now_utc_ts()
    start_ts = _start_ts_for_range(rng, now_ts)

    # Always load ALL trades (via positions → conditionIds → trades per market)
    with st.spinner("Loading positions & trades..."):
        pos = _positions_for_user(wallet)
        condition_ids = _group_markets_from_positions(pos)

    cards: List[dict] = []
    pnl_points: List[pd.DataFrame] = []
    total_pnl = 0.0

    for cid in condition_ids:
        try:
            market = _market_for_condition(cid)
        except Exception:
            continue

        df = _df_for_market_wallet(wallet, cid)
        if df is None or df.empty:
            continue

        # Filter for display range ONLY (but data is still loaded from full history)
        df_disp = df
        if start_ts is not None:
            df_disp = df[df["timestamp"] >= int(start_ts)].copy()
            if df_disp.empty:
                continue

        # Use analyzer math
        mid_by_asset = _mid_by_assets_for_market(market)
        try:
            pnl_open, _m2m = pair_pnl_open_estimate(df_disp, mid_by_asset)
        except Exception:
            pnl_open = 0.0

        nets = net_shares_by_outcome(df_disp)
        avgs = avg_buy_price_by_outcome(df_disp)

        # floor/ceiling from max_profit_loss_curves (end state)
        try:
            df_axis = build_time_axis(df_disp, *(market_start_end_ts(market)))
            mm = max_profit_loss_curves(df_axis)
            floor = float(mm["max_loss"].iloc[-1]) if not mm.empty else None
            ceiling = float(mm["max_profit"].iloc[-1]) if not mm.empty else None
        except Exception:
            floor, ceiling = None, None

        # PnL series for chart
        s = _pnl_series_open(df_disp, mid_by_asset)
        if not s.empty:
            pnl_points.append(s)

        # last ts to sort newest first
        last_ts = int(df_disp["timestamp"].max())

        cards.append({
            "condition_id": cid,
            "market_title": str(market.get("question") or market.get("title") or cid),
            "last_ts": last_ts,
            "pnl": float(pnl_open),
            "up_sh": float(nets.get("UP", 0.0)),
            "down_sh": float(nets.get("DOWN", 0.0)),
            "up_avg": avgs.get("UP"),
            "down_avg": avgs.get("DOWN"),
            "floor": floor,
            "ceiling": ceiling,
        })
        total_pnl += float(pnl_open)

    # Header metrics
    st.subheader(f"Total PnL: {format_money(total_pnl)}")

    # Chart: merge series
    if pnl_points:
        chart_df = pd.concat(pnl_points, ignore_index=True).sort_values("timestamp")
        chart_df["time"] = pd.to_datetime(chart_df["timestamp"], unit="s", utc=True)
        st.altair_chart(
            alt.Chart(chart_df).mark_line().encode(
                x=alt.X("time:T", title="Time (UTC)"),
                y=alt.Y("pnl:Q", title="PnL"),
                tooltip=["time:T", "pnl:Q"],
            ).properties(height=220),
            use_container_width=True,
        )

    # Cards newest first
    cards = sorted(cards, key=lambda x: x["last_ts"], reverse=True)

    st.divider()
    st.caption("Cards are grouped per market (UP+DOWN). Newest first.")

    for i, c in enumerate(cards):
        with st.container(border=True):
            row1 = st.columns([5, 2, 2, 2, 2])
            row1[0].markdown(f"**{c['market_title']}**")
            row1[1].markdown(f"**PNL**  \n{format_money(c['pnl'])}")
            row1[2].markdown(f"**UP sh**  \n{c['up_sh']:.4f}")
            row1[3].markdown(f"**DOWN sh**  \n{c['down_sh']:.4f}")
            row1[4].markdown(f"**Floor/Ceil**  \n{format_money(c['floor']) if c['floor'] is not None else '—'} / {format_money(c['ceiling']) if c['ceiling'] is not None else '—'}")

            row2 = st.columns([2, 2, 6])
            row2[0].markdown(f"**UP avg**  \n{(str(round(float(c['up_avg']),4))+'¢') if c['up_avg'] is not None else '—'}")
            row2[1].markdown(f"**DOWN avg**  \n{(str(round(float(c['down_avg']),4))+'¢') if c['down_avg'] is not None else '—'}")

            # Actions
            goA, goB, _sp = st.columns([2, 2, 6])
            if goA.button("Open market analysis", key=f"open_{i}_{c['condition_id']}"):
                _set_qp(page="analyzer", wallet=wallet, market=c["condition_id"])
                st.rerun()

            # Details popup/expander
            with goB:
                try:
                    @st.dialog("Trade details")
                    def _dlg():
                        df_full = _df_for_market_wallet(wallet, c["condition_id"])
                        st.dataframe(df_full.sort_values("timestamp", ascending=False), use_container_width=True, height=260)
                    if st.button("View details", key=f"details_{i}_{c['condition_id']}"):
                        _dlg()
                except Exception:
                    with st.expander("View details", expanded=False):
                        df_full = _df_for_market_wallet(wallet, c["condition_id"])
                        st.dataframe(df_full.sort_values("timestamp", ascending=False), use_container_width=True, height=260)

# If trader page is requested, we will render it AFTER helper functions are defined.
TRADER_PAGE_REQUESTED = (_page == "trader")




TRADES_URL = "https://data-api.polymarket.com/trades"
PRICE_RESOLUTION_THRESHOLD = 0.5  # like your script


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



# -----------------------
# Render Trader Profile (after helper functions exist)
# -----------------------
if 'TRADER_PAGE_REQUESTED' in globals() and TRADER_PAGE_REQUESTED:
    render_trader_profile()
    st.stop()


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
# UI
# ============================
colA, colB = st.columns([2, 3])
with colA:
    wallet_in = st.text_input("Wallet (0x...)", value="", placeholder="0x1234...")
with colB:
    market_in = st.text_input("Market (slug / conditionId / URL)", value="")

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
