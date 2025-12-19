from __future__ import annotations

from typing import Dict, Optional, List, Tuple
import pandas as pd
import datetime as dt


def _to_datetime_utc(ts: int) -> dt.datetime:
    return dt.datetime.fromtimestamp(int(ts), tz=dt.timezone.utc)


def trades_to_df(trades: List[dict]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(columns=[
            "timestamp", "time_utc", "side", "outcome", "price", "size",
            "notional", "asset", "transactionHash"
        ])

    rows = []
    for t in trades:
        price = float(t.get("price", 0) or 0)
        size = float(t.get("size", 0) or 0)
        ts = int(t.get("timestamp", 0) or 0)
        rows.append({
            "timestamp": ts,
            "time_utc": _to_datetime_utc(ts),
            "side": str(t.get("side", "")).upper(),
            "outcome": t.get("outcome"),
            "price": price,
            "size": size,
            "notional": price * size,
            "asset": str(t.get("asset") or ""),
            "transactionHash": t.get("transactionHash"),
        })

    return pd.DataFrame(rows).sort_values("timestamp", ascending=True).reset_index(drop=True)


def format_money(x: float) -> str:
    sign = "-" if x < 0 else ""
    return f"{sign}${abs(x):,.2f}"


def safe_div(a: float, b: float) -> Optional[float]:
    if b == 0:
        return None
    return a / b


def cashflows(df: pd.DataFrame) -> Tuple[float, float, float]:
    buys = float(df[df["side"] == "BUY"]["notional"].sum())
    sells = float(df[df["side"] == "SELL"]["notional"].sum())
    net = -buys + sells
    return buys, sells, net


def net_shares_by_outcome(df: pd.DataFrame) -> Dict[str, float]:
    res: Dict[str, float] = {}
    outcomes = [o for o in df["outcome"].dropna().unique().tolist()]
    for o in outcomes:
        o = str(o)
        b = float(df[(df["outcome"] == o) & (df["side"] == "BUY")]["size"].sum())
        s = float(df[(df["outcome"] == o) & (df["side"] == "SELL")]["size"].sum())
        res[o] = b - s
    return res


def avg_buy_price_by_outcome(df: pd.DataFrame) -> Dict[str, Optional[float]]:
    res: Dict[str, Optional[float]] = {}
    outcomes = [o for o in df["outcome"].dropna().unique().tolist()]
    for o in outcomes:
        o = str(o)
        b_notional = float(df[(df["outcome"] == o) & (df["side"] == "BUY")]["notional"].sum())
        b_size = float(df[(df["outcome"] == o) & (df["side"] == "BUY")]["size"].sum())
        res[o] = safe_div(b_notional, b_size)
    return res


def max_exposure(df: pd.DataFrame) -> float:
    exp = 0.0
    max_exp = 0.0
    for _, r in df.sort_values("timestamp", ascending=True).iterrows():
        side = str(r["side"]).upper()
        if side == "BUY":
            exp += float(r["notional"])
        elif side == "SELL":
            exp -= float(r["notional"])
        if exp > max_exp:
            max_exp = exp
    return float(max_exp)


def pair_pnl_open_estimate(df: pd.DataFrame, mid_by_asset: Dict[str, Optional[float]]) -> Tuple[float, float]:
    """
    PAIR PnL (OPEN) = net_cashflow + mark_to_market_value
    mark_to_market_value = Σ(net_shares_token * mid_token)
    """
    _, _, net_cf = cashflows(df)

    token_net: Dict[str, float] = {}
    for _, r in df.iterrows():
        token = str(r.get("asset") or "")
        side = str(r.get("side") or "").upper()
        sz = float(r.get("size") or 0.0)
        token_net[token] = token_net.get(token, 0.0) + (sz if side == "BUY" else -sz if side == "SELL" else 0.0)

    m2m = 0.0
    for token, sh in token_net.items():
        mid = mid_by_asset.get(token)
        if mid is None:
            continue
        m2m += float(sh) * float(mid)

    return float(net_cf + m2m), float(m2m)


def pair_pnl_closed_real(df: pd.DataFrame, winning_outcome: Optional[str]) -> Optional[float]:
    if not winning_outcome:
        return None
    _, _, net_cf = cashflows(df)
    nets = net_shares_by_outcome(df)
    win_sh = float(nets.get(str(winning_outcome), 0.0))
    payout = max(win_sh, 0.0) * 1.0
    return float(net_cf + payout)


# ---------- NEW: time-normalized series + lines ----------
def build_time_axis(df: pd.DataFrame, market_start_ts: Optional[int], market_end_ts: Optional[int]) -> pd.DataFrame:
    d = df.copy()

    if market_start_ts and market_end_ts and market_end_ts > market_start_ts:
        t0 = int(market_start_ts)
        t1 = int(market_end_ts)
    else:
        t0 = int(d["timestamp"].min())
        t1 = t0 + 900  # default 15m

    d["t0"] = t0
    d["t1"] = t1
    d["sec_from_start"] = (d["timestamp"] - t0).clip(lower=0, upper=(t1 - t0))
    d["min_from_start"] = d["sec_from_start"] / 60.0
    return d


def running_vwap(df: pd.DataFrame, outcome: Optional[str] = None, only_buys: bool = True) -> pd.DataFrame:
    d = df.copy()
    if outcome is not None:
        d = d[d["outcome"].astype(str) == str(outcome)].copy()
    if only_buys:
        d = d[d["side"] == "BUY"].copy()

    d = d.sort_values("timestamp", ascending=True).copy()
    d["cum_notional"] = d["notional"].cumsum()
    d["cum_size"] = d["size"].cumsum()
    d["vwap"] = d["cum_notional"] / d["cum_size"].replace(0, pd.NA)
    return d[["timestamp", "min_from_start", "vwap"]].dropna()


def max_profit_loss_curves(df: pd.DataFrame) -> pd.DataFrame:
    """
    At each time, compute:
      pnl_if_up_wins   = net_cashflow_t + net_up_shares_t * 1
      pnl_if_down_wins = net_cashflow_t + net_down_shares_t * 1
      max_profit = max(...)
      max_loss   = min(...)
    This is your 'possible Max Loss / Max Profit' through time.
    """
    d = df.sort_values("timestamp", ascending=True).copy()

    net_cf = 0.0
    up_sh = 0.0
    down_sh = 0.0

    rows = []
    for _, r in d.iterrows():
        side = str(r["side"]).upper()
        outcome = str(r["outcome"])
        notional = float(r["notional"])
        size = float(r["size"])

        if side == "BUY":
            net_cf -= notional
            if outcome.lower() == "up":
                up_sh += size
            elif outcome.lower() == "down":
                down_sh += size

        elif side == "SELL":
            net_cf += notional
            if outcome.lower() == "up":
                up_sh -= size
            elif outcome.lower() == "down":
                down_sh -= size

        pnl_up = net_cf + up_sh * 1.0
        pnl_down = net_cf + down_sh * 1.0

        rows.append({
            "timestamp": int(r["timestamp"]),
            "min_from_start": float(r["min_from_start"]),
            "max_profit": max(pnl_up, pnl_down),
            "max_loss": min(pnl_up, pnl_down),
        })

    return pd.DataFrame(rows)
