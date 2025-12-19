from __future__ import annotations

import re
from typing import Any, Optional, Tuple, List, Dict
import requests

DATA_API = "https://data-api.polymarket.com"
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"

UA = "polymarket-wallet-analyzer/FINAL-7.0"


class PMError(RuntimeError):
    pass


def _get_json(url: str, params: Optional[dict] = None, timeout: int = 20) -> Any:
    try:
        r = requests.get(url, params=params, timeout=timeout, headers={"User-Agent": UA})
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        raise PMError(f"HTTP error calling {url}: {e}") from e
    except ValueError as e:
        raise PMError(f"Non-JSON response from {url}") from e


def normalize_address(addr: str) -> str:
    addr = (addr or "").strip()
    if not re.fullmatch(r"0x[a-fA-F0-9]{40}", addr):
        raise PMError("Wallet must be a 0x-prefixed 40-hex address.")
    return addr.lower()


def extract_slug_or_condition(market_input: str) -> Tuple[Optional[str], Optional[str]]:
    s = (market_input or "").strip()

    if re.fullmatch(r"0x[a-fA-F0-9]{64}", s):
        return None, s.lower()

    if s.startswith("http://") or s.startswith("https://"):
        parts = re.findall(r"/([a-z0-9-]{6,})", s.lower())
        if parts:
            return parts[-1], None

    if re.fullmatch(r"[a-z0-9-]{6,}", s.lower()):
        return s.lower(), None

    raise PMError("Market input must be a Polymarket URL, a market slug, or a conditionId (0x + 64 hex).")


def gamma_get_market_by_slug(slug: str) -> dict:
    data = _get_json(f"{GAMMA_API}/markets", params={"slug": slug, "limit": 5, "offset": 0})
    if isinstance(data, list) and data:
        return data[0]
    raise PMError(f"No market found for slug: {slug}")


def gamma_get_market_by_condition(condition_id: str) -> dict:
    data = _get_json(f"{GAMMA_API}/markets", params={"conditionId": condition_id, "limit": 5, "offset": 0})
    if isinstance(data, list) and data:
        return data[0]
    # fallback key
    data = _get_json(f"{GAMMA_API}/markets", params={"condition_id": condition_id, "limit": 5, "offset": 0})
    if isinstance(data, list) and data:
        return data[0]
    raise PMError(f"No market found for conditionId: {condition_id}")


def resolve_market(market_input: str) -> dict:
    slug, condition = extract_slug_or_condition(market_input)
    if slug:
        return gamma_get_market_by_slug(slug)
    assert condition
    return gamma_get_market_by_condition(condition)


# ✅ IMPORTANT: Data-API /trades defaults takerOnly=true (taker trades only).
# We FORCE takerOnly=false to get taker + maker (much more complete).
def data_api_get_trades(
    user_addr: str,
    condition_id: str,
    limit: int = 500,
    offset: int = 0,
    taker_only: bool = False,
) -> List[dict]:
    params: Dict[str, object] = {
        "user": user_addr,
        "market": [condition_id],          # per docs: string[] (comma-separated list)
        "limit": int(max(1, min(int(limit), 10000))),
        "offset": int(max(0, min(int(offset), 10000))),  # docs cap offset at 10000
        "takerOnly": "true" if taker_only else "false",
    }
    data = _get_json(f"{DATA_API}/trades", params=params, timeout=25)
    if not isinstance(data, list):
        raise PMError("Unexpected trades response shape.")
    return data


def data_api_get_positions(user_addr: str, condition_id: str) -> Optional[list]:
    try:
        data = _get_json(f"{DATA_API}/positions", params={"user": user_addr, "market": [condition_id]}, timeout=15)
        if isinstance(data, list):
            return data
        return None
    except Exception:
        return None


def clob_get_token_price(token_id: str, side: str) -> Optional[float]:
    try:
        data = _get_json(f"{CLOB_API}/price", params={"token_id": token_id, "side": side.upper()}, timeout=12)
        return float(data["price"])
    except Exception:
        return None


def clob_get_mid_price(token_id: str) -> Optional[float]:
    bid = clob_get_token_price(token_id, "SELL")
    ask = clob_get_token_price(token_id, "BUY")
    if bid is None and ask is None:
        return None
    if bid is None:
        return ask
    if ask is None:
        return bid
    return (bid + ask) / 2.0
