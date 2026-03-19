"""
data/tradier_client.py
──────────────────────
Tradier paper trading — equity + options.

Sandbox base: https://sandbox.tradier.com/v1/
Live base:    https://api.tradier.com/v1/

.env keys:
  TRADIER_ACCESS_TOKEN
  TRADIER_ACCOUNT_ID
  TRADIER_PAPER=true
"""

import re, logging
import requests

log = logging.getLogger(__name__)

SANDBOX_BASE = "https://sandbox.tradier.com/v1"
LIVE_BASE    = "https://api.tradier.com/v1"

# OCC option symbol: root (1-6 chars) + YYMMDD + C/P + 8-digit strike
_OCC = re.compile(r'^[A-Z]{1,6}\d{6}[CP]\d{8}$')


def is_option_symbol(symbol: str) -> bool:
    return bool(_OCC.match(symbol))


class TradierClient:
    def __init__(self, access_token: str, account_id: str, paper: bool = True):
        self._token      = access_token
        self._account_id = account_id
        self._base       = SANDBOX_BASE if paper else LIVE_BASE
        self._session    = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {access_token}",
            "Accept":        "application/json",
        })

    # ── HTTP helpers ──────────────────────────────

    def _get(self, path: str, params: dict = None) -> dict | None:
        try:
            r = self._session.get(f"{self._base}{path}", params=params, timeout=10)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            log.error(f"Tradier GET {path} error: {e}")
            return None

    def _post(self, path: str, data: dict) -> dict | None:
        try:
            r = self._session.post(f"{self._base}{path}", data=data, timeout=10)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            log.error(f"Tradier POST {path} error: {e}")
            return None

    # ── Account ───────────────────────────────────

    def get_balances(self) -> dict | None:
        res = self._get(f"/accounts/{self._account_id}/balances")
        return res.get("balances") if res else None

    def get_positions(self) -> list[dict]:
        res = self._get(f"/accounts/{self._account_id}/positions")
        if not res:
            return []
        pos = res.get("positions", {})
        if not pos or pos == "null":
            return []
        raw = pos.get("position", [])
        return raw if isinstance(raw, list) else [raw]

    def get_orders(self) -> list[dict]:
        res = self._get(f"/accounts/{self._account_id}/orders")
        if not res:
            return []
        orders = res.get("orders", {})
        if not orders or orders == "null":
            return []
        raw = orders.get("order", [])
        return raw if isinstance(raw, list) else [raw]

    # ── Equity orders ─────────────────────────────

    def place_equity_order(
        self,
        symbol:     str,
        side:       str,           # buy | sell | sell_short | buy_to_cover
        quantity:   int,
        order_type: str   = "market",
        price:      float = None,
        stop:       float = None,
        duration:   str   = "day",
        tag:        str   = "tradeagent",
    ) -> dict | None:
        data = {
            "class":    "equity",
            "symbol":   symbol.upper(),
            "side":     side,
            "quantity": str(quantity),
            "type":     order_type,
            "duration": duration,
            "tag":      tag,
        }
        if price is not None:
            data["price"] = str(round(price, 2))
        if stop is not None:
            data["stop"] = str(round(stop, 2))
        res = self._post(f"/accounts/{self._account_id}/orders", data)
        if res and res.get("order", {}).get("status") == "ok":
            log.info(f"Equity order: {side} {quantity}x {symbol}")
        return res

    # ── Option orders ─────────────────────────────

    def place_option_order(
        self,
        option_symbol: str,        # full OCC symbol e.g. NVDA260321C00500000
        side:          str,        # buy_to_open | sell_to_close | sell_to_open | buy_to_close
        quantity:      int,
        order_type:    str   = "limit",
        price:         float = None,
        duration:      str   = "day",
        tag:           str   = "tradeagent",
    ) -> dict | None:
        # Tradier requires `symbol` = underlying root, `option_symbol` = full OCC string
        _occ_match = re.match(r'^([A-Z]{1,6})\d{6}[CP]\d{8}$', option_symbol)
        underlying = _occ_match.group(1) if _occ_match else option_symbol
        data = {
            "class":         "option",
            "symbol":        underlying,
            "option_symbol": option_symbol,
            "side":          side,
            "quantity":      str(quantity),
            "type":          order_type,
            "duration":      duration,
            "tag":           tag,
        }
        if price is not None:
            data["price"] = str(round(price, 2))
        res = self._post(f"/accounts/{self._account_id}/orders", data)
        if res:
            status = res.get("order", {}).get("status")
            if status == "ok":
                log.info(f"Option order placed: {side} {quantity}x {option_symbol} @ ${price}")
            else:
                log.warning(f"Option order rejected: {res}")
        return res

    # ── Close position ────────────────────────────

    def close_position(self, symbol: str, quantity: int) -> dict | None:
        """
        Close an open position at market price.
        Handles both equity and options automatically.
        quantity: positive = long position, negative = short position.
        """
        if is_option_symbol(symbol):
            side = "sell_to_close" if quantity > 0 else "buy_to_close"
            return self.place_option_order(symbol, side, abs(quantity), "market")
        else:
            side = "sell" if quantity > 0 else "buy"
            return self.place_equity_order(symbol, side, abs(quantity), "market")

    def cancel_order(self, order_id: int) -> dict | None:
        try:
            r = self._session.delete(
                f"{self._base}/accounts/{self._account_id}/orders/{order_id}",
                timeout=10,
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            log.error(f"Tradier cancel {order_id} error: {e}")
            return None

    # ── Options chain ─────────────────────────────

    def get_option_expirations(self, symbol: str) -> list[str]:
        """List of available expiration dates for a symbol."""
        res = self._get("/markets/options/expirations",
                        params={"symbol": symbol, "includeAllRoots": "true"})
        if not res:
            return []
        exp = res.get("expirations", {})
        if not exp or exp == "null":
            return []
        dates = exp.get("date", [])
        return dates if isinstance(dates, list) else [dates]

    def get_option_chain(self, symbol: str, expiration: str) -> list[dict]:
        """Full option chain (calls + puts) for a symbol and expiration."""
        res = self._get("/markets/options/chains",
                        params={"symbol": symbol, "expiration": expiration, "greeks": "true"})
        if not res:
            return []
        opts = res.get("options", {})
        if not opts or opts == "null":
            return []
        raw = opts.get("option", [])
        return raw if isinstance(raw, list) else [raw]

    def get_option_quote(self, option_symbol: str) -> dict | None:
        """Current bid/ask/last for a single option contract."""
        res = self._get("/markets/quotes",
                        params={"symbols": option_symbol, "greeks": "true"})
        if not res:
            return None
        quotes = res.get("quotes", {})
        if not quotes:
            return None
        q = quotes.get("quote", {})
        return q if isinstance(q, dict) else None
