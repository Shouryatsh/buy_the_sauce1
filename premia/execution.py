"""
premia/execution.py — phase 7: the IBKR **paper-trading** executor.

Hard safety rules (enforced here, not configurable):
  * connects only to port 7497 (TWS paper); the live port 7496 and IB Gateway live 4001 are refused before connecting
  * after connecting, every managed account must be a paper account (IBKR paper accounts start with "DU"); otherwise it
    disconnects and refuses to trade
  * the kill switch (persisted on disk) blocks every opening order until a person releases it
  * every opening order must pass premia.risk.check_open (portfolio limits incl. vanna/volga) and premia.risk.liquidity_ok
    (two-sided quote, open interest, spread) on the live quote; closing orders are always allowed
  * limit orders only, priced from the live bid/ask (sell: mid - s x half spread; buy: mid + s x half spread)
  * every proposal, decision, order and fill is appended to an audit log (JSONL)

Going live is out of scope (SPEC.md, conflict 10). The live account stays read-only.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Protocol

from . import risk
from .config import COSTS, RESULTS_DIR

PAPER_PORT = 7497
LIVE_PORTS = {7496, 4001}
PAPER_DIR = RESULTS_DIR / "paper"


class Refused(Exception):
    pass


@dataclass
class Proposal:
    """One option order the models want, with the evidence for it."""
    symbol: str
    right: str
    strike: float
    expiry: str                   # YYYYMMDD
    qty: int                      # signed contracts (negative = sell)
    kind: str                     # csp | cc | straddle | tail | close
    reason: str
    edge_per_share: Optional[float] = None
    fair_value: Optional[float] = None
    impact: dict = field(default_factory=dict)       # risk.Exposures fields of the trade (incl. its hedge)
    closing: bool = False


class Broker(Protocol):
    def connect(self, host: str, port: int, client_id: int) -> None: ...
    def managed_accounts(self) -> list[str]: ...
    def quote(self, symbol: str, expiry: str, strike: float, right: str) -> dict: ...   # bid, ask, open_interest
    def place_limit(self, symbol: str, expiry: str, strike: float, right: str, qty: int, limit: float, ref: str) -> dict: ...
    def disconnect(self) -> None: ...


class AuditLog:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, **data) -> None:
        rec = {"at": datetime.now().isoformat(timespec="seconds"), "event": event, **data}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]


class PersistentKillSwitch:
    """The kill switch as a file, so the CLI, the dashboard and the executor all see the same state."""

    def __init__(self, path: Path):
        self.path = path

    @property
    def engaged(self) -> bool:
        try:
            return bool(json.loads(self.path.read_text()).get("engaged"))
        except (OSError, ValueError):
            return False

    def set(self, engaged: bool, reason: str = "") -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"engaged": engaged, "reason": reason, "at": datetime.now().isoformat()}))


class PaperExecutor:
    def __init__(self, broker: Broker, base_dir: Path = PAPER_DIR, slip: float = COSTS.slippage_frac_of_half_spread):
        self.broker = broker
        self.dir = base_dir
        self.log = AuditLog(base_dir / "audit.jsonl")
        self.kill = PersistentKillSwitch(base_dir / "kill_switch.json")
        self.slip = slip
        self.connected = False

    def connect(self, host: str = "127.0.0.1", port: int = PAPER_PORT, client_id: int = 91) -> None:
        if port in LIVE_PORTS or port != PAPER_PORT:
            self.log.write("refused", why=f"port {port} is not the paper port {PAPER_PORT}")
            raise Refused(f"refusing port {port}: premia trades paper only (port {PAPER_PORT})")
        self.broker.connect(host, port, client_id)
        accounts = self.broker.managed_accounts()
        if not accounts or not all(a.upper().startswith("DU") for a in accounts):
            self.broker.disconnect()
            self.log.write("refused", why=f"accounts {accounts} are not all paper accounts")
            raise Refused(f"connected accounts {accounts} are not paper accounts (DU...): disconnected")
        self.connected = True
        self.log.write("connected", port=port, accounts=accounts)

    def submit(self, p: Proposal, exposures: risk.Exposures, nav: float) -> dict:
        """Apply every rule; place a limit order only if all pass. Returns the decision record."""
        self.log.write("proposal", **asdict(p))
        if not self.connected:
            raise Refused("not connected to a verified paper account")
        if not p.closing and self.kill.engaged:
            return self._decide(p, "blocked", ["kill switch engaged"])
        q = self.broker.quote(p.symbol, p.expiry, p.strike, p.right)
        if not p.closing:
            ok, why = risk.liquidity_ok(q.get("open_interest"), q.get("bid"), q.get("ask"))
            if not ok:
                return self._decide(p, "blocked", [why])
            ok, reasons = risk.check_open(exposures, risk.Exposures(**p.impact), nav)
            if not ok:
                return self._decide(p, "blocked", reasons)
        bid, ask = q.get("bid"), q.get("ask")
        if not (bid and ask and ask >= bid > 0):
            return self._decide(p, "blocked", ["no two-sided quote to price the limit order"])
        mid, half = 0.5 * (bid + ask), 0.5 * (ask - bid)
        limit = round(mid + self.slip * half, 2) if p.qty > 0 else round(max(mid - self.slip * half, 0.01), 2)
        ref = f"premia-{datetime.now():%Y%m%d%H%M%S}-{p.symbol}-{p.kind}"
        fill = self.broker.place_limit(p.symbol, p.expiry, p.strike, p.right, p.qty, limit, ref)
        rec = self._decide(p, "placed", [], limit=limit, ref=ref, broker=fill)
        return rec

    def _decide(self, p: Proposal, status: str, reasons: list, **extra) -> dict:
        rec = {"status": status, "reasons": reasons, "symbol": p.symbol, "kind": p.kind, "strike": p.strike,
               "expiry": p.expiry, "right": p.right, "qty": p.qty, **extra}
        self.log.write("decision", **rec)
        return rec

    def daily_check(self, nav: float, prev_nav: float, hwm: float) -> Optional[str]:
        """Trip the persisted kill switch on the daily-loss or drawdown limit. A person must release it."""
        ks = risk.KillSwitch(hwm=hwm)
        if ks.update(nav, prev_nav, datetime.now().date().isoformat()):
            self.kill.set(True, ks.reason)
            self.log.write("kill_switch", reason=ks.reason, nav=nav)
            return ks.reason
        return None


class IBPaperBroker:
    """ib_insync implementation of Broker. Only ever constructed by PaperExecutor.connect's checks (paper port)."""

    def __init__(self):
        from ib_insync import IB
        self.ib = IB()

    def connect(self, host: str, port: int, client_id: int) -> None:
        if port != PAPER_PORT:
            raise Refused("IBPaperBroker connects to the paper port only")
        self.ib.connect(host, port, clientId=client_id, timeout=15)

    def managed_accounts(self) -> list[str]:
        return list(self.ib.managedAccounts())

    def _contract(self, symbol, expiry, strike, right):
        from ib_insync import Option
        c = Option(symbol, expiry, strike, right, "SMART", currency="USD")
        q = self.ib.qualifyContracts(c)
        if not q:
            raise Refused(f"cannot qualify {symbol} {expiry} {strike}{right}")
        return q[0]

    def quote(self, symbol, expiry, strike, right) -> dict:
        c = self._contract(symbol, expiry, strike, right)
        self.ib.reqMarketDataType(1)
        t = self.ib.reqMktData(c, "101", False, False)            # 101 = option open interest
        self.ib.sleep(3)
        oi = t.putOpenInterest if right == "P" else t.callOpenInterest
        out = {"bid": t.bid if t.bid and t.bid > 0 else None, "ask": t.ask if t.ask and t.ask > 0 else None,
               "open_interest": oi if oi == oi else None}
        self.ib.cancelMktData(c)
        return out

    def place_limit(self, symbol, expiry, strike, right, qty, limit, ref) -> dict:
        from ib_insync import LimitOrder
        if not all(a.upper().startswith("DU") for a in self.ib.managedAccounts()):
            raise Refused("account changed to a non-paper account: refusing")
        c = self._contract(symbol, expiry, strike, right)
        o = LimitOrder("BUY" if qty > 0 else "SELL", abs(qty), limit)
        o.orderRef = ref
        o.tif = "DAY"
        tr = self.ib.placeOrder(c, o)
        self.ib.sleep(1)
        return {"order_id": tr.order.orderId, "status": tr.orderStatus.status}

    def disconnect(self) -> None:
        if self.ib.isConnected():
            self.ib.disconnect()
