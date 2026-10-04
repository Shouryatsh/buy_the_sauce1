"""
wheelcli/data/ibkr.py — IBKR market-data access via ib_insync.

Architecture
------------
IBKRClient
  connect() / disconnect() / context-manager
  get_underlying_price(symbol)           → float | None
  get_option_expirations(symbol, ...)    → list[str]   (YYYYMMDD)
  get_strikes_for_expiry(symbol, expiry) → list[float]
  get_put_contracts(symbol, expiry, ...) → list[OptionContract]

Pacing strategy
---------------
IBKR allows ≈ 50 simultaneous market-data lines per account.  This client
batches option requests in groups of ``ibkr_batch_size`` (default 40) and
sleeps ``ibkr_batch_delay`` seconds between batches plus an additional
``ibkr_mktdata_wait`` seconds after the final request to give IBKR's model
time to compute implied-volatility and greeks before we read them.

Paper vs live
-------------
Default port 7497 → TWS paper trading.
To use a live account set WHEEL_IBKR_PORT=7496 (TWS live) or 4002 (Gateway).
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Optional

from ib_insync import IB, Option, Stock

from ..config import WheelConfig
from ..models import OptionContract

logger = logging.getLogger(__name__)


class IBKRClient:
    """
    Synchronous IBKR data client built on ``ib_insync``.

    All ``ib.sleep()`` calls drive the ib_insync event loop, allowing
    callbacks (e.g. market-data ticks) to be processed while we wait.
    """

    def __init__(self, config: WheelConfig) -> None:
        self.config = config
        self.ib = IB()

    # ── Connection ─────────────────────────────────────────────────────────

    def connect(self) -> None:
        """Connect to IB Gateway / TWS.  Raises on failure."""
        self.ib.connect(
            self.config.ibkr_host,
            self.config.ibkr_port,
            clientId=self.config.ibkr_client_id,
            timeout=self.config.ibkr_connect_timeout,
        )
        logger.info(
            "Connected to IBKR at %s:%d (clientId=%d)",
            self.config.ibkr_host,
            self.config.ibkr_port,
            self.config.ibkr_client_id,
        )

    def disconnect(self) -> None:
        if self.ib.isConnected():
            self.ib.disconnect()
            logger.info("Disconnected from IBKR.")

    def __enter__(self) -> "IBKRClient":
        self.connect()
        return self

    def __exit__(self, *_: object) -> None:
        self.disconnect()

    # ── Underlying price ──────────────────────────────────────────────────

    def get_underlying_price(self, symbol: str) -> Optional[float]:
        """
        Fetch the current market price for the underlying stock.

        Returns None if the price cannot be obtained (e.g. outside market hours
        and no last/delayed price is available).
        """
        contract = Stock(symbol, "SMART", "USD")
        try:
            qualified = self.ib.qualifyContracts(contract)
            if not qualified:
                logger.warning("%s: contract could not be qualified.", symbol)
                return None
        except Exception as exc:
            logger.warning("%s: qualifyContracts failed: %s", symbol, exc)
            return None

        ticker = self.ib.reqMktData(contract, "", False, False)
        self.ib.sleep(2.0)
        price = ticker.marketPrice()
        self.ib.cancelMktData(contract)

        if price is not None and price > 0:
            logger.debug("%s spot = %.2f", symbol, price)
            return float(price)

        # Fallback: try last/close
        for fallback in (ticker.last, ticker.close):
            if fallback is not None and fallback > 0:
                logger.debug("%s spot (fallback) = %.2f", symbol, fallback)
                return float(fallback)

        logger.warning("%s: no valid price available.", symbol)
        return None

    # ── Option chain metadata ─────────────────────────────────────────────

    def get_option_expirations(
        self, symbol: str, max_dte: int, weekly_only: bool
    ) -> list[str]:
        """
        Return valid option expiration strings (YYYYMMDD) within *max_dte*
        calendar days, optionally filtered to Friday-only (weekly) expirations.

        Uses ``reqSecDefOptParams`` which does not count against the
        simultaneous market-data limit.
        """
        try:
            chains = self.ib.reqSecDefOptParams(symbol, "", "STK", 0)
        except Exception as exc:
            logger.warning("%s: reqSecDefOptParams failed: %s", symbol, exc)
            return []

        if not chains:
            logger.warning("%s: no option chains returned.", symbol)
            return []

        today = date.today()
        cutoff = today + timedelta(days=max_dte)

        # Aggregate expirations across all exchanges / trading classes
        all_exps: set[str] = set()
        for chain in chains:
            for exp_str in chain.expirations:
                try:
                    exp_date = datetime.strptime(exp_str, "%Y%m%d").date()
                except ValueError:
                    continue
                if today < exp_date <= cutoff:
                    all_exps.add(exp_str)

        exps = sorted(all_exps)

        if weekly_only:
            exps = [
                e
                for e in exps
                if datetime.strptime(e, "%Y%m%d").weekday() == 4  # Friday
            ]

        logger.debug("%s: %d expirations found (weekly_only=%s)", symbol, len(exps), weekly_only)
        return exps

    def get_strikes_for_expiry(self, symbol: str, expiry: str) -> list[float]:
        """
        Return sorted list of available strikes for *symbol* at *expiry* (YYYYMMDD).

        Aggregates across all option chains (exchanges / trading classes).
        """
        try:
            chains = self.ib.reqSecDefOptParams(symbol, "", "STK", 0)
        except Exception as exc:
            logger.warning("%s: reqSecDefOptParams failed: %s", symbol, exc)
            return []

        strikes: set[float] = set()
        for chain in chains:
            if expiry in chain.expirations:
                strikes.update(chain.strikes)

        return sorted(strikes)

    # ── Option market data ────────────────────────────────────────────────

    def get_put_contracts(
        self,
        symbol: str,
        expiry: str,
        spot: float,
        strikes: list[float],
    ) -> list[OptionContract]:
        """
        Fetch bid/ask/delta/IV for put options at *strikes* for one *expiry*.

        Requests are sent in batches to respect IBKR pacing rules.
        Greeks are read from ``ticker.modelGreeks`` (IBKR theoretical model).

        Parameters
        ----------
        symbol  : underlying ticker
        expiry  : expiration string in YYYYMMDD format
        spot    : current underlying price (used to filter ITM strikes)
        strikes : list of strikes to request (pre-filtered to desired range)
        """
        if not strikes:
            return []

        exp_date = datetime.strptime(expiry, "%Y%m%d").date()
        dte = (exp_date - date.today()).days

        # Build IBKR Option objects
        ibkr_opts: list[Option] = []
        for K in strikes:
            ibkr_opts.append(Option(symbol, expiry, K, "P", "SMART"))

        # Qualify contracts (fills conId; skip un-qualifiable ones)
        try:
            qualified = self.ib.qualifyContracts(*ibkr_opts)
        except Exception as exc:
            logger.warning("%s %s: qualifyContracts failed: %s", symbol, expiry, exc)
            return []

        if not qualified:
            return []

        results: list[OptionContract] = []
        batch_size = self.config.ibkr_batch_size
        wait_after = self.config.ibkr_batch_delay + self.config.ibkr_mktdata_wait

        for i in range(0, len(qualified), batch_size):
            batch = qualified[i : i + batch_size]

            # Subscribe to market data for the batch
            tickers = [
                self.ib.reqMktData(c, "100,101,106", False, False) for c in batch
            ]

            # Wait for ticks + model greeks
            self.ib.sleep(wait_after)

            for contract, ticker in zip(batch, tickers):
                self.ib.cancelMktData(contract)
                results.append(
                    self._parse_ticker(symbol, contract.strike, exp_date, dte, ticker)
                )

            # Extra inter-batch pause (already included in wait_after for first batch)
            if i + batch_size < len(qualified):
                self.ib.sleep(self.config.ibkr_batch_delay)

        valid = [c for c in results if c.mid is not None and c.mid > 0]
        logger.debug(
            "%s %s: %d/%d contracts have valid mid price",
            symbol,
            expiry,
            len(valid),
            len(results),
        )
        return results

    # ── Internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _parse_ticker(
        symbol: str,
        strike: float,
        exp_date: date,
        dte: int,
        ticker: object,  # ib_insync Ticker
    ) -> OptionContract:
        """
        Extract bid, ask, mid, delta, and IV from an ib_insync Ticker object.

        Priority for greeks: modelGreeks → lastGreeks → None.
        Mid is computed as (bid + ask) / 2; falls back to last trade.
        """
        bid: Optional[float] = None
        ask: Optional[float] = None
        mid: Optional[float] = None
        delta: Optional[float] = None
        iv: Optional[float] = None
        oi: Optional[int] = None
        vol: Optional[int] = None

        raw_bid = getattr(ticker, "bid", None)
        raw_ask = getattr(ticker, "ask", None)

        if raw_bid is not None and raw_bid > 0:
            bid = float(raw_bid)
        if raw_ask is not None and raw_ask > 0:
            ask = float(raw_ask)

        if bid is not None and ask is not None:
            mid = (bid + ask) / 2.0
        else:
            last = getattr(ticker, "last", None)
            if last is not None and last > 0:
                mid = float(last)

        # Greeks
        greeks = getattr(ticker, "modelGreeks", None) or getattr(
            ticker, "lastGreeks", None
        )
        if greeks is not None:
            raw_delta = getattr(greeks, "delta", None)
            raw_iv = getattr(greeks, "impliedVol", None)
            if raw_delta is not None:
                delta = abs(float(raw_delta))  # always positive
            if raw_iv is not None and raw_iv > 0:
                iv = float(raw_iv)

        raw_oi = getattr(ticker, "openInterest", None)
        raw_vol = getattr(ticker, "volume", None)
        if raw_oi is not None and raw_oi >= 0:
            oi = int(raw_oi)
        if raw_vol is not None and raw_vol >= 0:
            vol = int(raw_vol)

        return OptionContract(
            symbol=symbol,
            expiry=exp_date,
            strike=strike,
            right="P",
            delta=delta,
            iv=iv,
            bid=bid,
            ask=ask,
            mid=mid,
            open_interest=oi,
            volume=vol,
            dte=dte,
        )
