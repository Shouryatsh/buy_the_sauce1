"""Colour system: green = profit, red = loss, amber = warning."""

GREEN = "#2ecc71"
RED = "#e74c3c"
AMBER = "#f5a623"
BLUE = "#4aa3ff"
MUTED = "#8a93a6"
BG = "#0e1117"
PLOTLY_TEMPLATE = "plotly_dark"

STATUS_COLORS = {
    "running": GREEN, "starting": AMBER, "degraded": AMBER, "error": RED, "stopped": MUTED,
    "approve": GREEN, "flag": AMBER, "block": RED,
    "filled": GREEN, "submitted": AMBER, "pending_confirmation": AMBER, "rejected": RED, "cancelled": MUTED,
}


def pnl_color(x: float) -> str:
    return GREEN if x > 0 else RED if x < 0 else MUTED


def money(x, signed: bool = False) -> str:
    if x is None:
        return "-"
    return f"{x:+,.0f}" if signed else f"{x:,.0f}"


def pct(x, digits: int = 1) -> str:
    return "-" if x is None else f"{x * 100:.{digits}f}%"


def utilization_color(u: float, warn: float = 0.8) -> str:
    return RED if u > 1.0 else AMBER if u >= warn else GREEN
