"""
agents/base.py
──────────────
Shared dataclass for all agent views.
"""
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any


def _safe(v):
    """Replace NaN/Inf floats with None so they serialize to JSON null."""
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def _clean(d: dict) -> dict:
    """Recursively sanitize a dict, replacing NaN/Inf with None."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _clean(v)
        elif isinstance(v, list):
            out[k] = [_clean(i) if isinstance(i, dict) else (_safe(i) if isinstance(i, float) else i) for i in v]
        elif isinstance(v, float):
            out[k] = _safe(v)
        else:
            out[k] = v
    return out


@dataclass
class AgentView:
    agent:      str               # "momentum" | "sentiment" | "fundamental" | "quant"
    symbol:     str
    direction:  str               # "call" | "put" | "neutral"
    confidence: float             # 0.0 – 1.0
    reasons:    list[str]         = field(default_factory=list)
    data:       dict[str, Any]    = field(default_factory=dict)
    timestamp:  str               = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "agent":          self.agent,
            "symbol":         self.symbol,
            "direction":      self.direction,
            "confidence":     round(self.confidence, 3),
            "confidence_pct": round(self.confidence * 100),
            "reasons":        self.reasons,
            "data":           _clean(self.data),
            "timestamp":      self.timestamp,
        }
