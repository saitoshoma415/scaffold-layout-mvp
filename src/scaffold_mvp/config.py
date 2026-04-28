from dataclasses import dataclass


@dataclass(frozen=True)
class StandardLayoutRule:
    preferred_span_mm: float = 1800.0
    min_end_span_mm: float = 900.0
    max_span_mm: float = 1800.0


DEFAULT_RULE = StandardLayoutRule()
