from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class Segment:
    name: str
    length_mm: float
    opening_deduction_mm: float = 0.0

    @property
    def effective_length_mm(self) -> float:
        return max(0.0, self.length_mm - self.opening_deduction_mm)


@dataclass
class SpanAllocation:
    segment_name: str
    span_lengths_mm: List[float] = field(default_factory=list)

    @property
    def span_count(self) -> int:
        return len(self.span_lengths_mm)

    @property
    def total_mm(self) -> float:
        return float(sum(self.span_lengths_mm))


@dataclass
class ScaffoldMaterialSummary:
    standards: int
    ledger_boards: int
    braces: int


@dataclass
class LayoutResult:
    allocations: List[SpanAllocation]
    material_summary: ScaffoldMaterialSummary
