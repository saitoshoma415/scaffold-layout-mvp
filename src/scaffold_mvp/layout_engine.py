from __future__ import annotations

from typing import List

try:
    from .config import StandardLayoutRule
    from .models import LayoutResult, ScaffoldMaterialSummary, Segment, SpanAllocation
except ImportError:
    from config import StandardLayoutRule
    from models import LayoutResult, ScaffoldMaterialSummary, Segment, SpanAllocation


def _split_segment_length(total_mm: float, rule: StandardLayoutRule) -> List[float]:
    if total_mm <= 0:
        return []

    full = int(total_mm // rule.preferred_span_mm)
    remainder = total_mm - full * rule.preferred_span_mm
    spans = [rule.preferred_span_mm] * full

    if remainder <= 1e-6:
        return spans

    if remainder >= rule.min_end_span_mm:
        spans.append(remainder)
        return spans

    if not spans:
        # When total is less than preferred span, keep single span.
        return [total_mm]

    # Borrow length from the previous span to satisfy minimum end span.
    need = rule.min_end_span_mm - remainder
    donor = spans[-1] - need
    if donor < rule.min_end_span_mm:
        # Fallback: distribute across all spans evenly.
        count = len(spans) + 1
        even = total_mm / count
        return [even] * count

    spans[-1] = donor
    spans.append(rule.min_end_span_mm)
    return spans


def _material_summary(allocations: List[SpanAllocation]) -> ScaffoldMaterialSummary:
    span_count = sum(a.span_count for a in allocations)
    standards = span_count + len(allocations)  # include segment boundary posts
    ledger_boards = span_count * 2
    braces = max(1, span_count // 3)
    return ScaffoldMaterialSummary(
        standards=standards,
        ledger_boards=ledger_boards,
        braces=braces,
    )


def allocate_layout(segments: List[Segment], rule: StandardLayoutRule) -> LayoutResult:
    allocations: List[SpanAllocation] = []
    for segment in segments:
        effective = segment.effective_length_mm
        spans = _split_segment_length(effective, rule)
        allocations.append(SpanAllocation(segment_name=segment.name, span_lengths_mm=spans))
    summary = _material_summary(allocations)
    return LayoutResult(allocations=allocations, material_summary=summary)
