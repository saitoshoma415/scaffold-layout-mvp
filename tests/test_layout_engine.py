from scaffold_mvp.config import StandardLayoutRule
from scaffold_mvp.layout_engine import allocate_layout
from scaffold_mvp.models import Segment
from scaffold_mvp.reporting import to_dataframes, to_excel_bytes


def test_allocate_layout_respects_min_end_span() -> None:
    rule = StandardLayoutRule(preferred_span_mm=1800, min_end_span_mm=900, max_span_mm=1800)
    segments = [Segment(name="North", length_mm=5000)]
    result = allocate_layout(segments, rule)
    spans = result.allocations[0].span_lengths_mm
    assert round(sum(spans), 3) == 5000
    assert all(span >= 900 for span in spans)


def test_reporting_tables_have_expected_columns() -> None:
    rule = StandardLayoutRule()
    result = allocate_layout([Segment(name="A", length_mm=3600)], rule)
    allocations_df, materials_df = to_dataframes(result)
    assert {"segment", "span_index", "span_mm"} <= set(allocations_df.columns)
    assert {"material", "qty"} <= set(materials_df.columns)


def test_excel_export_returns_binary_payload() -> None:
    result = allocate_layout([Segment(name="A", length_mm=3600)], StandardLayoutRule())
    data = to_excel_bytes(result)
    assert isinstance(data, bytes)
    assert len(data) > 100
