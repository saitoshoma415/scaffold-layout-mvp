from __future__ import annotations

from io import BytesIO
from typing import Dict, List

import pandas as pd

from .models import LayoutResult


def allocation_rows(result: LayoutResult) -> List[Dict[str, float | str | int]]:
    rows: List[Dict[str, float | str | int]] = []
    for allocation in result.allocations:
        for i, span in enumerate(allocation.span_lengths_mm, start=1):
            rows.append(
                {
                    "segment": allocation.segment_name,
                    "span_index": i,
                    "span_mm": round(span, 1),
                }
            )
    return rows


def material_rows(result: LayoutResult) -> List[Dict[str, int | str]]:
    ms = result.material_summary
    return [
        {"material": "Standard", "qty": ms.standards},
        {"material": "LedgerBoard", "qty": ms.ledger_boards},
        {"material": "Brace", "qty": ms.braces},
    ]


def to_dataframes(result: LayoutResult) -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.DataFrame(allocation_rows(result)), pd.DataFrame(material_rows(result))


def to_excel_bytes(result: LayoutResult) -> bytes:
    allocations_df, materials_df = to_dataframes(result)
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        allocations_df.to_excel(writer, index=False, sheet_name="spans")
        materials_df.to_excel(writer, index=False, sheet_name="materials")
    return output.getvalue()
