from __future__ import annotations

import io

import cv2
import numpy as np
import pandas as pd
import streamlit as st

try:
    from scaffold_mvp.config import StandardLayoutRule
    from scaffold_mvp.image_pipeline import run_pipeline
    from scaffold_mvp.layout_engine import allocate_layout
    from scaffold_mvp.models import Segment
    from scaffold_mvp.reporting import to_dataframes, to_excel_bytes
except ModuleNotFoundError:
    # Streamlit Cloud may execute this file directly with its directory on sys.path.
    from config import StandardLayoutRule
    from image_pipeline import run_pipeline
    from layout_engine import allocate_layout
    from models import Segment
    from reporting import to_dataframes, to_excel_bytes


def _decode_image(uploaded_file) -> np.ndarray:
    content = uploaded_file.read()
    arr = np.frombuffer(content, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Image could not be decoded.")
    return image


def _editable_segments(default_segments: list[Segment]) -> list[Segment]:
    st.subheader("寸法入力（ここに平面図の数値を入力）")
    st.info("平面図の外周を面ごとに分けて、`長さ(mm)` を入力してください。開口がある面は `開口控除(mm)` も入力します。")
    rows = [
        {
            "name": seg.name,
            "length_mm": round(seg.length_mm, 1),
            "opening_deduction_mm": round(seg.opening_deduction_mm, 1),
        }
        for seg in default_segments
    ]
    df = st.data_editor(
        pd.DataFrame(rows),
        num_rows="dynamic",
        use_container_width=True,
        key="segment_editor",
        column_config={
            "name": st.column_config.TextColumn("面名", help="例: 北面 / 南面 / 東面 / 西面"),
            "length_mm": st.column_config.NumberColumn("長さ(mm)", min_value=0.0, step=100.0),
            "opening_deduction_mm": st.column_config.NumberColumn(
                "開口控除(mm)", min_value=0.0, step=50.0, help="窓・出入口などで足場不要となる長さ"
            ),
        },
    )
    segments: list[Segment] = []
    for _, row in df.iterrows():
        if not row["name"]:
            continue
        segments.append(
            Segment(
                name=str(row["name"]),
                length_mm=float(row["length_mm"]),
                opening_deduction_mm=float(row["opening_deduction_mm"]),
            )
        )
    return segments


def _build_clearance_proposal_df(segments: list[Segment], preferred_span_mm: float) -> pd.DataFrame:
    rows = []
    for seg in segments:
        effective = max(0.0, seg.length_mm - seg.opening_deduction_mm)
        if preferred_span_mm <= 0:
            scaffold_total = effective
        else:
            span_count = int(effective // preferred_span_mm)
            scaffold_total = span_count * preferred_span_mm
            if scaffold_total <= 0 and effective > 0:
                scaffold_total = effective

        side_gap = max(0.0, (effective - scaffold_total) / 2.0)
        rows.append(
            {
                "面名": seg.name,
                "入力長さ(mm)": round(seg.length_mm, 1),
                "開口控除(mm)": round(seg.opening_deduction_mm, 1),
                "有効長(mm)": round(effective, 1),
                "足場総延長提案(mm)": round(scaffold_total, 1),
                "左右離れ提案(mm)": round(side_gap, 1),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    st.set_page_config(page_title="住宅平面図から足場割付を自動提案", layout="wide")
    st.title("住宅平面図から足場割付を自動提案")
    st.caption("MVP: 手入力寸法 -> 標準ルール割付 -> 人が調整 -> 帳票出力")

    use_image = st.toggle("画像読取を使う（任意）", value=False)
    with st.expander("入力ガイド", expanded=True):
        st.markdown(
            "- 1行が1つの面です（例: 北面、東面）。\n"
            "- `長さ(mm)` はその面の総延長を入力します。\n"
            "- `開口控除(mm)` は窓や通路など、足場不要分を差し引く長さです。\n"
            "- 入力後、下に `自動提案スパン` と `部材集計` が表示されます。"
        )

    default_segments = [
        Segment(name="North", length_mm=10000.0),
        Segment(name="East", length_mm=8000.0),
        Segment(name="South", length_mm=10000.0),
        Segment(name="West", length_mm=8000.0),
    ]

    if use_image:
        uploaded = st.file_uploader("平面図画像をアップロード", type=["png", "jpg", "jpeg"])
        if uploaded:
            image = _decode_image(uploaded)
            pipe = run_pipeline(image)
            default_segments = pipe.segments or default_segments

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("入力図面")
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
            with col2:
                st.subheader("前処理結果")
                st.image(pipe.processed_image, clamp=True, use_container_width=True)
                st.metric("外周長(px)", f"{pipe.perimeter_px:.1f}")
                st.metric("縮尺(mm/px)", f"{pipe.mm_per_px:.4f}")
                st.metric("OCR検出寸法数", f"{len(pipe.dimensions)}")
        else:
            st.info("画像読取を使う場合は画像をアップロードしてください。未アップロード時は手入力で計算します。")

    st.subheader("割付ルール")
    preferred = st.number_input("優先スパン (mm)", min_value=300.0, value=1800.0, step=50.0)
    min_end = st.number_input("端部最小スパン (mm)", min_value=300.0, value=900.0, step=50.0)
    max_span = st.number_input("最大スパン (mm)", min_value=500.0, value=1800.0, step=50.0)
    rule = StandardLayoutRule(preferred_span_mm=preferred, min_end_span_mm=min_end, max_span_mm=max_span)

    segments = _editable_segments(default_segments)
    result = allocate_layout(segments, rule)
    allocations_df, materials_df = to_dataframes(result)
    clearance_df = _build_clearance_proposal_df(segments, preferred)

    st.subheader("自動提案スパン")
    st.dataframe(allocations_df, use_container_width=True)
    st.subheader("離れ提案（標準スパン優先）")
    st.caption("例: 面長10000mmで優先スパン1800mmの場合、足場総延長9000mm・左右離れ500mmを提案")
    st.dataframe(clearance_df, use_container_width=True)
    st.subheader("部材集計")
    st.dataframe(materials_df, use_container_width=True)

    csv_data = allocations_df.to_csv(index=False).encode("utf-8")
    st.download_button("スパン一覧CSVをダウンロード", data=csv_data, file_name="spans.csv", mime="text/csv")
    st.download_button(
        "帳票Excelをダウンロード",
        data=to_excel_bytes(result),
        file_name="scaffold_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


if __name__ == "__main__":
    main()
