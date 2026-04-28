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


def _segments_to_house_rows(segments: list[Segment]) -> list[dict]:
    return [
        {
            "name": seg.name,
            "length_mm": round(seg.length_mm, 1),
            "opening_deduction_mm": round(seg.opening_deduction_mm, 1),
        }
        for seg in segments
    ]


def _edit_house_dimensions(default_segments: list[Segment]) -> pd.DataFrame:
    st.subheader("1) 家の寸法を入力")
    st.info("外周を面ごとに分けて `長さ(mm)` を入力します。開口がある面は `開口控除(mm)` も入力します。")
    rows = _segments_to_house_rows(default_segments)
    return st.data_editor(
        pd.DataFrame(rows),
        num_rows="dynamic",
        use_container_width=True,
        key="house_editor",
        column_config={
            "name": st.column_config.TextColumn("面名", help="例: 北面 / 南面 / 東面 / 西面"),
            "length_mm": st.column_config.NumberColumn("長さ(mm)", min_value=0.0, step=100.0),
            "opening_deduction_mm": st.column_config.NumberColumn(
                "開口控除(mm)", min_value=0.0, step=50.0, help="窓・出入口などで足場不要となる長さ"
            ),
        },
    )


def _edit_site_constraints(house_df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("2) 敷地条件を入力")
    st.info("`敷地側最大離れ(mm)` は、その面で左右に取れる離れの上限（敷地・隣地・障害物など）です。")
    base_names = [str(x) for x in house_df["name"].tolist() if str(x)]
    rows = [{"name": name, "site_max_gap_mm": None, "site_min_gap_mm": None} for name in base_names]
    return st.data_editor(
        pd.DataFrame(rows),
        num_rows="dynamic",
        use_container_width=True,
        key="site_editor",
        column_config={
            "name": st.column_config.TextColumn("面名", help="1) の面名と一致させてください"),
            "site_max_gap_mm": st.column_config.NumberColumn(
                "敷地側最大離れ(mm)",
                min_value=0.0,
                step=50.0,
                help="離れの上限。未入力なら制約なし",
            ),
            "site_min_gap_mm": st.column_config.NumberColumn(
                "敷地側最小離れ(mm)",
                min_value=0.0,
                step=50.0,
                help="離れの下限（任意）。未入力なら0扱い",
            ),
        },
    )


def _merge_house_and_site(house_df: pd.DataFrame, site_df: pd.DataFrame) -> list[Segment]:
    house_by_name: dict[str, dict] = {}
    for _, row in house_df.iterrows():
        name = str(row["name"]).strip()
        if not name:
            continue
        house_by_name[name] = row

    site_by_name: dict[str, dict] = {}
    for _, row in site_df.iterrows():
        name = str(row["name"]).strip()
        if not name:
            continue
        site_by_name[name] = row

    missing_site = sorted(set(house_by_name.keys()) - set(site_by_name.keys()))
    extra_site = sorted(set(site_by_name.keys()) - set(house_by_name.keys()))
    if missing_site:
        st.warning("敷地表に面名が無い行があります: " + ", ".join(missing_site))
    if extra_site:
        st.warning("家の寸法表に無い面名が敷地表にあります: " + ", ".join(extra_site))

    segments: list[Segment] = []
    for name, hrow in house_by_name.items():
        srow = site_by_name.get(name, {})
        site_max = srow.get("site_max_gap_mm") if isinstance(srow, dict) else None
        site_min = srow.get("site_min_gap_mm") if isinstance(srow, dict) else None
        segments.append(
            Segment(
                name=name,
                length_mm=float(hrow["length_mm"]),
                opening_deduction_mm=float(hrow["opening_deduction_mm"]),
                site_max_gap_mm=None if pd.isna(site_max) else float(site_max),
                site_min_gap_mm=None if pd.isna(site_min) else float(site_min),
            )
        )
    return segments


def _distance_to_intervals(value: float, intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    distances: list[float] = []
    for lo, hi in intervals:
        if lo <= value <= hi:
            return 0.0
        if value < lo:
            distances.append(lo - value)
        else:
            distances.append(value - hi)
    return float(min(distances))


def _choose_span_for_site_and_anti(
    effective_length_mm: float,
    preferred_span_mm: float,
    wide_min_mm: float,
    wide_max_mm: float,
    narrow_min_mm: float,
    narrow_max_mm: float,
    site_max_gap_mm: float | None,
    site_min_gap_mm: float | None,
) -> tuple[float, float, str, str]:
    if effective_length_mm <= 0 or preferred_span_mm <= 0:
        return effective_length_mm, 0.0, "対象外", ""

    max_count = int(effective_length_mm // preferred_span_mm)
    if max_count <= 0:
        return effective_length_mm, 0.0, "スパン不足", ""

    site_max = float(site_max_gap_mm) if site_max_gap_mm is not None else float("inf")
    site_min = float(site_min_gap_mm) if site_min_gap_mm is not None else 0.0

    anti_intervals = [(wide_min_mm, wide_max_mm), (narrow_min_mm, narrow_max_mm)]
    wide_center = (wide_min_mm + wide_max_mm) / 2.0
    narrow_center = (narrow_min_mm + narrow_max_mm) / 2.0

    best: tuple[float, float, float, str, str] | None = None
    for count in range(max_count, 0, -1):
        scaffold_total = count * preferred_span_mm
        side_gap = (effective_length_mm - scaffold_total) / 2.0
        if side_gap < 0:
            continue

        site_penalty = 0.0
        if side_gap > site_max:
            site_penalty += (side_gap - site_max) * 1_000_000.0
        if side_gap < site_min:
            site_penalty += (site_min - side_gap) * 1_000_000.0

        anti_type = ""
        anti_penalty = 0.0
        if wide_min_mm <= side_gap <= wide_max_mm:
            anti_type = "広いアンチ"
            anti_penalty = abs(side_gap - wide_center)
        elif narrow_min_mm <= side_gap <= narrow_max_mm:
            anti_type = "狭いアンチ"
            anti_penalty = abs(side_gap - narrow_center)
        else:
            anti_type = "範囲外"
            anti_penalty = _distance_to_intervals(side_gap, anti_intervals) * 10_000.0

        score = site_penalty + anti_penalty
        candidate = (score, scaffold_total, side_gap, anti_type, "")
        if best is None or score < best[0]:
            best = candidate

    if best is None:
        base_count = int(effective_length_mm // preferred_span_mm)
        scaffold_total = base_count * preferred_span_mm if base_count > 0 else effective_length_mm
        side_gap = max(0.0, (effective_length_mm - scaffold_total) / 2.0)
        return scaffold_total, side_gap, "範囲外", ""

    _, scaffold_total, side_gap, anti_type, _ = best

    notes: list[str] = []
    if site_max_gap_mm is not None and side_gap > site_max + 1e-6:
        notes.append("敷地上限超過")
    if site_min_gap_mm is not None and side_gap + 1e-6 < site_min:
        notes.append("敷地下限未達")

    return scaffold_total, side_gap, anti_type, " / ".join(notes)


def _build_clearance_proposal_df(
    segments: list[Segment],
    preferred_span_mm: float,
    wide_min_mm: float,
    wide_max_mm: float,
    narrow_min_mm: float,
    narrow_max_mm: float,
) -> pd.DataFrame:
    rows = []
    for seg in segments:
        effective = max(0.0, seg.length_mm - seg.opening_deduction_mm)
        scaffold_total, side_gap, anti_type, site_note = _choose_span_for_site_and_anti(
            effective,
            preferred_span_mm,
            wide_min_mm,
            wide_max_mm,
            narrow_min_mm,
            narrow_max_mm,
            seg.site_max_gap_mm,
            seg.site_min_gap_mm,
        )
        rows.append(
            {
                "面名": seg.name,
                "入力長さ(mm)": round(seg.length_mm, 1),
                "開口控除(mm)": round(seg.opening_deduction_mm, 1),
                "敷地側最大離れ(mm)": "" if seg.site_max_gap_mm is None else round(float(seg.site_max_gap_mm), 1),
                "敷地側最小離れ(mm)": "" if seg.site_min_gap_mm is None else round(float(seg.site_min_gap_mm), 1),
                "有効長(mm)": round(effective, 1),
                "足場総延長提案(mm)": round(scaffold_total, 1),
                "左右離れ提案(mm)": round(side_gap, 1),
                "アンチ提案": anti_type,
                "敷地メモ": site_note,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    st.set_page_config(page_title="住宅平面図から足場割付を自動提案", layout="wide")
    st.title("住宅平面図から足場割付を自動提案")
    st.caption("MVP: 家の寸法 -> 敷地 -> 離れ -> 割付・帳票")

    with st.expander("入力ガイド（図面作成の流れ）", expanded=True):
        st.markdown(
            "1. 家の寸法（面ごとの延長）を入力\n"
            "2. 敷地条件（面ごとの離れの上限/下限）を入力\n"
            "3. 離れ提案を確認（アンチ幅レンジと敷地条件を両立するよう自動計算）\n"
            "4. 割付・帳票を出力"
        )

    default_segments = [
        Segment(name="North", length_mm=10000.0),
        Segment(name="East", length_mm=8000.0),
        Segment(name="South", length_mm=10000.0),
        Segment(name="West", length_mm=8000.0),
    ]

    house_df = _edit_house_dimensions(default_segments)
    site_df = _edit_site_constraints(house_df)
    segments = _merge_house_and_site(house_df, site_df)

    use_image = st.toggle("画像読取を使う（任意）", value=False)
    if use_image:
        uploaded = st.file_uploader("平面図画像をアップロード", type=["png", "jpg", "jpeg"])
        if uploaded:
            image = _decode_image(uploaded)
            pipe = run_pipeline(image)
            preview_segments = pipe.segments or default_segments

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
            st.caption("※ 画像から推定した寸法は参考です。必要に応じて 1) の表を手で修正してください。")
            with st.expander("画像推定の寸法プレビュー（任意）", expanded=False):
                st.dataframe(pd.DataFrame(_segments_to_house_rows(preview_segments)), use_container_width=True)
        else:
            st.info("画像読取を使う場合は画像をアップロードしてください（任意）。")

    st.subheader("3) 離れ計算の条件")
    preferred = st.number_input("優先スパン (mm)", min_value=300.0, value=1800.0, step=50.0)
    min_end = st.number_input("端部最小スパン (mm)", min_value=300.0, value=900.0, step=50.0)
    max_span = st.number_input("最大スパン (mm)", min_value=500.0, value=1800.0, step=50.0)
    rule = StandardLayoutRule(preferred_span_mm=preferred, min_end_span_mm=min_end, max_span_mm=max_span)
    st.caption("アンチ幅条件")
    c1, c2 = st.columns(2)
    with c1:
        wide_min = st.number_input("広いアンチ最小(mm)", min_value=0.0, value=730.0, step=10.0)
        wide_max = st.number_input("広いアンチ最大(mm)", min_value=0.0, value=1050.0, step=10.0)
    with c2:
        narrow_min = st.number_input("狭いアンチ最小(mm)", min_value=0.0, value=430.0, step=10.0)
        narrow_max = st.number_input("狭いアンチ最大(mm)", min_value=0.0, value=650.0, step=10.0)

    clearance_df = _build_clearance_proposal_df(
        segments,
        preferred,
        wide_min,
        wide_max,
        narrow_min,
        narrow_max,
    )

    st.subheader("離れ提案（標準スパン優先）")
    st.caption("広いアンチ(730-1050) / 狭いアンチ(430-650) に収まる離れを優先し、敷地の上限/下限も考慮します")
    st.dataframe(clearance_df, use_container_width=True)

    st.subheader("4) 割付・帳票")
    result = allocate_layout(segments, rule)
    allocations_df, materials_df = to_dataframes(result)
    st.subheader("自動提案スパン")
    st.dataframe(allocations_df, use_container_width=True)
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
