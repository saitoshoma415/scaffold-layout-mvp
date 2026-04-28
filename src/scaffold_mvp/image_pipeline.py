from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import cv2
import numpy as np
import pytesseract

try:
    from .models import Segment
except ImportError:
    from models import Segment


@dataclass
class DimensionCandidate:
    text: str
    value_mm: float
    center: Tuple[float, float]
    confidence: float


@dataclass
class PipelineResult:
    processed_image: np.ndarray
    dimensions: List[DimensionCandidate]
    segments: List[Segment]
    perimeter_px: float
    mm_per_px: float


def preprocess_image(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def _extract_numeric_mm(text: str) -> float | None:
    normalized = text.replace(",", "").replace(" ", "")
    match = re.search(r"(\d{3,5})(?:mm)?", normalized, flags=re.IGNORECASE)
    if not match:
        return None
    value = float(match.group(1))
    if 100 <= value <= 50000:
        return value
    return None


def extract_dimensions(processed_image: np.ndarray) -> List[DimensionCandidate]:
    data = pytesseract.image_to_data(processed_image, output_type=pytesseract.Output.DICT, config="--psm 6")
    results: List[DimensionCandidate] = []
    for i, text in enumerate(data["text"]):
        text = (text or "").strip()
        if not text:
            continue
        value = _extract_numeric_mm(text)
        if value is None:
            continue
        conf = float(data["conf"][i]) if data["conf"][i] != "-1" else 0.0
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        results.append(
            DimensionCandidate(
                text=text,
                value_mm=value,
                center=(x + w / 2.0, y + h / 2.0),
                confidence=conf,
            )
        )
    return sorted(results, key=lambda d: d.confidence, reverse=True)


def _largest_contour_perimeter(processed_image: np.ndarray) -> float:
    # Invert to treat lines as foreground.
    inv = 255 - processed_image
    contours, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    largest = max(contours, key=cv2.contourArea)
    return float(cv2.arcLength(largest, closed=True))


def estimate_mm_per_px(dimensions: Iterable[DimensionCandidate], perimeter_px: float) -> float:
    dims = list(dimensions)
    if not dims or perimeter_px <= 0:
        return 1.0
    median_mm = float(np.median([d.value_mm for d in dims]))
    # Use a conservative heuristic that median dimension spans about a quarter of visible perimeter.
    estimated_span_px = max(10.0, perimeter_px / 4.0)
    return median_mm / estimated_span_px


def infer_segments(perimeter_px: float, mm_per_px: float) -> List[Segment]:
    perimeter_mm = perimeter_px * mm_per_px
    if perimeter_mm <= 0:
        return []
    # MVP assumes a simple rectangle-equivalent split by four sides.
    side = perimeter_mm / 4.0
    return [
        Segment(name="North", length_mm=side),
        Segment(name="East", length_mm=side),
        Segment(name="South", length_mm=side),
        Segment(name="West", length_mm=side),
    ]


def run_pipeline(image_bgr: np.ndarray) -> PipelineResult:
    processed = preprocess_image(image_bgr)
    dimensions = extract_dimensions(processed)
    perimeter_px = _largest_contour_perimeter(processed)
    mm_per_px = estimate_mm_per_px(dimensions, perimeter_px)
    segments = infer_segments(perimeter_px, mm_per_px)
    return PipelineResult(
        processed_image=processed,
        dimensions=dimensions,
        segments=segments,
        perimeter_px=perimeter_px,
        mm_per_px=mm_per_px,
    )
