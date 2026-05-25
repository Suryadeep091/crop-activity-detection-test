import math
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


ACTIVE_ACTIVITY_THRESHOLD = 30.0
REVIEW_ACTIVITY_THRESHOLD = 15.0
NON_CROP_DOMINANCE_THRESHOLD = 60.0
LOW_CROP_PROBABILITY_THRESHOLD = 25.0
STRONG_CROP_PROBABILITY_THRESHOLD = 45.0

DW_CLASSES = [
    'water', 'trees', 'grass', 'flooded_vegetation',
    'crops', 'shrub_and_scrub', 'built', 'bare', 'snow_and_ice'
]

HARD_VETO_CLASSES = {'water', 'built', 'bare', 'snow_and_ice'}
SOFT_NON_CROP_CLASSES = {'trees', 'grass', 'flooded_vegetation', 'shrub_and_scrub'}
NON_CROP_DOMINANT_CLASSES = HARD_VETO_CLASSES | SOFT_NON_CROP_CLASSES


def _as_dataframe(data: Any) -> pd.DataFrame:
    if data is None:
        return pd.DataFrame()
    if isinstance(data, pd.DataFrame):
        return data.copy()
    return pd.DataFrame(data)


def _clip(value: float, low: float = 0.0, high: float = 100.0) -> float:
    if pd.isna(value) or np.isinf(value):
        return low
    return float(min(max(value, low), high))


def _mean_percent(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    return _clip(float(pd.to_numeric(series, errors='coerce').fillna(0).mean() * 100))


def summarize_land_cover(land_cover_data: Any) -> Dict[str, Any]:
    df = _as_dataframe(land_cover_data)
    available_cols = [c for c in DW_CLASSES if c in df.columns]
    if df.empty or not available_cols:
        return {
            "dominant_class": None,
            "dominant_percent": 0.0,
            "runner_up_class": None,
            "runner_up_percent": 0.0,
            "crop_probability_mean": 0.0,
            "crop_probability_median": 0.0,
            "crop_dominance_percent": 0.0,
            "hard_non_crop_dominance_percent": 0.0,
            "soft_non_crop_dominance_percent": 0.0,
            "top_class_margin_percent": 0.0,
            "class_means": {},
        }

    numeric_df = df[available_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    dominant_classes = numeric_df.idxmax(axis=1)
    dominance_share = dominant_classes.value_counts(normalize=True)

    dominant_class = dominance_share.index[0] if not dominance_share.empty else None
    dominant_percent = float(dominance_share.iloc[0] * 100) if not dominance_share.empty else 0.0
    runner_up_class = dominance_share.index[1] if len(dominance_share) > 1 else None
    runner_up_percent = float(dominance_share.iloc[1] * 100) if len(dominance_share) > 1 else 0.0

    crop_probability = numeric_df.get('crops', pd.Series(0, index=numeric_df.index))
    crop_probability_mean = _mean_percent(crop_probability)
    crop_probability_median = _clip(float(pd.to_numeric(crop_probability, errors='coerce').fillna(0).median() * 100))
    crop_dominance_percent = float((dominant_classes == 'crops').mean() * 100)
    hard_non_crop_dominance_percent = float(dominant_classes.isin(HARD_VETO_CLASSES).mean() * 100)
    soft_non_crop_dominance_percent = float(dominant_classes.isin(SOFT_NON_CROP_CLASSES).mean() * 100)

    class_means = {
        c: _mean_percent(numeric_df[c])
        for c in available_cols
    }

    return {
        "dominant_class": dominant_class,
        "dominant_percent": _clip(dominant_percent),
        "runner_up_class": runner_up_class,
        "runner_up_percent": _clip(runner_up_percent),
        "crop_probability_mean": crop_probability_mean,
        "crop_probability_median": crop_probability_median,
        "crop_dominance_percent": _clip(crop_dominance_percent),
        "hard_non_crop_dominance_percent": _clip(hard_non_crop_dominance_percent),
        "soft_non_crop_dominance_percent": _clip(soft_non_crop_dominance_percent),
        "top_class_margin_percent": _clip(dominant_percent - runner_up_percent),
        "class_means": class_means,
    }


def summarize_raw_observations(raw_vegetation_data: Any, daily_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    df = _as_dataframe(raw_vegetation_data)
    daily_count = len(daily_data) if daily_data is not None else 0

    if df.empty:
        return {
            "raw_s2_count": 0,
            "raw_s1_count": 0,
            "months_with_s2": 0,
            "months_with_s1": 0,
            "interpolation_ratio": 1.0 if daily_count else 0.0,
            "raw_observation_count": 0,
        }

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

    raw_s2_mask = pd.Series(False, index=df.index)
    if 'NDVI' in df.columns:
        raw_s2_mask = raw_s2_mask | df['NDVI'].notna()
    if 'EVI' in df.columns:
        raw_s2_mask = raw_s2_mask | df['EVI'].notna()

    raw_s1_mask = pd.Series(False, index=df.index)
    if 'RVI' in df.columns:
        raw_s1_mask = raw_s1_mask | df['RVI'].notna()

    raw_s2_count = int(raw_s2_mask.sum())
    raw_s1_count = int(raw_s1_mask.sum())
    months_with_s2 = int(df.loc[raw_s2_mask, 'date'].dt.to_period('M').nunique()) if 'date' in df.columns and raw_s2_count else 0
    months_with_s1 = int(df.loc[raw_s1_mask, 'date'].dt.to_period('M').nunique()) if 'date' in df.columns and raw_s1_count else 0
    raw_observation_count = max(raw_s2_count, raw_s1_count)
    interpolation_ratio = 0.0
    if daily_count:
        interpolation_ratio = _clip(1 - min(raw_observation_count / daily_count, 1), 0, 1)

    return {
        "raw_s2_count": raw_s2_count,
        "raw_s1_count": raw_s1_count,
        "months_with_s2": months_with_s2,
        "months_with_s1": months_with_s1,
        "interpolation_ratio": round(float(interpolation_ratio), 3),
        "raw_observation_count": raw_observation_count,
    }


def summarize_vegetation(daily_data: Optional[pd.DataFrame]) -> Dict[str, float]:
    df = _as_dataframe(daily_data)
    if df.empty:
        return {
            "ndvi_max": 0.0, "ndvi_range": 0.0,
            "evi_max": 0.0, "rvi_max": 0.0, "rvi_range": 0.0,
            "ndvi_rvi_corr": 0.0,
        }

    def col_max(name: str) -> float:
        series = pd.to_numeric(df.get(name, pd.Series(dtype=float)), errors='coerce').dropna()
        return float(series.max()) if not series.empty else 0.0

    def col_range(name: str) -> float:
        series = pd.to_numeric(df.get(name, pd.Series(dtype=float)), errors='coerce').dropna()
        return float((series.max() - series.min()) if not series.empty else 0)

    ndvi = pd.to_numeric(df.get('NDVI', pd.Series(dtype=float)), errors='coerce')
    rvi = pd.to_numeric(df.get('RVI', pd.Series(dtype=float)), errors='coerce')
    corr = ndvi.corr(rvi) if not ndvi.empty and not rvi.empty else 0.0
    if pd.isna(corr):
        corr = 0.0

    return {
        "ndvi_max": round(col_max('NDVI'), 3),
        "ndvi_range": round(col_range('NDVI'), 3),
        "evi_max": round(col_max('EVI'), 3),
        "rvi_max": round(col_max('RVI'), 3),
        "rvi_range": round(col_range('RVI'), 3),
        "ndvi_rvi_corr": round(float(corr), 3),
    }


def compute_evidence_scores(
    activity_ratio: float,
    daily_data: Optional[pd.DataFrame],
    land_cover_data: Any,
    raw_vegetation_data: Any,
    cycle_info: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    landcover = summarize_land_cover(land_cover_data)
    raw_quality = summarize_raw_observations(raw_vegetation_data, daily_data)
    vegetation = summarize_vegetation(daily_data)
    cycle_info = cycle_info or {}

    total_cycles = int(cycle_info.get("total_cycles", 0) or 0)
    cycle_confidences = [
        float(c.get("confidence", 0))
        for c in cycle_info.get("details", [])
        if isinstance(c, dict) and c.get("confidence") is not None
    ]
    avg_cycle_conf = float(np.mean(cycle_confidences)) if cycle_confidences else 0.0
    detected_seasons = cycle_info.get("detected_seasons", []) or []

    phenology_score = _clip((min(total_cycles, 3) / 3) * 70 + avg_cycle_conf * 0.30)
    if "Perennial/Evergreen" in detected_seasons:
        phenology_score = max(phenology_score, 35.0)

    optical_score = _clip(
        vegetation["ndvi_max"] * 55
        + vegetation["evi_max"] * 35
        + min(vegetation["ndvi_range"] / 0.45, 1) * 25
    )
    radar_score = _clip(
        vegetation["rvi_max"] * 55
        + min(vegetation["rvi_range"] / 0.45, 1) * 25
        + max(vegetation["ndvi_rvi_corr"], 0) * 20
    )
    landcover_score = _clip(
        landcover["crop_probability_mean"] * 0.65
        + landcover["crop_dominance_percent"] * 0.35
        - landcover["hard_non_crop_dominance_percent"] * 0.35
        - landcover["soft_non_crop_dominance_percent"] * 0.20
    )
    quality_score = _clip(
        min(raw_quality["raw_s2_count"] / 30, 1) * 35
        + min(raw_quality["raw_s1_count"] / 20, 1) * 35
        + min((raw_quality["months_with_s2"] + raw_quality["months_with_s1"]) / 18, 1) * 30
    )

    conflict_score = _clip(
        landcover["hard_non_crop_dominance_percent"] * 0.65
        + landcover["soft_non_crop_dominance_percent"] * 0.40
        + max(0, LOW_CROP_PROBABILITY_THRESHOLD - landcover["crop_probability_mean"]) * 0.80
        - phenology_score * 0.20
    )

    composite_score = _clip(
        phenology_score * 0.30
        + optical_score * 0.20
        + radar_score * 0.20
        + landcover_score * 0.20
        + quality_score * 0.10
        - conflict_score * 0.15
    )

    return {
        "phenology_score": round(phenology_score, 2),
        "optical_score": round(optical_score, 2),
        "radar_score": round(radar_score, 2),
        "landcover_score": round(landcover_score, 2),
        "quality_score": round(quality_score, 2),
        "conflict_score": round(conflict_score, 2),
        "composite_score": round(composite_score, 2),
        "landcover": landcover,
        "raw_quality": raw_quality,
        "vegetation": vegetation,
    }


def decide_parcel(
    activity_ratio: float,
    daily_data: Optional[pd.DataFrame],
    land_cover_data: Any,
    raw_vegetation_data: Any,
    cycle_info: Optional[Dict[str, Any]],
    p1_crop_mean: Optional[float] = None,
    p1_nocrop_mean: Optional[float] = None,
    p2_crop_mean: Optional[float] = None,
    p2_nocrop_mean: Optional[float] = None,
) -> Dict[str, Any]:
    evidence = compute_evidence_scores(
        activity_ratio=activity_ratio,
        daily_data=daily_data,
        land_cover_data=land_cover_data,
        raw_vegetation_data=raw_vegetation_data,
        cycle_info=cycle_info,
    )
    landcover = evidence["landcover"]
    reasons: List[str] = []
    review_reasons: List[str] = []

    dominant_class = landcover["dominant_class"]
    strong_hard_veto = (
        dominant_class in HARD_VETO_CLASSES
        and landcover["dominant_percent"] >= NON_CROP_DOMINANCE_THRESHOLD
        and landcover["crop_probability_mean"] <= LOW_CROP_PROBABILITY_THRESHOLD
    )
    noncrop_dominance_veto = (
        dominant_class in NON_CROP_DOMINANT_CLASSES
        and landcover["dominant_percent"] >= NON_CROP_DOMINANCE_THRESHOLD
        and landcover["crop_probability_mean"] <= LOW_CROP_PROBABILITY_THRESHOLD
    )
    mixed_vegetation = (
        dominant_class in SOFT_NON_CROP_CLASSES
        and landcover["dominant_percent"] >= 45
        and landcover["crop_probability_mean"] <= STRONG_CROP_PROBABILITY_THRESHOLD
        and evidence["phenology_score"] < 70
    )
    low_quality = evidence["quality_score"] < 35
    high_activity_low_confidence = activity_ratio > ACTIVE_ACTIVITY_THRESHOLD and evidence["composite_score"] < 45
    perennial_candidate = (
        dominant_class in {'trees', 'shrub_and_scrub'}
        and landcover["dominant_percent"] >= 60
        and evidence["phenology_score"] < 50
    )

    if strong_hard_veto:
        reasons.append(f"Strong {dominant_class} dominance with weak crop probability")
    if noncrop_dominance_veto:
        reasons.append("Non-crop land cover dominates the year")
    if mixed_vegetation:
        review_reasons.append("Mixed vegetation signal: crop activity competes with trees/shrub/grass")
    if low_quality:
        review_reasons.append("Low raw observation support for a confident parcel verdict")
    if high_activity_low_confidence:
        review_reasons.append("High active-day ratio but weak combined evidence confidence")
    if perennial_candidate:
        review_reasons.append("Tree/shrub dominant perennial-or-orchard pattern")

    if activity_ratio <= REVIEW_ACTIVITY_THRESHOLD and evidence["composite_score"] < 55:
        decision_label = "Inactive"
        is_active = False
        reasons.append("Low active-day ratio and weak crop evidence")
    elif strong_hard_veto:
        decision_label = "Inactive"
        is_active = False
    elif mixed_vegetation or high_activity_low_confidence or perennial_candidate:
        decision_label = "Review"
        is_active = False
    elif activity_ratio > ACTIVE_ACTIVITY_THRESHOLD and evidence["composite_score"] >= 45:
        decision_label = "Active"
        is_active = True
        reasons.append("Active-day ratio and combined crop evidence support cultivation")
    elif activity_ratio > REVIEW_ACTIVITY_THRESHOLD and evidence["composite_score"] >= 55:
        decision_label = "Review"
        is_active = False
        review_reasons.append("Moderate activity with supportive evidence but below active threshold")
    else:
        decision_label = "Inactive"
        is_active = False
        reasons.append("Crop evidence does not pass activity threshold")

    if p1_crop_mean is not None and p2_crop_mean is not None:
        p1_win = p1_crop_mean if is_active else (p1_nocrop_mean if p1_nocrop_mean is not None else 100 - p1_crop_mean)
        p2_win = p2_crop_mean if is_active else (p2_nocrop_mean if p2_nocrop_mean is not None else 100 - p2_crop_mean)
        model_confidence = math.sqrt(max(p1_win, 0) * max(p2_win, 0))
    else:
        model_confidence = evidence["composite_score"]

    if decision_label == "Review":
        final_confidence = min(model_confidence, max(35.0, evidence["composite_score"]))
    else:
        final_confidence = max(model_confidence * 0.55 + evidence["composite_score"] * 0.45, 0)

    reason_list: List[str] = reasons + review_reasons
    if not reason_list:
        reason_list.append("Evidence is internally consistent")

    return {
        "decision_label": decision_label,
        "is_active": bool(is_active),
        "decision_reason": "; ".join(reason_list),
        "reason_codes": reason_list,
        "review_reasons": review_reasons,
        "noncrop_dominance_veto": bool(noncrop_dominance_veto),
        "confidence": round(_clip(final_confidence), 2),
        "evidence_scores": {
            key: evidence[key]
            for key in [
                "phenology_score", "optical_score", "radar_score",
                "landcover_score", "quality_score", "conflict_score",
                "composite_score"
            ]
        },
        "landcover_summary": evidence["landcover"],
        "data_quality": evidence["raw_quality"],
        "vegetation_summary": evidence["vegetation"],
    }


def has_noncrop_dominance_veto(land_cover_data: Any) -> bool:
    summary = summarize_land_cover(land_cover_data)
    return (
        summary["dominant_class"] in NON_CROP_DOMINANT_CLASSES
        and summary["dominant_percent"] >= NON_CROP_DOMINANCE_THRESHOLD
        and summary["crop_probability_mean"] <= LOW_CROP_PROBABILITY_THRESHOLD
    )
