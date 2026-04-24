import asyncio
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pydantic import BaseModel, Field
from tools.get_trend_data_tool import _get_trend_data_impl


class TrendSegment(BaseModel):
    start_time_ms: str
    end_time_ms: str
    start_index: int
    end_index: int
    point_count: int

    mean: float | None = None
    min: float | None = None
    max: float | None = None
    std: float | None = None
    range: float | None = None
    mad: float | None = None

    slope: float | None = None
    relative_slope: float | None = None
    slope_label: str = "unknown"

    level_label: str = "unknown"
    volatility_label: str = "unknown"
    pattern_label: str = "unknown"

    mean_shift_vs_prev: float | None = None
    volatility_shift_vs_prev: float | None = None
    slope_shift_vs_prev: float | None = None


class TrendFeatureDetail(BaseModel):
    # 1. 水平统计特征
    current: float | None = None
    mean: float | None = None
    median: float | None = None
    p95: float | None = None
    min: float | None = None
    max: float | None = None
    std: float | None = None

    # 2. 波动统计特征
    coefficient_of_variation: float | None = None
    range: float | None = None
    mad: float | None = None

    # 3. 多尺度变化速率特征
    slope: float | None = None
    relative_slope: float | None = None
    short_slope: float | None = None
    medium_slope: float | None = None
    long_slope: float | None = None
    trend_class: str = "unknown"
    change_rate_grade: str = "unknown"

    # 4. 变点检测特征
    changepoints: list[dict[str, Any]] = Field(default_factory=list)
    changepoint_severity: str = "none"
    step_change_magnitude: float | None = None
    step_change_relative: float | None = None

    # 5. 越限与告警特征
    alarm_status: str = "normal"
    over_threshold_time: float | None = None
    max_over_threshold_duration: float | None = None
    over_threshold_ratio: float | None = None

    # 6. 异常段特征
    outliers: list[dict[str, Any]] = Field(default_factory=list)
    outlier_clusters: list[dict[str, Any]] = Field(default_factory=list)
    spike_detected: bool = False
    recovery_after_spike: bool = False
    transient_events: list[dict[str, Any]] = Field(default_factory=list)

    # 7. 新增：趋势分段与状态特征
    segment_stats: list[TrendSegment] = Field(default_factory=list)
    dominant_pattern: str | None = None
    level_regime: str | None = None
    volatility_regime: str | None = None
    overall_direction: str | None = None
    narrative_summary: str | None = None

    # 8. 新增：分类后的时间点
    changepoint_time_ms: list[str] = Field(default_factory=list)
    outlier_time_ms: list[str] = Field(default_factory=list)
    transient_event_time_ms: list[str] = Field(default_factory=list)


class TrendPointAnalysisResult(BaseModel):
    component_id: str = Field(description="测点 ID")
    features: list[str] = Field(default_factory=list, description="当前测点的特征值字段")
    feature_stats: dict[str, TrendFeatureDetail] = Field(description="当前测点各特征的趋势特征")

    # 兼容旧版 test_trend.py，继续保留
    anomaly_time_ms: list[str] = Field(description="当前测点识别出的关键异常/变点时间点毫秒时间戳")

    summary: list[str] = Field(description="当前测点的趋势概括")
    notable_points: list[str] = Field(description="当前测点的显著时间点与波动说明")


class TrendAnalysisResult(BaseModel):
    component_ids: list[str] = Field(description="测点 ID 列表")
    start_time: str = Field(description="开始时间，毫秒时间戳")
    end_time: str = Field(description="结束时间，毫秒时间戳")
    component_features: dict[str, list[str]] = Field(description="各测点对应的特征值字段")
    point_results: list[TrendPointAnalysisResult] = Field(description="每个测点各自的趋势分析结果")


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.fmean(values))


def _safe_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(statistics.pstdev(values))


def _round_float(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    vals = sorted(values)
    pos = (len(vals) - 1) * q
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))
    if lower == upper:
        return float(vals[lower])
    weight = pos - lower
    return float(vals[lower] * (1 - weight) + vals[upper] * weight)


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def _mad(values: list[float]) -> float | None:
    if not values:
        return None
    med = _median(values)
    if med is None:
        return None
    deviations = [abs(v - med) for v in values]
    return float(statistics.median(deviations))


def _parse_time_ms(value: str) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_feature_series(
    point_data: list[dict[str, Any]],
    feature: str,
) -> list[tuple[str, float]]:
    series: list[tuple[str, float]] = []
    for item in point_data:
        time_ms = str(item.get("time_ms") or "")
        values = item.get("values") or {}
        if not isinstance(values, dict):
            continue
        raw = values.get(feature)
        if isinstance(raw, (int, float)) and math.isfinite(raw):
            series.append((time_ms, float(raw)))
    return series


def _infer_thresholds(point_data: list[dict[str, Any]]) -> tuple[float | None, float | None, float | None, float | None]:
    h_alarm = None
    hh_alarm = None
    l_alarm = None
    ll_alarm = None

    for item in point_data:
        for key in ("h_alarm", "hh_alarm", "l_alarm", "ll_alarm"):
            raw = item.get(key)
            if isinstance(raw, (int, float)) and math.isfinite(raw):
                if key == "h_alarm":
                    h_alarm = float(raw)
                elif key == "hh_alarm":
                    hh_alarm = float(raw)
                elif key == "l_alarm":
                    l_alarm = float(raw)
                elif key == "ll_alarm":
                    ll_alarm = float(raw)
    return h_alarm, hh_alarm, l_alarm, ll_alarm


def _moving_average(values: list[float], window: int) -> list[float]:
    if not values:
        return []
    if window <= 1:
        return [float(v) for v in values]

    n = len(values)
    half = window // 2
    result: list[float] = []
    for i in range(n):
        left = max(0, i - half)
        right = min(n, i + half + 1)
        result.append(float(statistics.fmean(values[left:right])))
    return result


def _rolling_std(values: list[float], window: int) -> list[float]:
    if not values:
        return []
    if window <= 1:
        return [0.0 for _ in values]

    n = len(values)
    half = window // 2
    result: list[float] = []
    for i in range(n):
        left = max(0, i - half)
        right = min(n, i + half + 1)
        result.append(_safe_std(values[left:right]))
    return result


def _rolling_slope(series: list[tuple[str, float]], window: int) -> list[float]:
    if not series:
        return []
    if window <= 2:
        return [0.0 for _ in series]

    n = len(series)
    half = window // 2
    result: list[float] = []

    for i in range(n):
        left = max(0, i - half)
        right = min(n, i + half + 1)
        sub = series[left:right]
        slope = _calc_slope(sub)
        result.append(float(slope or 0.0))

    return result


def _calc_slope_from_xy(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None

    x_mean = _safe_mean(xs)
    y_mean = _safe_mean(ys)
    if x_mean is None or y_mean is None:
        return None

    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if abs(denominator) < 1e-12:
        return None

    return float(numerator / denominator)


def _calc_slope(series: list[tuple[str, float]]) -> float | None:
    if len(series) < 2:
        return None

    xs: list[float] = []
    ys: list[float] = []
    for ts, value in series:
        ts_i = _parse_time_ms(ts)
        if ts_i is None:
            continue
        xs.append(float(ts_i) / 1000.0)
        ys.append(value)

    return _calc_slope_from_xy(xs, ys)


def _calc_smoothed_slope(series: list[tuple[str, float]], window: int) -> float | None:
    if len(series) < 2:
        return None

    xs: list[float] = []
    ys_raw: list[float] = []
    for ts, value in series:
        ts_i = _parse_time_ms(ts)
        if ts_i is None:
            continue
        xs.append(float(ts_i) / 1000.0)
        ys_raw.append(value)

    if len(xs) < 2:
        return None

    ys = _moving_average(ys_raw, max(1, window))
    return _calc_slope_from_xy(xs, ys)


def _classify_trend_from_relative_slope(relative_slope: float | None) -> tuple[str, str]:
    if relative_slope is None:
        return "unknown", "unknown"

    abs_rs = abs(relative_slope)
    if abs_rs <= 1e-6:
        trend_class = "stable"
    elif relative_slope > 0:
        trend_class = "increasing"
    else:
        trend_class = "decreasing"

    if abs_rs >= 8e-5:
        grade = "rapid"
    elif abs_rs >= 2e-5:
        grade = "moderate"
    else:
        grade = "slow"

    return trend_class, grade


def _classify_trend(slope: float | None, mean_value: float | None) -> tuple[str, str]:
    if slope is None or mean_value is None:
        return "unknown", "unknown"

    baseline = abs(mean_value) if abs(mean_value) > 1e-12 else 1.0
    relative_slope = slope / baseline
    return _classify_trend_from_relative_slope(relative_slope)


def _label_level(segment_mean: float, global_mean: float, global_std: float) -> str:
    std_ref = global_std if global_std > 1e-12 else max(abs(global_mean) * 0.02, 1e-6)
    if segment_mean >= global_mean + 0.6 * std_ref:
        return "high"
    if segment_mean <= global_mean - 0.6 * std_ref:
        return "low"
    return "mid"


def _label_volatility(segment_std: float, global_std: float) -> str:
    std_ref = global_std if global_std > 1e-12 else max(segment_std, 1e-6)
    ratio = segment_std / std_ref if std_ref > 1e-12 else 1.0
    if ratio >= 1.2:
        return "wide"
    if ratio <= 0.8:
        return "narrow"
    return "normal"


def _label_segment_pattern(level_label: str, volatility_label: str, slope_label: str) -> str:
    if level_label == "high" and slope_label == "stable" and volatility_label in {"normal", "narrow"}:
        return "high_level_running"
    if slope_label == "increasing" and volatility_label == "narrow":
        return "narrowing_and_rising"
    if slope_label == "increasing":
        return "rising"
    if slope_label == "decreasing" and volatility_label == "wide":
        return "wide_fluctuating_and_falling"
    if slope_label == "decreasing":
        return "falling"
    if volatility_label == "wide":
        return "wide_fluctuating"
    if volatility_label == "narrow":
        return "narrow_fluctuating"
    return "stable_platform"


def _merge_small_breakpoints(
    candidates: list[tuple[int, dict[str, Any]]],
    n: int,
    min_seg_len: int,
) -> list[tuple[int, dict[str, Any]]]:
    if not candidates:
        return []

    points = sorted(candidates, key=lambda x: x[0])
    merged: list[tuple[int, dict[str, Any]]] = []
    prev = 0

    for idx, payload in points:
        if not (0 < idx < n):
            continue
        if idx - prev < min_seg_len:
            continue
        merged.append((idx, payload))
        prev = idx

    if merged and n - merged[-1][0] < min_seg_len:
        merged.pop()

    return merged


def _segment_series(
    series: list[tuple[str, float]],
) -> tuple[list[TrendSegment], list[dict[str, Any]], str | None, str | None, str | None, str | None]:
    if len(series) < 8:
        return [], [], None, None, None, None

    times = [ts for ts, _ in series]
    values = [v for _, v in series]

    global_mean = _safe_mean(values) or 0.0
    global_std = _safe_std(values)
    baseline = abs(global_mean) if abs(global_mean) > 1e-12 else 1.0
    n = len(values)

    feature_window = max(5, min(15, max(5, n // 12)))
    smooth_long = _moving_average(values, feature_window)
    rolling_vol = _rolling_std(values, feature_window)
    rolling_slopes = _rolling_slope(series, feature_window)

    diff_level = [abs(smooth_long[i] - smooth_long[i - 1]) for i in range(1, n)]
    diff_vol = [abs(rolling_vol[i] - rolling_vol[i - 1]) for i in range(1, n)]
    diff_slope = [abs(rolling_slopes[i] - rolling_slopes[i - 1]) for i in range(1, n)]

    level_threshold = max(global_std * 0.18, baseline * 0.01)
    vol_threshold = max(global_std * 0.12, baseline * 0.006)
    slope_threshold = max(abs(_safe_mean([abs(x) for x in rolling_slopes]) or 0.0) * 3.0, 1e-7)

    candidate_breakpoints: list[tuple[int, dict[str, Any]]] = []

    for i in range(1, n - 1):
        level_jump = diff_level[i - 1]
        vol_jump = diff_vol[i - 1]
        slope_jump = diff_slope[i - 1]

        chosen_type = None
        chosen_mag = 0.0

        if level_jump >= level_threshold:
            chosen_type = "level_shift"
            chosen_mag = level_jump

        if vol_jump >= vol_threshold and vol_jump > chosen_mag:
            chosen_type = "volatility_shift"
            chosen_mag = vol_jump

        if slope_jump >= slope_threshold and slope_jump > chosen_mag:
            chosen_type = "slope_shift"
            chosen_mag = slope_jump

        if chosen_type is not None:
            score = chosen_mag / max(global_std, baseline * 0.01, 1e-6)
            payload = {
                "time_ms": times[i],
                "type": chosen_type,
                "magnitude": _round_float(chosen_mag, 6),
                "score": _round_float(score, 6),
                "relative_change": _round_float(chosen_mag / baseline, 6),
            }
            candidate_breakpoints.append((i, payload))

    min_seg_len = max(8, n // 10)
    kept = _merge_small_breakpoints(candidate_breakpoints, n, min_seg_len)

    boundaries = [0] + [idx for idx, _ in kept] + [n]
    changepoints = [payload for _, payload in kept]

    segments: list[TrendSegment] = []
    prev_seg_mean = None
    prev_seg_std = None
    prev_seg_slope = None

    for idx in range(len(boundaries) - 1):
        start = boundaries[idx]
        end = boundaries[idx + 1]
        seg_values = values[start:end]
        seg_times = times[start:end]
        if len(seg_values) < 2:
            continue

        seg_mean = _safe_mean(seg_values)
        seg_std = _safe_std(seg_values)
        seg_min = min(seg_values)
        seg_max = max(seg_values)
        seg_range = seg_max - seg_min
        seg_mad = _mad(seg_values)

        seg_series = series[start:end]
        seg_slope = _calc_slope(seg_series)
        seg_relative_slope = None
        if seg_slope is not None and seg_mean is not None and abs(seg_mean) > 1e-12:
            seg_relative_slope = seg_slope / abs(seg_mean)

        slope_label, _ = _classify_trend_from_relative_slope(seg_relative_slope)
        level_label = _label_level(seg_mean or 0.0, global_mean, global_std)
        volatility_label = _label_volatility(seg_std, global_std)
        pattern_label = _label_segment_pattern(level_label, volatility_label, slope_label)

        mean_shift_vs_prev = None
        volatility_shift_vs_prev = None
        slope_shift_vs_prev = None

        if prev_seg_mean is not None and seg_mean is not None:
            mean_shift_vs_prev = seg_mean - prev_seg_mean
        if prev_seg_std is not None:
            volatility_shift_vs_prev = seg_std - prev_seg_std
        if prev_seg_slope is not None and seg_slope is not None:
            slope_shift_vs_prev = seg_slope - prev_seg_slope

        segments.append(
            TrendSegment(
                start_time_ms=seg_times[0],
                end_time_ms=seg_times[-1],
                start_index=start,
                end_index=end - 1,
                point_count=len(seg_values),
                mean=_round_float(seg_mean, 6),
                min=_round_float(seg_min, 6),
                max=_round_float(seg_max, 6),
                std=_round_float(seg_std, 6),
                range=_round_float(seg_range, 6),
                mad=_round_float(seg_mad, 6),
                slope=_round_float(seg_slope, 9),
                relative_slope=_round_float(seg_relative_slope, 9),
                slope_label=slope_label,
                level_label=level_label,
                volatility_label=volatility_label,
                pattern_label=pattern_label,
                mean_shift_vs_prev=_round_float(mean_shift_vs_prev, 6),
                volatility_shift_vs_prev=_round_float(volatility_shift_vs_prev, 6),
                slope_shift_vs_prev=_round_float(slope_shift_vs_prev, 9),
            )
        )

        prev_seg_mean = seg_mean
        prev_seg_std = seg_std
        prev_seg_slope = seg_slope

    if not segments:
        return [], [], None, None, None, None

    dominant_pattern = max(
        (seg.pattern_label for seg in segments),
        key=lambda name: sum(s.point_count for s in segments if s.pattern_label == name),
        default=None,
    )
    level_regime = max(
        (seg.level_label for seg in segments),
        key=lambda name: sum(s.point_count for s in segments if s.level_label == name),
        default=None,
    )
    volatility_regime = max(
        (seg.volatility_label for seg in segments),
        key=lambda name: sum(s.point_count for s in segments if s.volatility_label == name),
        default=None,
    )

    first_mean = segments[0].mean
    last_mean = segments[-1].mean
    overall_direction = None
    if first_mean is not None and last_mean is not None:
        delta = last_mean - first_mean
        if abs(delta) <= max(global_std * 0.2, baseline * 0.01):
            overall_direction = "stable"
        elif delta > 0:
            overall_direction = "up"
        else:
            overall_direction = "down"

    return segments, changepoints[:8], dominant_pattern, level_regime, volatility_regime, overall_direction


def _detect_residual_outliers(series: list[tuple[str, float]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    基于“相对局部趋势残差”检测离群点，避免把阶段性抬升/下降整体误判为异常。
    """
    values = [v for _, v in series]
    if len(values) < 7:
        return [], []

    n = len(values)
    window = max(5, min(15, max(5, n // 12)))
    smooth = _moving_average(values, window)
    residuals = [v - s for v, s in zip(values, smooth)]

    med = _median(residuals)
    mad_v = _mad(residuals)
    if med is None or mad_v is None or mad_v <= 1e-12:
        return [], []

    outliers: list[dict[str, Any]] = []
    for (ts, value), resid in zip(series, residuals):
        robust_z = 0.6745 * (resid - med) / mad_v
        if abs(robust_z) >= 3.5:
            outliers.append(
                {
                    "time_ms": ts,
                    "value": _round_float(value, 6),
                    "residual": _round_float(resid, 6),
                    "z_score": _round_float(robust_z, 6),
                }
            )

    clusters: list[dict[str, Any]] = []
    if outliers:
        current_cluster = [outliers[0]]
        for item in outliers[1:]:
            prev_ts = _parse_time_ms(str(current_cluster[-1]["time_ms"]))
            curr_ts = _parse_time_ms(str(item["time_ms"]))
            if prev_ts is not None and curr_ts is not None and (curr_ts - prev_ts) <= 10 * 60 * 1000:
                current_cluster.append(item)
            else:
                clusters.append(
                    {
                        "start_time_ms": current_cluster[0]["time_ms"],
                        "end_time_ms": current_cluster[-1]["time_ms"],
                        "count": len(current_cluster),
                    }
                )
                current_cluster = [item]
        clusters.append(
            {
                "start_time_ms": current_cluster[0]["time_ms"],
                "end_time_ms": current_cluster[-1]["time_ms"],
                "count": len(current_cluster),
            }
        )

    return outliers[:20], clusters[:10]


def _detect_spike_and_recovery(series: list[tuple[str, float]]) -> tuple[bool, bool, list[dict[str, Any]]]:
    values = [v for _, v in series]
    if len(values) < 4:
        return False, False, []

    diffs = [values[i] - values[i - 1] for i in range(1, len(values))]
    diff_std = _safe_std(diffs)
    if diff_std <= 1e-12:
        return False, False, []

    transient_events: list[dict[str, Any]] = []
    spike_detected = False
    recovery_after_spike = False

    for i in range(1, len(values) - 1):
        rise = values[i] - values[i - 1]
        fall = values[i + 1] - values[i]
        if abs(rise) >= 3 * diff_std:
            event_type = "spike_up" if rise > 0 else "spike_down"
            spike_detected = True

            recovered = False
            if rise > 0 and fall < 0 and abs(fall) >= abs(rise) * 0.5:
                recovered = True
            if rise < 0 and fall > 0 and abs(fall) >= abs(rise) * 0.5:
                recovered = True

            if recovered:
                recovery_after_spike = True

            transient_events.append(
                {
                    "time_ms": series[i][0],
                    "type": event_type,
                    "magnitude": _round_float(rise, 6),
                    "recovered": recovered,
                }
            )

    return spike_detected, recovery_after_spike, transient_events[:10]


def _calc_over_threshold_metrics(
    series: list[tuple[str, float]],
    h_alarm: float | None,
    hh_alarm: float | None,
    l_alarm: float | None,
    ll_alarm: float | None,
) -> tuple[str, float | None, float | None, float | None]:
    if len(series) < 2:
        current = series[-1][1] if series else None
        if current is None:
            return "normal", None, None, None
        if hh_alarm is not None and current >= hh_alarm:
            return "HH", 0.0, 0.0, 0.0
        if h_alarm is not None and current >= h_alarm:
            return "H", 0.0, 0.0, 0.0
        if ll_alarm is not None and current <= ll_alarm:
            return "LL", 0.0, 0.0, 0.0
        if l_alarm is not None and current <= l_alarm:
            return "L", 0.0, 0.0, 0.0
        return "normal", 0.0, 0.0, 0.0

    current_value = series[-1][1]
    if hh_alarm is not None and current_value >= hh_alarm:
        alarm_status = "HH"
    elif h_alarm is not None and current_value >= h_alarm:
        alarm_status = "H"
    elif ll_alarm is not None and current_value <= ll_alarm:
        alarm_status = "LL"
    elif l_alarm is not None and current_value <= l_alarm:
        alarm_status = "L"
    else:
        alarm_status = "normal"

    def is_over(v: float) -> bool:
        if hh_alarm is not None and v >= hh_alarm:
            return True
        if h_alarm is not None and v >= h_alarm:
            return True
        if ll_alarm is not None and v <= ll_alarm:
            return True
        if l_alarm is not None and v <= l_alarm:
            return True
        return False

    total_duration = 0.0
    current_run = 0.0
    max_run = 0.0
    total_window = 0.0

    for i in range(1, len(series)):
        prev_ts = _parse_time_ms(series[i - 1][0])
        curr_ts = _parse_time_ms(series[i][0])
        if prev_ts is None or curr_ts is None or curr_ts <= prev_ts:
            continue

        dt = (curr_ts - prev_ts) / 1000.0
        total_window += dt

        over_prev = is_over(series[i - 1][1])
        over_curr = is_over(series[i][1])

        if over_prev and over_curr:
            total_duration += dt
            current_run += dt
        else:
            if current_run > max_run:
                max_run = current_run
            current_run = 0.0

    if current_run > max_run:
        max_run = current_run

    ratio = (total_duration / total_window) if total_window > 0 else 0.0
    return (
        alarm_status,
        _round_float(total_duration, 3),
        _round_float(max_run, 3),
        _round_float(ratio, 6),
    )


def _segment_to_phrase(seg: TrendSegment) -> str:
    phrase_parts: list[str] = []

    if seg.level_label == "high":
        phrase_parts.append("高位")
    elif seg.level_label == "low":
        phrase_parts.append("低位")
    else:
        phrase_parts.append("中位")

    if seg.pattern_label == "high_level_running":
        phrase_parts.append("运行")
        return "".join(phrase_parts)

    if seg.volatility_label == "wide":
        phrase_parts.append("宽幅波动")
    elif seg.volatility_label == "narrow":
        phrase_parts.append("窄幅波动")
    else:
        phrase_parts.append("波动")

    if seg.slope_label == "increasing":
        phrase_parts.append("并上升")
    elif seg.slope_label == "decreasing":
        phrase_parts.append("并回落")
    else:
        if seg.pattern_label == "stable_platform":
            phrase_parts.append("较平稳")

    return "".join(phrase_parts)


def _build_narrative_summary(
    segments: list[TrendSegment],
    overall_direction: str | None,
    dominant_pattern: str | None,
    volatility_regime: str | None,
) -> str | None:
    if not segments:
        return None

    phrases = [_segment_to_phrase(seg) for seg in segments[:4]]
    first_sentence = "整体"
    if overall_direction == "up":
        first_sentence += "呈阶段性上行，"
    elif overall_direction == "down":
        first_sentence += "呈阶段性回落，"
    else:
        first_sentence += "呈分阶段波动，"

    first_sentence += "、".join(phrases)

    if dominant_pattern == "high_level_running":
        first_sentence += "，以高位运行阶段为主"
    elif dominant_pattern == "wide_fluctuating":
        first_sentence += "，以宽幅波动为主"
    elif dominant_pattern == "narrowing_and_rising":
        first_sentence += "，中段存在收窄并抬升特征"

    if volatility_regime == "wide":
        first_sentence += "，整体波动偏大"
    elif volatility_regime == "narrow":
        first_sentence += "，整体波动相对收敛"

    return first_sentence


def _calc_changepoint_severity(
    changepoints: list[dict[str, Any]],
) -> tuple[str, float | None, float | None]:
    if not changepoints:
        return "none", None, None

    main_cp = changepoints[0]
    magnitude = None
    relative = None

    raw_mag = main_cp.get("magnitude")
    if isinstance(raw_mag, (int, float)):
        magnitude = float(raw_mag)

    raw_rel = main_cp.get("relative_change")
    if isinstance(raw_rel, (int, float)):
        relative = float(raw_rel)

    severity = "minor"
    if relative is not None:
        abs_rel = abs(relative)
        if abs_rel >= 0.30:
            severity = "critical"
        elif abs_rel >= 0.10:
            severity = "major"

    return severity, _round_float(magnitude, 6), _round_float(relative, 6)


def _dedupe_time_list(items: list[str], limit: int) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for ts in items:
        if ts and ts not in seen:
            seen.add(ts)
            result.append(ts)
        if len(result) >= limit:
            break
    return result


def _build_feature_detail(point_data: list[dict[str, Any]], series: list[tuple[str, float]]) -> TrendFeatureDetail:
    values = [v for _, v in series]
    if not values:
        return TrendFeatureDetail()

    current = values[-1]
    mean_v = _safe_mean(values)
    median_v = _median(values)
    p95_v = _percentile(values, 0.95)
    min_v = min(values)
    max_v = max(values)
    std_v = _safe_std(values)
    range_v = max_v - min_v
    mad_v = _mad(values)

    cov = None
    if mean_v is not None and abs(mean_v) > 1e-12:
        cov = std_v / abs(mean_v)

    slope = _calc_slope(series)
    relative_slope = None
    if slope is not None and mean_v is not None and abs(mean_v) > 1e-12:
        relative_slope = slope / abs(mean_v)

    n = len(series)
    short_window = max(3, min(5, n // 24 if n >= 24 else 3))
    medium_window = max(5, min(9, n // 12 if n >= 12 else 5))
    long_window = max(9, min(15, n // 6 if n >= 6 else 9))

    short_slope = _calc_smoothed_slope(series, short_window)
    medium_slope = _calc_smoothed_slope(series, medium_window)
    long_slope = _calc_smoothed_slope(series, long_window)

    trend_class, change_rate_grade = _classify_trend(long_slope if long_slope is not None else slope, mean_v)

    segments, regime_changepoints, dominant_pattern, level_regime, volatility_regime, overall_direction = _segment_series(series)
    cp_severity, step_mag, step_rel = _calc_changepoint_severity(regime_changepoints)

    outliers, outlier_clusters = _detect_residual_outliers(series)
    spike_detected, recovery_after_spike, transient_events = _detect_spike_and_recovery(series)

    h_alarm, hh_alarm, l_alarm, ll_alarm = _infer_thresholds(point_data)
    alarm_status, over_threshold_time, max_over_threshold_duration, over_threshold_ratio = _calc_over_threshold_metrics(
        series, h_alarm, hh_alarm, l_alarm, ll_alarm
    )

    narrative_summary = _build_narrative_summary(
        segments=segments,
        overall_direction=overall_direction,
        dominant_pattern=dominant_pattern,
        volatility_regime=volatility_regime,
    )

    changepoint_time_ms = _dedupe_time_list([str(x.get("time_ms") or "") for x in regime_changepoints], 12)
    outlier_time_ms = _dedupe_time_list([str(x.get("time_ms") or "") for x in outliers], 12)
    transient_event_time_ms = _dedupe_time_list([str(x.get("time_ms") or "") for x in transient_events], 12)

    return TrendFeatureDetail(
        current=_round_float(current, 6),
        mean=_round_float(mean_v, 6),
        median=_round_float(median_v, 6),
        p95=_round_float(p95_v, 6),
        min=_round_float(min_v, 6),
        max=_round_float(max_v, 6),
        std=_round_float(std_v, 6),
        coefficient_of_variation=_round_float(cov, 6),
        range=_round_float(range_v, 6),
        mad=_round_float(mad_v, 6),
        slope=_round_float(slope, 9),
        relative_slope=_round_float(relative_slope, 9),
        short_slope=_round_float(short_slope, 9),
        medium_slope=_round_float(medium_slope, 9),
        long_slope=_round_float(long_slope, 9),
        trend_class=trend_class,
        change_rate_grade=change_rate_grade,
        changepoints=regime_changepoints,
        changepoint_severity=cp_severity,
        step_change_magnitude=step_mag,
        step_change_relative=step_rel,
        alarm_status=alarm_status,
        over_threshold_time=over_threshold_time,
        max_over_threshold_duration=max_over_threshold_duration,
        over_threshold_ratio=over_threshold_ratio,
        outliers=outliers,
        outlier_clusters=outlier_clusters,
        spike_detected=spike_detected,
        recovery_after_spike=recovery_after_spike,
        transient_events=transient_events,
        segment_stats=segments,
        dominant_pattern=dominant_pattern,
        level_regime=level_regime,
        volatility_regime=volatility_regime,
        overall_direction=overall_direction,
        narrative_summary=narrative_summary,
        changepoint_time_ms=changepoint_time_ms,
        outlier_time_ms=outlier_time_ms,
        transient_event_time_ms=transient_event_time_ms,
    )


def _build_summary_for_feature(feature: str, detail: TrendFeatureDetail) -> list[str]:
    if detail.current is None:
        return [f"{feature} 无有效数据"]

    lines: list[str] = []

    if detail.narrative_summary:
        lines.append(f"{feature} {detail.narrative_summary}")
    else:
        lines.append(
            f"{feature} 当前值 {detail.current}，均值 {detail.mean}，整体趋势 {detail.trend_class}"
        )

    lines.append(
        f"{feature} 当前值 {detail.current}，均值 {detail.mean}，P95 {detail.p95}，主导模式 {detail.dominant_pattern or 'unknown'}"
    )

    if detail.alarm_status != "normal":
        lines.append(f"{feature} 当前告警状态为 {detail.alarm_status}")
    elif detail.changepoint_severity != "none":
        lines.append(f"{feature} 检测到趋势状态切换，严重度 {detail.changepoint_severity}")
    elif detail.spike_detected:
        lines.append(f"{feature} 检测到瞬态突跳")
    else:
        lines.append(
            f"{feature} 整体无明显越限，变化速率 {detail.change_rate_grade}，波动状态 {detail.volatility_regime or 'unknown'}"
        )

    return lines[:3]


def _build_notable_points_for_feature(feature: str, series: list[tuple[str, float]], detail: TrendFeatureDetail) -> list[str]:
    if not series:
        return []

    values = [v for _, v in series]
    min_idx = values.index(min(values))
    max_idx = values.index(max(values))

    notable = [
        f"{series[min_idx][0]} 出现区间最低值 {round(series[min_idx][1], 3)}",
        f"{series[max_idx][0]} 出现区间最高值 {round(series[max_idx][1], 3)}",
    ]

    for seg in detail.segment_stats[:4]:
        notable.append(
            f"{seg.start_time_ms} 至 {seg.end_time_ms} 为一段 {_segment_to_phrase(seg)} 区间"
        )

    for cp in detail.changepoints[:3]:
        notable.append(
            f"{cp['time_ms']} 检测到变点 {cp['type']}，幅度 {cp.get('magnitude')}"
        )

    for outlier in detail.outliers[:2]:
        notable.append(
            f"{outlier['time_ms']} 出现局部残差离群，数值 {outlier['value']}，z={outlier['z_score']}"
        )

    for event in detail.transient_events[:2]:
        notable.append(
            f"{event['time_ms']} 出现瞬态事件 {event['type']}，幅度 {event['magnitude']}"
        )

    deduped: list[str] = []
    seen: set[str] = set()
    for item in notable:
        if item not in seen:
            seen.add(item)
            deduped.append(item)

    return deduped[:10]


def _collect_primary_anomaly_time_ms(detail: TrendFeatureDetail) -> list[str]:
    """
    兼容旧版 test_trend.py。
    这里只汇总“更适合作为总览红线”的时间点：
    - 先取 changepoints
    - 再取 transient events
    - outliers 只取孤立的前几个，不再整串灌进去
    """
    merged: list[str] = []

    merged.extend(detail.changepoint_time_ms[:8])
    merged.extend(detail.transient_event_time_ms[:4])

    isolated_outliers: list[str] = []
    for cluster in detail.outlier_clusters[:3]:
        if int(cluster.get("count") or 0) <= 2:
            ts = str(cluster.get("start_time_ms") or "")
            if ts:
                isolated_outliers.append(ts)

    merged.extend(isolated_outliers)
    return _dedupe_time_list(merged, 12)


def _merge_feature_summaries(feature_summaries: dict[str, list[str]]) -> list[str]:
    merged: list[str] = []
    for _, lines in feature_summaries.items():
        merged.extend(lines[:3])
    return merged[:20]


def _merge_feature_notable_points(feature_notables: dict[str, list[str]]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for _, lines in feature_notables.items():
        for line in lines:
            if line not in seen:
                seen.add(line)
                merged.append(line)
    return merged[:24]


async def extract_trend_features_tool(
    component_features: dict[str, list[str]],
    start: str,
    end: str,
) -> dict[str, Any]:
    """
    提取趋势特征。
    支持多个测点，不同测点可配置不同 feature。

    输入格式：
    {
      "component_features": {
        "component_id_1": ["pp_value", "rms"],
        "component_id_2": ["value"]
      },
      "start": "...",
      "end": "..."
    }
    """
    trend_data_payload = await _get_trend_data_impl(component_features, start, end)

    component_ids = trend_data_payload.get("component_ids") or []
    start_time = str(trend_data_payload.get("start_time") or "")
    end_time = str(trend_data_payload.get("end_time") or "")
    component_features = trend_data_payload.get("component_features") or {}
    grouped_data = trend_data_payload.get("data") or {}

    point_results: list[TrendPointAnalysisResult] = []

    for component_id in component_ids:
        point_data = grouped_data.get(component_id) or []
        if not isinstance(point_data, list):
            point_data = []

        features = component_features.get(component_id) or []
        if not isinstance(features, list):
            features = []

        feature_stats: dict[str, TrendFeatureDetail] = {}
        feature_summaries: dict[str, list[str]] = {}
        feature_notables: dict[str, list[str]] = {}
        anomaly_ts_merged: list[str] = []
        seen_ts: set[str] = set()

        for feature in features:
            series = _extract_feature_series(point_data, feature)
            detail = _build_feature_detail(point_data, series)
            feature_stats[feature] = detail
            feature_summaries[feature] = _build_summary_for_feature(feature, detail)
            feature_notables[feature] = _build_notable_points_for_feature(feature, series, detail)

            for ts in _collect_primary_anomaly_time_ms(detail):
                if ts and ts not in seen_ts:
                    seen_ts.add(ts)
                    anomaly_ts_merged.append(ts)

        point_results.append(
            TrendPointAnalysisResult(
                component_id=component_id,
                features=features,
                feature_stats=feature_stats,
                anomaly_time_ms=anomaly_ts_merged[:12],
                summary=_merge_feature_summaries(feature_summaries),
                notable_points=_merge_feature_notable_points(feature_notables),
            )
        )

    result = TrendAnalysisResult(
        component_ids=component_ids,
        start_time=start_time,
        end_time=end_time,
        component_features=component_features,
        point_results=point_results,
    )
    return result.model_dump()


async def main() -> None:
    """
    用法:
    python extract_trend_features_tool.py '{"1801":["pp_value","rms"]}' '2026-04-15 00:00:00' '2026-04-16 00:00:00'
    """
    if len(sys.argv) < 4:
        raise SystemExit(
            "用法: python extract_trend_features_tool.py '<component_features_json>' <start> <end>"
        )

    component_features = json.loads(sys.argv[1])
    result = await extract_trend_features_tool(component_features=component_features, start=sys.argv[2], end=sys.argv[3])
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
