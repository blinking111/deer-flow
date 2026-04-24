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
from tools.get_waveform_data_tool import _get_waveform_data_impl


class WaveformFeatureDetail(BaseModel):
    # 时域基础特征
    current_mean: float | None = None
    rms: float | None = None
    std: float | None = None
    peak: float | None = None
    peak_to_peak: float | None = None
    crest_factor: float | None = None
    impulse_factor: float | None = None
    margin_factor: float | None = None
    skewness: float | None = None
    kurtosis: float | None = None
    zero_crossing_rate: float | None = None
    periodicity_score: float | None = None
    shock_detected: bool = False

    # 频域特征
    dominant_frequency_hz: float | None = None
    dominant_amplitude: float | None = None
    spectral_centroid_hz: float | None = None
    total_spectral_energy: float | None = None
    low_band_energy_ratio: float | None = None
    mid_band_energy_ratio: float | None = None
    high_band_energy_ratio: float | None = None
    broadband_energy_ratio: float | None = None
    top_peaks: list[dict[str, float]] = Field(default_factory=list)

    # 阶次特征
    running_frequency_hz: float | None = None
    amp_0_5x: float | None = None
    amp_1x: float | None = None
    amp_2x: float | None = None
    amp_3x: float | None = None
    amp_1x_ratio: float | None = None
    amp_2x_to_1x_ratio: float | None = None
    harmonic_count: int = 0
    harmonic_detected: bool = False

    # 冲击 / 摩擦 / 削波 / 毛刺 / 畸变特征
    top_clipping_ratio: float | None = None
    bottom_clipping_ratio: float | None = None
    clipping_ratio: float | None = None
    asymmetric_clipping_index: float | None = None
    clipping_detected: bool = False

    impact_count: int = 0
    impact_count_per_sec: float | None = None
    impact_amplitude_p95: float | None = None
    impact_amplitude_p99: float | None = None
    impact_periodicity: float | None = None
    impact_interval_cv: float | None = None
    impact_interval_stability: float | None = None
    dominant_impact_interval_sec: float | None = None

    spike_density: float | None = None
    sharpness_score: float | None = None
    glitch_ratio: float | None = None
    local_spike_density: float | None = None
    roughness_score: float | None = None

    abnormal_phase_stability: float | None = None
    dominant_abnormal_phase: float | None = None

    event_count: int = 0
    glitch_count: int = 0


class SpectralWaveformAnalysisResult(BaseModel):
    component_id: str = Field(description="测点 ID，应为 type_num/type_enum=83 的轴振测点")
    time_ms: str = Field(description="查询时间点，毫秒时间戳")
    summary: list[str] = Field(description="波形和频谱的整体概括")
    spectral_findings: list[str] = Field(description="频谱特征")
    waveform_findings: list[str] = Field(description="时域波形特征")
    suspected_faults: list[str] = Field(description="可能的故障类型或机理")
    feature_details: WaveformFeatureDetail = Field(description="提取出的波形/频谱/阶次/冲击/削波结构化特征")


# =========================
# 基础统计函数
# =========================

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


def _safe_abs_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.fmean(abs(v) for v in values))


def _safe_rms(values: list[float]) -> float | None:
    if not values:
        return None
    return math.sqrt(sum(v * v for v in values) / len(values))


def _nth_central_moment(values: list[float], n: int) -> float | None:
    if not values:
        return None
    mean_v = _safe_mean(values)
    if mean_v is None:
        return None
    return sum((v - mean_v) ** n for v in values) / len(values)


def _skewness(values: list[float]) -> float | None:
    std = _safe_std(values)
    if len(values) < 3 or std <= 1e-12:
        return None
    m3 = _nth_central_moment(values, 3)
    if m3 is None:
        return None
    return m3 / (std ** 3)


def _kurtosis(values: list[float]) -> float | None:
    std = _safe_std(values)
    if len(values) < 4 or std <= 1e-12:
        return None
    m4 = _nth_central_moment(values, 4)
    if m4 is None:
        return None
    return m4 / (std ** 4)


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


def _zero_crossing_rate(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    count = 0
    for i in range(1, len(values)):
        if (values[i - 1] <= 0 < values[i]) or (values[i - 1] >= 0 > values[i]):
            count += 1
    return count / (len(values) - 1)


def _periodicity_score(values: list[float]) -> float | None:
    if len(values) < 8:
        return None
    mean_v = _safe_mean(values) or 0.0
    centered = [v - mean_v for v in values]
    denom = sum(v * v for v in centered)
    if denom <= 1e-12:
        return None

    max_corr = 0.0
    max_lag = min(len(values) // 2, 200)
    for lag in range(1, max_lag):
        num = sum(centered[i] * centered[i - lag] for i in range(lag, len(centered)))
        corr = num / denom
        if corr > max_corr:
            max_corr = corr
    return max_corr


def _shock_detected(values: list[float]) -> bool:
    if len(values) < 8:
        return False
    mean_v = _safe_mean(values)
    std_v = _safe_std(values)
    if mean_v is None or std_v <= 1e-12:
        return False
    return any(abs(v - mean_v) >= 4.0 * std_v for v in values)


# =========================
# 频域函数
# =========================

def _find_top_peaks(spec_x: list[float], spec_y: list[float], top_n: int = 5) -> list[dict[str, float]]:
    if len(spec_x) != len(spec_y) or len(spec_x) < 3:
        return []

    peaks: list[dict[str, float]] = []
    for i in range(1, len(spec_y) - 1):
        if spec_y[i] >= spec_y[i - 1] and spec_y[i] >= spec_y[i + 1]:
            peaks.append(
                {
                    "frequency_hz": float(spec_x[i]),
                    "amplitude": float(spec_y[i]),
                }
            )

    peaks.sort(key=lambda x: x["amplitude"], reverse=True)
    return peaks[:top_n]


def _band_energy_ratio(spec_x: list[float], spec_y: list[float], low: float, high: float) -> float | None:
    if len(spec_x) != len(spec_y) or not spec_x:
        return None
    total = sum(max(v, 0.0) for v in spec_y)
    if total <= 1e-12:
        return 0.0
    band = sum(max(y, 0.0) for x, y in zip(spec_x, spec_y) if low <= x < high)
    return band / total


def _spectral_centroid(spec_x: list[float], spec_y: list[float]) -> float | None:
    if len(spec_x) != len(spec_y) or not spec_x:
        return None
    denom = sum(max(v, 0.0) for v in spec_y)
    if denom <= 1e-12:
        return None
    num = sum(x * max(y, 0.0) for x, y in zip(spec_x, spec_y))
    return num / denom


def _find_nearest_amp(spec_x: list[float], spec_y: list[float], target_freq: float, tolerance_ratio: float = 0.03) -> float | None:
    if len(spec_x) != len(spec_y) or not spec_x or target_freq <= 0:
        return None

    tolerance = max(target_freq * tolerance_ratio, 0.5)
    candidates = [(abs(x - target_freq), y) for x, y in zip(spec_x, spec_y) if abs(x - target_freq) <= tolerance]
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return float(candidates[0][1])


def _harmonic_count(spec_x: list[float], spec_y: list[float], running_freq_hz: float | None) -> int:
    if running_freq_hz is None or running_freq_hz <= 0:
        return 0

    one_x = _find_nearest_amp(spec_x, spec_y, running_freq_hz)
    if one_x is None or one_x <= 1e-12:
        return 0

    count = 0
    for n in [2, 3, 4, 5, 6]:
        amp = _find_nearest_amp(spec_x, spec_y, running_freq_hz * n)
        if amp is not None and amp >= one_x * 0.08:
            count += 1
    return count


# =========================
# 波形预处理与事件检测
# =========================

def _moving_average(values: list[float], window: int) -> list[float]:
    if not values:
        return []
    if window <= 1:
        return [float(v) for v in values]

    half = window // 2
    result: list[float] = []
    for i in range(len(values)):
        left = max(0, i - half)
        right = min(len(values), i + half + 1)
        result.append(float(statistics.fmean(values[left:right])))
    return result


def _diff(values: list[float]) -> list[float]:
    if len(values) < 2:
        return []
    return [values[i] - values[i - 1] for i in range(1, len(values))]


def _robust_scale(values: list[float]) -> float:
    mad_v = _mad(values)
    if mad_v is not None and mad_v > 1e-12:
        return 1.4826 * mad_v
    std_v = _safe_std(values)
    if std_v > 1e-12:
        return std_v
    rms_v = _safe_rms(values)
    if rms_v not in (None, 0.0):
        return float(rms_v)
    return 1.0


def _local_peaks(values: list[float], threshold: float, min_distance: int = 1) -> list[int]:
    if len(values) < 3:
        return []

    peaks: list[int] = []
    last_idx = -10**9
    for i in range(1, len(values) - 1):
        if values[i] < threshold:
            continue
        if values[i] >= values[i - 1] and values[i] >= values[i + 1]:
            if i - last_idx < min_distance:
                if peaks and values[i] > values[peaks[-1]]:
                    peaks[-1] = i
                    last_idx = i
                continue
            peaks.append(i)
            last_idx = i
    return peaks


def _event_width(abs_signal: list[float], peak_idx: int, threshold: float) -> int:
    left = peak_idx
    while left > 0 and abs_signal[left - 1] >= threshold:
        left -= 1
    right = peak_idx
    while right < len(abs_signal) - 1 and abs_signal[right + 1] >= threshold:
        right += 1
    return right - left + 1


def _extract_event_amplitudes(signal: list[float], event_indices: list[int], half_window: int) -> list[float]:
    if not signal or not event_indices:
        return []
    abs_signal = [abs(v) for v in signal]
    result: list[float] = []
    for idx in event_indices:
        left = max(0, idx - half_window)
        right = min(len(signal), idx + half_window + 1)
        result.append(max(abs_signal[left:right], default=0.0))
    return result


def _circular_stability(phases: list[float]) -> tuple[float | None, float | None]:
    if len(phases) < 2:
        return None, None
    angles = [2.0 * math.pi * p for p in phases]
    c = sum(math.cos(a) for a in angles) / len(angles)
    s = sum(math.sin(a) for a in angles) / len(angles)
    r = math.sqrt(c * c + s * s)
    if r <= 1e-12:
        return 0.0, None
    mean_angle = math.atan2(s, c)
    if mean_angle < 0:
        mean_angle += 2.0 * math.pi
    dominant_phase = mean_angle / (2.0 * math.pi)
    return r, dominant_phase


def _phase_stability_from_events(event_indices: list[int], samples_per_rev: float | None) -> tuple[float | None, float | None]:
    if samples_per_rev is None or samples_per_rev <= 2 or not event_indices:
        return None, None
    phases: list[float] = []
    for idx in event_indices:
        phase = (idx % samples_per_rev) / samples_per_rev
        phases.append(float(phase))
    return _circular_stability(phases)


def _build_time_domain_enhanced_features(
    wave_y: list[float],
    sample_rate: float | None,
    running_frequency_hz: float | None,
) -> dict[str, float | int | bool | None]:
    if len(wave_y) < 8:
        return {
            "top_clipping_ratio": None,
            "bottom_clipping_ratio": None,
            "clipping_ratio": None,
            "asymmetric_clipping_index": None,
            "clipping_detected": False,
            "impact_count": 0,
            "impact_count_per_sec": None,
            "impact_amplitude_p95": None,
            "impact_amplitude_p99": None,
            "impact_periodicity": None,
            "impact_interval_cv": None,
            "impact_interval_stability": None,
            "dominant_impact_interval_sec": None,
            "spike_density": None,
            "sharpness_score": None,
            "glitch_ratio": None,
            "local_spike_density": None,
            "roughness_score": None,
            "abnormal_phase_stability": None,
            "dominant_abnormal_phase": None,
            "event_count": 0,
            "glitch_count": 0,
        }

    mean_v = _safe_mean(wave_y) or 0.0
    centered = [v - mean_v for v in wave_y]
    abs_centered = [abs(v) for v in centered]

    robust_scale = _robust_scale(centered)
    smooth_window = max(5, min(31, len(centered) // 25 if len(centered) >= 25 else 5))
    if smooth_window % 2 == 0:
        smooth_window += 1
    smooth = _moving_average(centered, smooth_window)
    residual = [x - s for x, s in zip(centered, smooth)]
    abs_residual = [abs(v) for v in residual]

    # 削波检测：极值平台 + 小斜率
    p95 = _percentile(centered, 0.95)
    p99 = _percentile(centered, 0.99)
    p05 = _percentile(centered, 0.05)
    p01 = _percentile(centered, 0.01)
    dx = _diff(centered)
    abs_dx = [abs(v) for v in dx]
    dx_ref = max(_percentile(abs_dx, 0.3) or 0.0, _robust_scale(dx) * 0.5, 1e-12)

    top_threshold = None
    bottom_threshold = None
    top_clipping_ratio = None
    bottom_clipping_ratio = None
    clipping_ratio = None
    asym_clip = None

    top_mask_count = 0
    bottom_mask_count = 0
    if p95 is not None and p99 is not None and p99 > p95:
        top_threshold = p99 - 0.3 * (p99 - p95)
    elif p99 is not None:
        top_threshold = p99

    if p05 is not None and p01 is not None and p05 > p01:
        bottom_threshold = p01 + 0.3 * (p05 - p01)
    elif p01 is not None:
        bottom_threshold = p01

    if top_threshold is not None or bottom_threshold is not None:
        for i, value in enumerate(centered):
            left_slope = abs(centered[i] - centered[i - 1]) if i > 0 else 0.0
            right_slope = abs(centered[i + 1] - centered[i]) if i < len(centered) - 1 else 0.0
            slope_small = max(left_slope, right_slope) <= dx_ref

            if top_threshold is not None and value >= top_threshold and slope_small:
                top_mask_count += 1
            if bottom_threshold is not None and value <= bottom_threshold and slope_small:
                bottom_mask_count += 1

        top_clipping_ratio = top_mask_count / len(centered)
        bottom_clipping_ratio = bottom_mask_count / len(centered)
        clipping_ratio = top_clipping_ratio + bottom_clipping_ratio
        denom = clipping_ratio if clipping_ratio > 1e-12 else None
        if denom is not None:
            asym_clip = (top_clipping_ratio - bottom_clipping_ratio) / denom

    clipping_detected = bool(clipping_ratio is not None and clipping_ratio >= 0.01)

    # 冲击 / 毛刺 / 尖峰检测
    impact_threshold = max((_median(abs_residual) or 0.0) + 6.0 * (_mad(abs_residual) or 0.0), _percentile(abs_residual, 0.995) or 0.0)
    spike_threshold = max((_median(abs_residual) or 0.0) + 3.5 * (_mad(abs_residual) or 0.0), _percentile(abs_residual, 0.98) or 0.0)
    glitch_threshold = max((_median(abs_residual) or 0.0) + 2.8 * (_mad(abs_residual) or 0.0), _percentile(abs_residual, 0.95) or 0.0)

    min_distance = 1
    if sample_rate and running_frequency_hz and running_frequency_hz > 0:
        samples_per_rev = sample_rate / running_frequency_hz
        min_distance = max(1, int(samples_per_rev * 0.03))
    else:
        samples_per_rev = None
        min_distance = max(1, len(centered) // 200)

    impact_indices = _local_peaks(abs_residual, impact_threshold, min_distance=min_distance)
    spike_indices = _local_peaks(abs_residual, spike_threshold, min_distance=max(1, min_distance // 2))
    glitch_indices = _local_peaks(abs_residual, glitch_threshold, min_distance=1)

    event_half_window = max(1, min(8, len(centered) // 100))
    impact_amplitudes = _extract_event_amplitudes(residual, impact_indices, event_half_window)

    impact_count = len(impact_indices)
    duration_sec = (len(centered) / sample_rate) if sample_rate and sample_rate > 1e-12 else None
    impact_count_per_sec = (impact_count / duration_sec) if duration_sec not in (None, 0.0) else None
    spike_density = (len(spike_indices) / duration_sec) if duration_sec not in (None, 0.0) else None
    local_spike_density = spike_density

    impact_p95 = _percentile(impact_amplitudes, 0.95)
    impact_p99 = _percentile(impact_amplitudes, 0.99)

    impact_periodicity = None
    impact_interval_cv = None
    impact_interval_stability = None
    dominant_interval_sec = None
    if len(impact_indices) >= 3 and sample_rate and sample_rate > 1e-12:
        intervals = [(impact_indices[i] - impact_indices[i - 1]) / sample_rate for i in range(1, len(impact_indices))]
        mean_interval = _safe_mean(intervals)
        std_interval = _safe_std(intervals)
        if mean_interval not in (None, 0.0):
            impact_interval_cv = std_interval / mean_interval
            impact_interval_stability = 1.0 / (1.0 + impact_interval_cv)
            impact_periodicity = max(0.0, min(1.0, 1.0 - impact_interval_cv))
            dominant_interval_sec = _median(intervals)

    # 毛刺占比：窄脉冲占比
    glitch_count = 0
    glitch_points = 0
    narrow_width_limit = max(1, min(3, int((sample_rate / 2000.0)) if sample_rate else 2))
    low_glitch_threshold = max(glitch_threshold * 0.6, 1e-12)
    for idx in glitch_indices:
        width = _event_width(abs_residual, idx, low_glitch_threshold)
        if width <= narrow_width_limit:
            glitch_count += 1
            glitch_points += width
    glitch_ratio = (glitch_points / len(centered)) if centered else None

    # 尖锐度 / 粗糙度
    rms_v = _safe_rms(centered)
    d1 = _diff(centered)
    d2 = _diff(d1)
    p95_d1 = _percentile([abs(v) for v in d1], 0.95) if d1 else None
    p95_d2 = _percentile([abs(v) for v in d2], 0.95) if d2 else None
    sharpness_score = None
    if rms_v not in (None, 0.0):
        parts: list[float] = []
        if p95_d1 is not None:
            parts.append(p95_d1 / rms_v)
        if p95_d2 is not None:
            parts.append((p95_d2 / rms_v) * 0.5)
        if parts:
            sharpness_score = sum(parts) / len(parts)

    roughness_score = None
    if robust_scale > 1e-12:
        roughness_score = (_safe_rms(residual) or 0.0) / robust_scale

    abnormal_phase_stability, dominant_phase = _phase_stability_from_events(impact_indices or spike_indices, samples_per_rev)

    return {
        "top_clipping_ratio": top_clipping_ratio,
        "bottom_clipping_ratio": bottom_clipping_ratio,
        "clipping_ratio": clipping_ratio,
        "asymmetric_clipping_index": asym_clip,
        "clipping_detected": clipping_detected,
        "impact_count": impact_count,
        "impact_count_per_sec": impact_count_per_sec,
        "impact_amplitude_p95": impact_p95,
        "impact_amplitude_p99": impact_p99,
        "impact_periodicity": impact_periodicity,
        "impact_interval_cv": impact_interval_cv,
        "impact_interval_stability": impact_interval_stability,
        "dominant_impact_interval_sec": dominant_interval_sec,
        "spike_density": spike_density,
        "sharpness_score": sharpness_score,
        "glitch_ratio": glitch_ratio,
        "local_spike_density": local_spike_density,
        "roughness_score": roughness_score,
        "abnormal_phase_stability": abnormal_phase_stability,
        "dominant_abnormal_phase": dominant_phase,
        "event_count": len(spike_indices),
        "glitch_count": glitch_count,
    }


# =========================
# 主特征提取
# =========================

def _extract_feature_detail(data: dict[str, Any]) -> WaveformFeatureDetail:
    wave_y = [float(v) for v in (data.get("wave_y") or []) if isinstance(v, (int, float)) and math.isfinite(v)]
    spec_x = [float(v) for v in (data.get("spec_x") or []) if isinstance(v, (int, float)) and math.isfinite(v)]
    spec_y = [float(v) for v in (data.get("spec_y") or []) if isinstance(v, (int, float)) and math.isfinite(v)]

    n_spec = min(len(spec_x), len(spec_y))
    spec_x = spec_x[:n_spec]
    spec_y = spec_y[:n_spec]

    mean_v = _safe_mean(wave_y)
    rms_v = _safe_rms(wave_y)
    std_v = _safe_std(wave_y)
    peak_v = max((abs(v) for v in wave_y), default=None)
    p2p_v = (max(wave_y) - min(wave_y)) if wave_y else None
    abs_mean_v = _safe_abs_mean(wave_y)
    sqrt_abs_mean = _safe_mean([math.sqrt(abs(v)) for v in wave_y]) if wave_y else None

    crest_factor = (peak_v / rms_v) if peak_v is not None and rms_v not in (None, 0.0) else None
    impulse_factor = (peak_v / abs_mean_v) if peak_v is not None and abs_mean_v not in (None, 0.0) else None
    margin_factor = None
    if peak_v is not None and sqrt_abs_mean not in (None, 0.0):
        margin_factor = peak_v / (sqrt_abs_mean ** 2)

    skewness_v = _skewness(wave_y)
    kurtosis_v = _kurtosis(wave_y)
    zcr_v = _zero_crossing_rate(wave_y)
    periodicity_v = _periodicity_score(wave_y)
    shock_flag = _shock_detected(wave_y)

    dominant_frequency = None
    dominant_amplitude = None
    if spec_y:
        idx = max(range(len(spec_y)), key=lambda i: spec_y[i])
        dominant_frequency = spec_x[idx]
        dominant_amplitude = spec_y[idx]

    centroid_v = _spectral_centroid(spec_x, spec_y)
    total_energy = sum(max(v, 0.0) for v in spec_y) if spec_y else None

    low_band_ratio = _band_energy_ratio(spec_x, spec_y, 0.0, 200.0)
    mid_band_ratio = _band_energy_ratio(spec_x, spec_y, 200.0, 1000.0)
    high_band_ratio = _band_energy_ratio(spec_x, spec_y, 1000.0, float("inf"))
    broadband_ratio = high_band_ratio

    speed = data.get("speed")
    running_frequency_hz = None
    if isinstance(speed, (int, float)) and math.isfinite(speed) and speed > 0:
        running_frequency_hz = float(speed) / 60.0

    amp_0_5x = _find_nearest_amp(spec_x, spec_y, running_frequency_hz * 0.5) if running_frequency_hz else None
    amp_1x = _find_nearest_amp(spec_x, spec_y, running_frequency_hz) if running_frequency_hz else None
    amp_2x = _find_nearest_amp(spec_x, spec_y, running_frequency_hz * 2.0) if running_frequency_hz else None
    amp_3x = _find_nearest_amp(spec_x, spec_y, running_frequency_hz * 3.0) if running_frequency_hz else None

    amp_1x_ratio = None
    if amp_1x is not None and total_energy not in (None, 0.0):
        amp_1x_ratio = amp_1x / total_energy

    amp_2x_to_1x_ratio = None
    if amp_1x not in (None, 0.0) and amp_2x is not None:
        amp_2x_to_1x_ratio = amp_2x / amp_1x

    harmonic_count = _harmonic_count(spec_x, spec_y, running_frequency_hz)
    harmonic_detected = harmonic_count >= 2

    sample_rate = data.get("sample_rate")
    sample_rate_float = float(sample_rate) if isinstance(sample_rate, (int, float)) and math.isfinite(sample_rate) and sample_rate > 0 else None
    enhanced = _build_time_domain_enhanced_features(wave_y, sample_rate_float, running_frequency_hz)

    shock_flag = shock_flag or bool(enhanced.get("impact_count", 0) >= 1) or bool((enhanced.get("impact_amplitude_p99") or 0.0) > ((_safe_rms(wave_y) or 0.0) * 3.5))

    return WaveformFeatureDetail(
        current_mean=_round_float(mean_v, 6),
        rms=_round_float(rms_v, 6),
        std=_round_float(std_v, 6),
        peak=_round_float(peak_v, 6),
        peak_to_peak=_round_float(p2p_v, 6),
        crest_factor=_round_float(crest_factor, 6),
        impulse_factor=_round_float(impulse_factor, 6),
        margin_factor=_round_float(margin_factor, 6),
        skewness=_round_float(skewness_v, 6),
        kurtosis=_round_float(kurtosis_v, 6),
        zero_crossing_rate=_round_float(zcr_v, 6),
        periodicity_score=_round_float(periodicity_v, 6),
        shock_detected=shock_flag,
        dominant_frequency_hz=_round_float(dominant_frequency, 6),
        dominant_amplitude=_round_float(dominant_amplitude, 6),
        spectral_centroid_hz=_round_float(centroid_v, 6),
        total_spectral_energy=_round_float(total_energy, 6),
        low_band_energy_ratio=_round_float(low_band_ratio, 6),
        mid_band_energy_ratio=_round_float(mid_band_ratio, 6),
        high_band_energy_ratio=_round_float(high_band_ratio, 6),
        broadband_energy_ratio=_round_float(broadband_ratio, 6),
        top_peaks=_find_top_peaks(spec_x, spec_y, top_n=5),
        running_frequency_hz=_round_float(running_frequency_hz, 6),
        amp_0_5x=_round_float(amp_0_5x, 6),
        amp_1x=_round_float(amp_1x, 6),
        amp_2x=_round_float(amp_2x, 6),
        amp_3x=_round_float(amp_3x, 6),
        amp_1x_ratio=_round_float(amp_1x_ratio, 6),
        amp_2x_to_1x_ratio=_round_float(amp_2x_to_1x_ratio, 6),
        harmonic_count=harmonic_count,
        harmonic_detected=harmonic_detected,
        top_clipping_ratio=_round_float(enhanced.get("top_clipping_ratio"), 6),
        bottom_clipping_ratio=_round_float(enhanced.get("bottom_clipping_ratio"), 6),
        clipping_ratio=_round_float(enhanced.get("clipping_ratio"), 6),
        asymmetric_clipping_index=_round_float(enhanced.get("asymmetric_clipping_index"), 6),
        clipping_detected=bool(enhanced.get("clipping_detected", False)),
        impact_count=int(enhanced.get("impact_count") or 0),
        impact_count_per_sec=_round_float(enhanced.get("impact_count_per_sec"), 6),
        impact_amplitude_p95=_round_float(enhanced.get("impact_amplitude_p95"), 6),
        impact_amplitude_p99=_round_float(enhanced.get("impact_amplitude_p99"), 6),
        impact_periodicity=_round_float(enhanced.get("impact_periodicity"), 6),
        impact_interval_cv=_round_float(enhanced.get("impact_interval_cv"), 6),
        impact_interval_stability=_round_float(enhanced.get("impact_interval_stability"), 6),
        dominant_impact_interval_sec=_round_float(enhanced.get("dominant_impact_interval_sec"), 6),
        spike_density=_round_float(enhanced.get("spike_density"), 6),
        sharpness_score=_round_float(enhanced.get("sharpness_score"), 6),
        glitch_ratio=_round_float(enhanced.get("glitch_ratio"), 6),
        local_spike_density=_round_float(enhanced.get("local_spike_density"), 6),
        roughness_score=_round_float(enhanced.get("roughness_score"), 6),
        abnormal_phase_stability=_round_float(enhanced.get("abnormal_phase_stability"), 6),
        dominant_abnormal_phase=_round_float(enhanced.get("dominant_abnormal_phase"), 6),
        event_count=int(enhanced.get("event_count") or 0),
        glitch_count=int(enhanced.get("glitch_count") or 0),
    )


# =========================
# 文本化输出
# =========================

def _build_waveform_findings(detail: WaveformFeatureDetail) -> list[str]:
    findings: list[str] = []

    if detail.rms is not None and detail.peak_to_peak is not None:
        findings.append(f"时域波形 RMS={detail.rms}，峰峰值={detail.peak_to_peak}")

    if detail.periodicity_score is not None:
        if detail.periodicity_score >= 0.6:
            findings.append("时域波形周期性较强")
        elif detail.periodicity_score >= 0.3:
            findings.append("时域波形存在一定周期性")
        else:
            findings.append("时域波形周期性不突出")

    if detail.shock_detected:
        findings.append("检测到明显冲击/瞬态尖峰特征")
    else:
        findings.append("未检测到明显冲击尖峰")

    if detail.impact_count_per_sec is not None and detail.impact_count_per_sec > 0:
        findings.append(f"单位时间冲击次数约 {detail.impact_count_per_sec}/s")

    if detail.impact_periodicity is not None:
        if detail.impact_periodicity >= 0.7:
            findings.append("冲击重复性较强，存在稳定重复接触迹象")
        elif detail.impact_periodicity >= 0.4:
            findings.append("冲击存在一定重复性")

    if detail.clipping_detected and detail.clipping_ratio is not None:
        findings.append(f"检测到削波/平台化迹象，削波占比约 {detail.clipping_ratio}")

    if detail.asymmetric_clipping_index is not None and abs(detail.asymmetric_clipping_index) >= 0.25:
        direction = "正向" if detail.asymmetric_clipping_index > 0 else "负向"
        findings.append(f"削波存在{direction}偏置，表现出一定非对称性")

    if detail.glitch_ratio is not None and detail.glitch_ratio >= 0.005:
        findings.append(f"波形中存在毛刺型窄脉冲，毛刺占比约 {detail.glitch_ratio}")

    if detail.sharpness_score is not None and detail.sharpness_score >= 1.5:
        findings.append(f"波形尖锐度较高，sharpness={detail.sharpness_score}")

    if detail.abnormal_phase_stability is not None and detail.abnormal_phase_stability >= 0.7:
        findings.append("异常点在周期内位置较稳定，接近固定相位重复出现")

    if detail.kurtosis is not None:
        findings.append(f"峭度={detail.kurtosis}")

    return findings[:10]



def _build_spectral_findings(detail: WaveformFeatureDetail) -> list[str]:
    findings: list[str] = []

    if detail.dominant_frequency_hz is not None and detail.dominant_amplitude is not None:
        findings.append(f"主峰位于 {detail.dominant_frequency_hz} Hz，幅值 {detail.dominant_amplitude}")

    if detail.running_frequency_hz is not None and detail.amp_1x is not None:
        findings.append(f"1X≈{detail.running_frequency_hz} Hz，幅值 {detail.amp_1x}")

    if detail.amp_2x is not None:
        findings.append(f"2X 幅值 {detail.amp_2x}")
    if detail.amp_3x is not None:
        findings.append(f"3X 幅值 {detail.amp_3x}")

    if detail.harmonic_detected:
        findings.append(f"检测到明显倍频族，倍频数约 {detail.harmonic_count}")
    else:
        findings.append("倍频族不明显")

    if detail.high_band_energy_ratio is not None:
        findings.append(f"高频能量占比 {detail.high_band_energy_ratio}")

    if detail.spectral_centroid_hz is not None:
        findings.append(f"频谱重心约 {detail.spectral_centroid_hz} Hz")

    return findings[:8]



def _build_summary(detail: WaveformFeatureDetail) -> list[str]:
    summary: list[str] = []

    if detail.amp_1x_ratio is not None:
        if detail.amp_1x_ratio >= 0.3:
            summary.append("振动能量对 1X 同步成分依赖明显")
        elif detail.amp_1x_ratio >= 0.1:
            summary.append("1X 成分较明显，但不是唯一主导")
        else:
            summary.append("1X 成分不占绝对主导")

    if detail.harmonic_detected:
        summary.append("频谱中存在一定倍频谐波结构")

    if detail.shock_detected:
        summary.append("时域存在冲击性特征")
    elif detail.high_band_energy_ratio is not None and detail.high_band_energy_ratio < 0.15:
        summary.append("高频宽带能量不突出")

    if detail.periodicity_score is not None and detail.periodicity_score >= 0.5:
        summary.append("波形呈较稳定的周期振动")

    if detail.clipping_detected:
        summary.append("波形存在削波/非线性平台化迹象")

    if detail.glitch_ratio is not None and detail.glitch_ratio >= 0.005:
        summary.append("波形包含一定毛刺/窄脉冲成分")

    if detail.abnormal_phase_stability is not None and detail.abnormal_phase_stability >= 0.7:
        summary.append("异常冲击在周期内位置较稳定")

    return summary[:8]



def _build_suspected_faults(detail: WaveformFeatureDetail) -> list[str]:
    suspects: list[str] = []

    if detail.amp_1x_ratio is not None and detail.amp_1x_ratio >= 0.25:
        suspects.append("疑似同步类振动问题（如不平衡方向）")

    if detail.amp_2x_to_1x_ratio is not None and detail.amp_2x_to_1x_ratio >= 0.3:
        suspects.append("疑似不对中或松动方向特征")

    if detail.harmonic_detected and detail.harmonic_count >= 3:
        suspects.append("存在倍频谐波，需排查松动/非线性接触")

    if detail.shock_detected or (detail.kurtosis is not None and detail.kurtosis >= 4.5):
        suspects.append("存在冲击特征，需关注碰摩/局部损伤/间歇异常")

    if detail.high_band_energy_ratio is not None and detail.high_band_energy_ratio >= 0.25:
        suspects.append("高频宽带成分较多，需关注摩擦或早期损伤类问题")

    if (
        detail.impact_periodicity is not None and detail.impact_periodicity >= 0.65
        and detail.abnormal_phase_stability is not None and detail.abnormal_phase_stability >= 0.65
    ):
        suspects.append("冲击在固定相位附近重复出现，需重点关注碰磨/密封摩擦")

    if detail.clipping_detected and detail.asymmetric_clipping_index is not None and abs(detail.asymmetric_clipping_index) >= 0.25:
        suspects.append("存在非对称削波，需排查单侧接触、偏置摩擦或测振区晃度")

    if detail.glitch_ratio is not None and detail.glitch_ratio >= 0.008 and detail.high_band_energy_ratio is not None and detail.high_band_energy_ratio >= 0.2:
        suspects.append("毛刺与高频宽带同时增多，需关注摩擦、擦碰或粗糙接触")

    if detail.sharpness_score is not None and detail.sharpness_score >= 1.8 and detail.spike_density is not None and detail.spike_density > 0:
        suspects.append("波形尖锐且局部尖峰密集，需关注冲击性接触或非线性摩擦")

    deduped: list[str] = []
    seen: set[str] = set()
    for item in suspects:
        if item not in seen:
            seen.add(item)
            deduped.append(item)

    return deduped[:8]


async def extract_spectral_waveform_features_tool(
    component_id: str,
    time: str | None = None,
    time_ms: str | None = None,
) -> dict[str, Any]:
    """
    提取波形/频谱/阶次/冲击/削波特征。

    输入格式：
    {
      "component_id": "type_num/type_enum=83 的轴振测点 ID",
      "time": "趋势分析返回的异常毫秒时间戳，或可解析时间字符串",
      "time_ms": "趋势分析返回的异常毫秒时间戳，可选，优先于 time"
    }
    """
    payload_time = str(time_ms or time or "")
    if not payload_time:
        raise ValueError("time or time_ms is required")
    waveform_payload = await _get_waveform_data_impl(component_id, payload_time)

    component_id = str(waveform_payload.get("component_id") or "")
    time_ms = str(waveform_payload.get("time_ms") or "")
    data = waveform_payload.get("data") or {}

    if not isinstance(data, dict):
        data = {}

    feature_details = _extract_feature_detail(data)
    result = SpectralWaveformAnalysisResult(
        component_id=component_id,
        time_ms=time_ms,
        summary=_build_summary(feature_details),
        spectral_findings=_build_spectral_findings(feature_details),
        waveform_findings=_build_waveform_findings(feature_details),
        suspected_faults=_build_suspected_faults(feature_details),
        feature_details=feature_details,
    )
    return result.model_dump()


async def main() -> None:
    """
    用法:
    python extract_spectral_waveform_features_tool.py <component_id> <time_or_time_ms>
    """
    if len(sys.argv) < 3:
        raise SystemExit(
            "用法: python extract_spectral_waveform_features_tool.py <component_id> <time_or_time_ms>"
        )

    result = await extract_spectral_waveform_features_tool(component_id=sys.argv[1], time=sys.argv[2])
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
