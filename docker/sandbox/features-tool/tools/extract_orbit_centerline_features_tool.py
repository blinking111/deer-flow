import asyncio
import json
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pydantic import BaseModel, Field
from tools.get_orbit_data_tool import _get_orbit_data_impl


class OrbitCenterlineFeatureDetail(BaseModel):
    # 原始轨迹基础几何特征
    point_count: int = Field(default=0, description="原始轨迹点数")

    width: float | None = Field(default=None, description="原始轨迹包络宽度")
    height: float | None = Field(default=None, description="原始轨迹包络高度")
    orbit_area: float | None = Field(default=None, description="原始轨迹面积")

    # 中心/主轴特征
    center_offset_radius: float | None = Field(default=None, description="原始轨迹中心偏移半径")
    principal_angle_deg: float | None = Field(default=None, description="原始轨迹主轴方向角，单位度")

    # 复杂度 / 截断 / 自交
    path_complexity: float | None = Field(default=None, description="原始轨迹路径复杂度，建议为路径长度相对包络尺度的归一化值")
    flattening_score: float | None = Field(default=None, description="原始轨迹削平/截断程度得分")
    flattened_side: str | None = Field(default=None, description="原始轨迹主要削平方向，如 left/right/top/bottom/multi")

    self_intersection_count: int = Field(default=0, description="原始轨迹自交次数")
    convex_fill_ratio: float | None = Field(default=None, description="原始轨迹对凸包的填充率，越低表示凹陷、腰缩、截断或多环特征越明显")

    # 多周期重复性
    repetition_score: float | None = Field(default=None, description="原始轨迹多周期重复性总分")
    cycle_shape_similarity: float | None = Field(default=None, description="周期间形状相似度")

    # =========================
    # 1X 轨迹几何特征
    # =========================
    one_x_point_count: int = Field(default=0, description="1X 轨迹点数")

    one_x_width: float | None = Field(default=None, description="1X 轨迹包络宽度")
    one_x_height: float | None = Field(default=None, description="1X 轨迹包络高度")

    one_x_major_axis: float | None = Field(default=None, description="1X 轨迹主轴长度")
    one_x_minor_axis: float | None = Field(default=None, description="1X 轨迹副轴长度")
    one_x_axis_ratio: float | None = Field(default=None, description="1X 轨迹主副轴比")
    one_x_eccentricity_ratio: float | None = Field(default=None, description="1X 轨迹偏心程度")

    one_x_orbit_area: float | None = Field(default=None, description="1X 轨迹面积")
    one_x_closure_ratio: float | None = Field(default=None, description="1X 轨迹闭合比例")
    one_x_roundness_score: float | None = Field(default=None, description="1X 近圆程度得分")

    one_x_principal_angle_deg: float | None = Field(default=None, description="1X 轨迹主轴方向角，单位度")
    one_x_precession_direction: str | None = Field(default=None, description="1X 轨迹进动方向，如 正进动/反进动")
    raw_to_one_x_angle_diff_deg: float | None = Field(default=None, description="原始轨迹主轴与 1X 主轴夹角差，单位度")

    one_x_center_x: float | None = Field(default=None, description="1X 轨迹中心 X")
    one_x_center_y: float | None = Field(default=None, description="1X 轨迹中心 Y")
    one_x_center_offset_radius: float | None = Field(default=None, description="1X 轨迹中心偏移半径")

    # =========================
    # 2X 轨迹几何特征
    # =========================
    two_x_point_count: int = Field(default=0, description="2X 轨迹点数")

    two_x_width: float | None = Field(default=None, description="2X 轨迹包络宽度")
    two_x_height: float | None = Field(default=None, description="2X 轨迹包络高度")

    two_x_major_axis: float | None = Field(default=None, description="2X 轨迹主轴长度")
    two_x_minor_axis: float | None = Field(default=None, description="2X 轨迹副轴长度")
    two_x_axis_ratio: float | None = Field(default=None, description="2X 轨迹主副轴比")
    two_x_orbit_area: float | None = Field(default=None, description="2X 轨迹面积")

    two_x_principal_angle_deg: float | None = Field(default=None, description="2X 轨迹主轴方向角，单位度")
    two_x_to_one_x_ratio: float | None = Field(default=None, description="2X 相对 1X 的主尺度比例")
    one_x_dominant: bool = Field(default=False, description="是否 1X 占主导")
    two_x_significant: bool = Field(default=False, description="是否 2X 显著")

    # =========================
    # 2X 形态特征：8字 / 双叶 / 腰缩 / 香蕉弯曲 / 不对称
    # =========================
    two_x_self_intersection_count: int = Field(default=0, description="2X 轨迹自交次数")
    two_x_figure_eight_score: float | None = Field(default=None, description="2X 轨迹 8 字形得分")
    two_x_double_lobe_score: float | None = Field(default=None, description="2X 轨迹双叶程度得分")

    two_x_waist_width: float | None = Field(default=None, description="2X 轨迹腰部宽度")
    two_x_lobe_width: float | None = Field(default=None, description="2X 轨迹叶片典型宽度")
    two_x_waist_constriction_ratio: float | None = Field(default=None, description="2X 腰部收缩比，越小表示腰部越细")

    two_x_banana_bending_score: float | None = Field(default=None, description="2X 香蕉形弯曲得分")
    two_x_quadratic_bending: float | None = Field(default=None, description="2X 主轴坐标二次拟合弯曲系数")

    two_x_lobe_area_asymmetry: float | None = Field(default=None, description="2X 两叶面积不对称性")
    two_x_lobe_size_asymmetry: float | None = Field(default=None, description="2X 两叶尺寸不对称性")
    two_x_shape_asymmetry_score: float | None = Field(default=None, description="2X 整体形态不对称得分")

    # =========================
    # 标签化字段
    # =========================
    shape_tags: list[str] = Field(default_factory=list, description="原始轨迹形态标签")
    centerline_tags: list[str] = Field(default_factory=list, description="中心线/偏置标签")
    one_x_tags: list[str] = Field(default_factory=list, description="1X 轨迹标签")
    two_x_tags: list[str] = Field(default_factory=list, description="2X 轨迹标签")


class OrbitCenterlineAnalysisResult(BaseModel):
    machine_id: str = Field(description="机组 ID")
    bearing_id: str = Field(description="轴承 ID，应为 type_num/type_enum=70 的轴承")
    time_ms: str = Field(description="查询时间点，毫秒时间戳")

    summary: list[str] = Field(description="轴心轨迹整体概括")
    shape_findings: list[str] = Field(description="原始轨迹形态特征")
    centerline_findings: list[str] = Field(description="中心线/中心偏置/重复性特征")
    one_x_findings: list[str] = Field(description="1X 轨迹特征")
    two_x_findings: list[str] = Field(description="2X 轨迹特征")
    suspected_faults: list[str] = Field(description="可能的故障类型或机理")

    feature_details: OrbitCenterlineFeatureDetail = Field(description="提取出的轨迹/中心线/1X/2X 结构化特征")

    probe_ids: list[str] = Field(default_factory=list, description="实际参与轨迹计算的探头 ID")


def _round_float(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _safe_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean_v = _safe_mean(values)
    if mean_v is None:
        return 0.0
    return math.sqrt(sum((v - mean_v) ** 2 for v in values) / len(values))


def _extract_xy(points: list[Any]) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    for item in points:
        if (
            isinstance(item, (list, tuple))
            and len(item) >= 2
            and isinstance(item[0], (int, float))
            and isinstance(item[1], (int, float))
            and math.isfinite(item[0])
            and math.isfinite(item[1])
        ):
            result.append((float(item[0]), float(item[1])))
    return result


def _bbox_metrics(points: list[tuple[float, float]]) -> dict[str, float | None]:
    if not points:
        return {
            "x_min": None,
            "x_max": None,
            "y_min": None,
            "y_max": None,
            "width": None,
            "height": None,
        }

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return {
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "width": x_max - x_min,
        "height": y_max - y_min,
    }


def _center_metrics(points: list[tuple[float, float]]) -> dict[str, float | None]:
    if not points:
        return {
            "center_x": None,
            "center_y": None,
            "center_offset_radius": None,
        }

    cx = _safe_mean([p[0] for p in points])
    cy = _safe_mean([p[1] for p in points])
    if cx is None or cy is None:
        return {
            "center_x": None,
            "center_y": None,
            "center_offset_radius": None,
        }

    radius = math.sqrt(cx * cx + cy * cy)
    return {
        "center_x": cx,
        "center_y": cy,
        "center_offset_radius": radius,
    }


def _principal_axis_metrics(points: list[tuple[float, float]]) -> dict[str, float | None]:
    if len(points) < 3:
        return {
            "major_axis": None,
            "minor_axis": None,
            "axis_ratio": None,
            "eccentricity_ratio": None,
            "principal_angle_deg": None,
        }

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    cx = _safe_mean(xs)
    cy = _safe_mean(ys)
    if cx is None or cy is None:
        return {
            "major_axis": None,
            "minor_axis": None,
            "axis_ratio": None,
            "eccentricity_ratio": None,
            "principal_angle_deg": None,
        }

    dx = [x - cx for x in xs]
    dy = [y - cy for y in ys]

    sxx = sum(v * v for v in dx) / len(dx)
    syy = sum(v * v for v in dy) / len(dy)
    sxy = sum(a * b for a, b in zip(dx, dy)) / len(dx)

    trace = sxx + syy
    det_part = math.sqrt(max((sxx - syy) ** 2 + 4 * sxy * sxy, 0.0))
    eig1 = (trace + det_part) / 2
    eig2 = (trace - det_part) / 2

    major = 2 * math.sqrt(max(eig1, 0.0))
    minor = 2 * math.sqrt(max(eig2, 0.0))
    axis_ratio = (major / minor) if minor > 1e-12 else None
    eccentricity_ratio = (1 - (minor / major)) if major > 1e-12 and minor is not None else None

    angle = 0.5 * math.atan2(2 * sxy, sxx - syy)
    angle_deg = math.degrees(angle)

    return {
        "major_axis": major,
        "minor_axis": minor,
        "axis_ratio": axis_ratio,
        "eccentricity_ratio": eccentricity_ratio,
        "principal_angle_deg": angle_deg,
    }


def _polygon_area(points: list[tuple[float, float]]) -> float | None:
    if len(points) < 3:
        return None
    area = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def _path_length(points: list[tuple[float, float]]) -> float | None:
    if len(points) < 2:
        return None
    total = 0.0
    for i in range(1, len(points)):
        x1, y1 = points[i - 1]
        x2, y2 = points[i]
        total += math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return total


def _path_complexity(points: list[tuple[float, float]], width: float | None, height: float | None) -> float | None:
    length = _path_length(points)
    if length is None:
        return None
    scale = max(width or 0.0, height or 0.0, 1e-12)
    return length / scale


def _cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    pts = sorted(set(points))
    if len(pts) <= 1:
        return pts

    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def _convex_fill_ratio(points: list[tuple[float, float]], area: float | None) -> float | None:
    if len(points) < 3 or area is None:
        return None
    hull = _convex_hull(points)
    hull_area = _polygon_area(hull)
    if hull_area in (None, 0.0):
        return None
    return area / hull_area


def _flattening_metrics(points: list[tuple[float, float]], bbox: dict[str, float | None]) -> dict[str, float | str | None]:
    width = bbox["width"]
    height = bbox["height"]
    if not points or width in (None, 0.0) or height in (None, 0.0):
        return {
            "flattening_score": None,
            "flattened_side": None,
        }

    x_min = bbox["x_min"]
    x_max = bbox["x_max"]
    y_min = bbox["y_min"]
    y_max = bbox["y_max"]
    if None in {x_min, x_max, y_min, y_max}:
        return {
            "flattening_score": None,
            "flattened_side": None,
        }

    x_band = max(width * 0.08, 1e-12)
    y_band = max(height * 0.08, 1e-12)

    left_ratio = sum(1 for x, _ in points if x <= float(x_min) + x_band) / len(points)
    right_ratio = sum(1 for x, _ in points if x >= float(x_max) - x_band) / len(points)
    bottom_ratio = sum(1 for _, y in points if y <= float(y_min) + y_band) / len(points)
    top_ratio = sum(1 for _, y in points if y >= float(y_max) - y_band) / len(points)

    side_map = {
        "left": left_ratio,
        "right": right_ratio,
        "bottom": bottom_ratio,
        "top": top_ratio,
    }
    sorted_sides = sorted(side_map.items(), key=lambda item: item[1], reverse=True)
    best_side, best_ratio = sorted_sides[0]
    second_ratio = sorted_sides[1][1] if len(sorted_sides) > 1 else 0.0

    flattened_side = best_side
    if best_ratio >= 0.12 and second_ratio >= best_ratio * 0.85:
        flattened_side = "multi"

    flattening_score = min(1.0, max(0.0, best_ratio / 0.2))
    if best_ratio < 0.06:
        flattened_side = None

    return {
        "flattening_score": flattening_score,
        "flattened_side": flattened_side,
    }


def _orientation(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> bool:
    return (
        min(a[0], c[0]) - 1e-12 <= b[0] <= max(a[0], c[0]) + 1e-12
        and min(a[1], c[1]) - 1e-12 <= b[1] <= max(a[1], c[1]) + 1e-12
    )


def _segments_intersect(
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    p4: tuple[float, float],
) -> bool:
    o1 = _orientation(p1, p2, p3)
    o2 = _orientation(p1, p2, p4)
    o3 = _orientation(p3, p4, p1)
    o4 = _orientation(p3, p4, p2)

    if (o1 > 0 > o2 or o1 < 0 < o2) and (o3 > 0 > o4 or o3 < 0 < o4):
        return True

    if abs(o1) <= 1e-12 and _on_segment(p1, p3, p2):
        return True
    if abs(o2) <= 1e-12 and _on_segment(p1, p4, p2):
        return True
    if abs(o3) <= 1e-12 and _on_segment(p3, p1, p4):
        return True
    if abs(o4) <= 1e-12 and _on_segment(p3, p2, p4):
        return True
    return False


def _self_intersection_count(points: list[tuple[float, float]]) -> int:
    if len(points) < 4:
        return 0
    count = 0
    segments = [(points[i], points[i + 1]) for i in range(len(points) - 1)]
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            if abs(i - j) <= 1:
                continue
            if i == 0 and j == len(segments) - 1:
                continue
            if _segments_intersect(segments[i][0], segments[i][1], segments[j][0], segments[j][1]):
                count += 1
    return count


def _estimate_cycles(points: list[tuple[float, float]]) -> list[list[tuple[float, float]]]:
    if len(points) < 16:
        return []

    center = _center_metrics(points)
    cx = center["center_x"]
    cy = center["center_y"]
    if cx is None or cy is None:
        return []

    phases = [math.atan2(y - cy, x - cx) for x, y in points]
    unwrapped: list[float] = [phases[0]]
    for i in range(1, len(phases)):
        diff = phases[i] - phases[i - 1]
        while diff > math.pi:
            diff -= 2 * math.pi
        while diff < -math.pi:
            diff += 2 * math.pi
        unwrapped.append(unwrapped[-1] + diff)

    boundaries = [0]
    base_turn = math.floor(unwrapped[0] / (2 * math.pi))
    next_turn = base_turn + 1
    for idx, value in enumerate(unwrapped[1:], start=1):
        if value >= next_turn * 2 * math.pi:
            boundaries.append(idx)
            next_turn += 1
    boundaries.append(len(points) - 1)

    cycles: list[list[tuple[float, float]]] = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        if end - start + 1 >= 8:
            cycles.append(points[start:end + 1])
    return cycles if len(cycles) >= 2 else []


def _resample_cycle_by_angle(points: list[tuple[float, float]], sample_count: int = 64) -> list[float]:
    center = _center_metrics(points)
    cx = center["center_x"]
    cy = center["center_y"]
    if cx is None or cy is None:
        return []

    bucket_values: list[list[float]] = [[] for _ in range(sample_count)]
    for x, y in points:
        angle = math.atan2(y - cy, x - cx)
        if angle < 0:
            angle += 2 * math.pi
        idx = min(sample_count - 1, int(angle / (2 * math.pi) * sample_count))
        radius = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        bucket_values[idx].append(radius)

    profile: list[float] = []
    for bucket in bucket_values:
        profile.append(_safe_mean(bucket) or 0.0)

    mean_r = _safe_mean(profile)
    if mean_r in (None, 0.0):
        return profile
    return [v / mean_r for v in profile]


def _profile_similarity(a: list[float], b: list[float]) -> float | None:
    if not a or not b or len(a) != len(b):
        return None
    diff = _safe_mean([abs(x - y) for x, y in zip(a, b)])
    if diff is None:
        return None
    return max(0.0, 1.0 - diff)


def _repetition_metrics(points: list[tuple[float, float]]) -> dict[str, float | None]:
    cycles = _estimate_cycles(points)
    if len(cycles) < 2:
        return {
            "repetition_score": None,
            "cycle_shape_similarity": None,
        }

    profiles = [_resample_cycle_by_angle(cycle, sample_count=64) for cycle in cycles]
    similarities: list[float] = []
    for i in range(len(profiles) - 1):
        similarity = _profile_similarity(profiles[i], profiles[i + 1])
        if similarity is not None:
            similarities.append(similarity)

    cycle_similarity = _safe_mean(similarities)
    if cycle_similarity is None:
        return {
            "repetition_score": None,
            "cycle_shape_similarity": None,
        }

    repetition_score = max(0.0, min(1.0, cycle_similarity))
    return {
        "repetition_score": repetition_score,
        "cycle_shape_similarity": cycle_similarity,
    }


def _closure_ratio(points: list[tuple[float, float]], width: float | None, height: float | None) -> float | None:
    if len(points) < 2:
        return None
    x1, y1 = points[0]
    x2, y2 = points[-1]
    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    scale = max(width or 0.0, height or 0.0, 1e-12)
    return dist / scale


def _roundness_score(axis_ratio: float | None) -> float | None:
    if axis_ratio is None or axis_ratio <= 0:
        return None
    return max(0.0, min(1.0, 1.0 / axis_ratio))


def _orbit_rotation_vote(points: list[tuple[float, float]]) -> int:
    n = len(points)
    if n < 3:
        return 0

    vote = 0
    for i in range(n - 2):
        ax, ay = points[i]
        bx, by = points[i + 1]
        cx, cy = points[i + 2]
        cross = (bx - ax) * (cy - by) - (by - ay) * (cx - bx)
        if cross > 0:
            vote += 1
        elif cross < 0:
            vote -= 1
    return vote


def _precession_direction(points_1x: list[tuple[float, float]], rotation_direction: Any) -> str | None:
    if rotation_direction not in (0, 1, "0", "1"):
        return None

    vote = _orbit_rotation_vote(points_1x)
    if vote == 0:
        return None

    orbit_dir = "ccw" if vote > 0 else "cw"
    machine_dir = "ccw" if str(rotation_direction) == "1" else "cw"
    return "正进动" if orbit_dir == machine_dir else "反进动"


def _project_to_axes(points: list[tuple[float, float]], angle_deg: float | None) -> list[tuple[float, float]]:
    if not points or angle_deg is None:
        return []
    center = _center_metrics(points)
    cx = center["center_x"]
    cy = center["center_y"]
    if cx is None or cy is None:
        return []

    angle = math.radians(angle_deg)
    ux = math.cos(angle)
    uy = math.sin(angle)
    vx = -math.sin(angle)
    vy = math.cos(angle)

    result: list[tuple[float, float]] = []
    for x, y in points:
        dx = x - cx
        dy = y - cy
        u = dx * ux + dy * uy
        v = dx * vx + dy * vy
        result.append((u, v))
    return result


def _two_x_lobe_metrics(points: list[tuple[float, float]], angle_deg: float | None) -> dict[str, float | int | None]:
    projected = _project_to_axes(points, angle_deg)
    if len(projected) < 8:
        return {
            "self_intersection_count": 0,
            "figure_eight_score": None,
            "double_lobe_score": None,
            "waist_width": None,
            "lobe_width": None,
            "waist_constriction_ratio": None,
            "banana_bending_score": None,
            "quadratic_bending": None,
            "lobe_area_asymmetry": None,
            "lobe_size_asymmetry": None,
            "shape_asymmetry_score": None,
        }

    intersection_count = _self_intersection_count(points)

    us = [u for u, _ in projected]
    vs = [v for _, v in projected]
    u_abs_max = max((abs(v) for v in us), default=0.0)
    if u_abs_max <= 1e-12:
        return {
            "self_intersection_count": intersection_count,
            "figure_eight_score": None,
            "double_lobe_score": None,
            "waist_width": None,
            "lobe_width": None,
            "waist_constriction_ratio": None,
            "banana_bending_score": None,
            "quadratic_bending": None,
            "lobe_area_asymmetry": None,
            "lobe_size_asymmetry": None,
            "shape_asymmetry_score": None,
        }

    waist_band = max(u_abs_max * 0.15, 1e-12)
    lobe_band_low = u_abs_max * 0.45
    lobe_band_high = u_abs_max * 0.85

    waist_vs = [abs(v) for u, v in projected if abs(u) <= waist_band]
    lobe_vs = [abs(v) for u, v in projected if lobe_band_low <= abs(u) <= lobe_band_high]

    waist_width = (2 * max(waist_vs)) if waist_vs else None
    lobe_width = (2 * max(lobe_vs)) if lobe_vs else None
    waist_ratio = None
    if waist_width is not None and lobe_width not in (None, 0.0):
        waist_ratio = waist_width / lobe_width

    positive_u = [abs(v) for u, v in projected if u >= 0]
    negative_u = [abs(v) for u, v in projected if u < 0]
    pos_size = max(positive_u, default=None)
    neg_size = max(negative_u, default=None)

    lobe_size_asymmetry = None
    if pos_size not in (None, 0.0) and neg_size is not None:
        lobe_size_asymmetry = abs(pos_size - neg_size) / max(pos_size, neg_size, 1e-12)

    pos_area = _polygon_area([(u, v) for u, v in projected if u >= 0])
    neg_area = _polygon_area([(u, v) for u, v in projected if u < 0])
    lobe_area_asymmetry = None
    if pos_area not in (None, 0.0) and neg_area is not None:
        lobe_area_asymmetry = abs(pos_area - neg_area) / max(pos_area, neg_area, 1e-12)

    # 二次弯曲拟合 v = a*u^2 + b*u + c
    quadratic_a = None
    if len(projected) >= 5:
        xs = us
        ys = vs
        s_x0 = float(len(xs))
        s_x1 = sum(xs)
        s_x2 = sum(x * x for x in xs)
        s_x3 = sum(x * x * x for x in xs)
        s_x4 = sum(x * x * x * x for x in xs)
        s_y = sum(ys)
        s_xy = sum(x * y for x, y in zip(xs, ys))
        s_x2y = sum((x * x) * y for x, y in zip(xs, ys))
        coeffs = _solve_3x3(
            [
                [s_x4, s_x3, s_x2],
                [s_x3, s_x2, s_x1],
                [s_x2, s_x1, s_x0],
            ],
            [s_x2y, s_xy, s_y],
        )
        if coeffs is not None:
            quadratic_a = coeffs[0]

    banana_score = None
    if quadratic_a is not None:
        banana_score = min(1.0, abs(quadratic_a) * max(u_abs_max, 1.0))

    double_lobe_score = None
    if waist_ratio is not None:
        double_lobe_score = max(0.0, min(1.0, 1.0 - waist_ratio))

    figure_eight_score = None
    if double_lobe_score is not None:
        base = double_lobe_score
        if intersection_count > 0:
            base = min(1.0, base + 0.35)
        figure_eight_score = base

    asym_parts = [v for v in [lobe_area_asymmetry, lobe_size_asymmetry] if v is not None]
    shape_asymmetry_score = _safe_mean(asym_parts) if asym_parts else None

    return {
        "self_intersection_count": intersection_count,
        "figure_eight_score": figure_eight_score,
        "double_lobe_score": double_lobe_score,
        "waist_width": waist_width,
        "lobe_width": lobe_width,
        "waist_constriction_ratio": waist_ratio,
        "banana_bending_score": banana_score,
        "quadratic_bending": quadratic_a,
        "lobe_area_asymmetry": lobe_area_asymmetry,
        "lobe_size_asymmetry": lobe_size_asymmetry,
        "shape_asymmetry_score": shape_asymmetry_score,
    }


def _solve_3x3(A: list[list[float]], B: list[float]) -> tuple[float, float, float] | None:
    M = [row[:] + [b] for row, b in zip(A, B)]
    n = 3
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[pivot][col]) < 1e-12:
            return None
        if pivot != col:
            M[col], M[pivot] = M[pivot], M[col]

        pivot_val = M[col][col]
        for j in range(col, n + 1):
            M[col][j] /= pivot_val

        for row in range(n):
            if row == col:
                continue
            factor = M[row][col]
            for j in range(col, n + 1):
                M[row][j] -= factor * M[col][j]

    return M[0][3], M[1][3], M[2][3]


def _shape_tags(
    width: float | None,
    height: float | None,
    flattening_score: float | None,
    self_intersection_count: int,
    repetition_score: float | None,
) -> list[str]:
    tags: list[str] = []

    if width is not None and height is not None:
        ratio = max(width, height) / max(min(width, height), 1e-12)
        if ratio <= 1.2:
            tags.append("近圆形")
        elif ratio <= 2.0:
            tags.append("椭圆形")
        else:
            tags.append("扁长椭圆")

    if flattening_score is not None:
        if flattening_score >= 0.65:
            tags.append("截断/削平明显")
        elif flattening_score >= 0.35:
            tags.append("存在一定削平")

    if self_intersection_count >= 1:
        tags.append("存在自交")

    if repetition_score is not None:
        if repetition_score >= 0.8:
            tags.append("多周期重复性高")
        elif repetition_score <= 0.5:
            tags.append("多周期重复性一般")

    if not tags:
        tags.append("形态较规则")

    return tags


def _centerline_tags(center_offset_radius: float | None, repetition_score: float | None) -> list[str]:
    tags: list[str] = []
    if center_offset_radius is not None:
        if center_offset_radius >= 0.3:
            tags.append("中心偏移明显")
        elif center_offset_radius > 0:
            tags.append("存在一定中心偏移")

    if repetition_score is not None:
        if repetition_score >= 0.8:
            tags.append("轨迹重复性较好")
        elif repetition_score <= 0.5:
            tags.append("轨迹重复性偏弱")

    return tags


def _one_x_tags(
    axis_ratio: float | None,
    precession_direction: str | None,
) -> list[str]:
    tags: list[str] = []
    if axis_ratio is not None:
        if axis_ratio <= 1.2:
            tags.append("1X近圆")
        elif axis_ratio <= 2.0:
            tags.append("1X椭圆")
        else:
            tags.append("1X扁长")
    if precession_direction is not None:
        tags.append(precession_direction)
    return tags


def _two_x_tags(
    figure_eight_score: float | None,
    waist_ratio: float | None,
    banana_score: float | None,
    asymmetry_score: float | None,
) -> list[str]:
    tags: list[str] = []
    if figure_eight_score is not None and figure_eight_score >= 0.55:
        tags.append("2X呈8字/双叶")
    if waist_ratio is not None and waist_ratio <= 0.55:
        tags.append("2X腰部收缩")
    if banana_score is not None and banana_score >= 0.2:
        tags.append("2X存在香蕉形弯曲")
    if asymmetry_score is not None and asymmetry_score >= 0.2:
        tags.append("2X存在不对称")
    return tags


def _build_feature_detail(data: dict[str, Any], rotation_direction: Any = None) -> OrbitCenterlineFeatureDetail:
    raw_points = _extract_xy(data.get("points") or [])
    points_1x = _extract_xy(data.get("points_1x") or [])
    points_2x = _extract_xy(data.get("points_2x") or [])

    bbox = _bbox_metrics(raw_points)
    center = _center_metrics(raw_points)
    principal = _principal_axis_metrics(raw_points)
    area = _polygon_area(raw_points)
    path_complexity = _path_complexity(raw_points, bbox["width"], bbox["height"])
    flattening = _flattening_metrics(raw_points, bbox)
    self_intersections = _self_intersection_count(raw_points)
    convex_fill_ratio = _convex_fill_ratio(raw_points, area)
    repetition = _repetition_metrics(raw_points)

    bbox_1x = _bbox_metrics(points_1x)
    center_1x = _center_metrics(points_1x)
    principal_1x = _principal_axis_metrics(points_1x)
    area_1x = _polygon_area(points_1x)
    closure_ratio_1x = _closure_ratio(points_1x, bbox_1x["width"], bbox_1x["height"])
    roundness_1x = _roundness_score(principal_1x["axis_ratio"])
    precession_1x = _precession_direction(points_1x, rotation_direction)

    bbox_2x = _bbox_metrics(points_2x)
    principal_2x = _principal_axis_metrics(points_2x)
    area_2x = _polygon_area(points_2x)
    two_x_metrics = _two_x_lobe_metrics(points_2x, principal_2x["principal_angle_deg"])

    two_x_to_one_x_ratio = None
    if principal_1x["major_axis"] not in (None, 0.0) and principal_2x["major_axis"] is not None:
        two_x_to_one_x_ratio = principal_2x["major_axis"] / principal_1x["major_axis"]

    one_x_dominant = bool(two_x_to_one_x_ratio is None or two_x_to_one_x_ratio < 0.3)
    two_x_significant = bool(two_x_to_one_x_ratio is not None and two_x_to_one_x_ratio >= 0.3)

    raw_to_one_x_angle_diff = None
    if principal["principal_angle_deg"] is not None and principal_1x["principal_angle_deg"] is not None:
        diff = abs(principal["principal_angle_deg"] - principal_1x["principal_angle_deg"])
        raw_to_one_x_angle_diff = min(diff, 180.0 - diff if diff <= 180.0 else diff % 180.0)

    shape_tags = _shape_tags(
        bbox["width"],
        bbox["height"],
        flattening["flattening_score"] if isinstance(flattening["flattening_score"], (int, float)) else None,
        self_intersections,
        repetition["repetition_score"],
    )
    center_tags = _centerline_tags(center["center_offset_radius"], repetition["repetition_score"])
    one_x_tags = _one_x_tags(principal_1x["axis_ratio"], precession_1x)
    two_x_tags = _two_x_tags(
        two_x_metrics["figure_eight_score"] if isinstance(two_x_metrics["figure_eight_score"], (int, float)) else None,
        two_x_metrics["waist_constriction_ratio"] if isinstance(two_x_metrics["waist_constriction_ratio"], (int, float)) else None,
        two_x_metrics["banana_bending_score"] if isinstance(two_x_metrics["banana_bending_score"], (int, float)) else None,
        two_x_metrics["shape_asymmetry_score"] if isinstance(two_x_metrics["shape_asymmetry_score"], (int, float)) else None,
    )

    return OrbitCenterlineFeatureDetail(
        point_count=len(raw_points),
        width=_round_float(bbox["width"], 6),
        height=_round_float(bbox["height"], 6),
        orbit_area=_round_float(area, 6),
        center_offset_radius=_round_float(center["center_offset_radius"], 6),
        principal_angle_deg=_round_float(principal["principal_angle_deg"], 6),
        path_complexity=_round_float(path_complexity, 6),
        flattening_score=_round_float(flattening["flattening_score"] if isinstance(flattening["flattening_score"], (int, float)) else None, 6),
        flattened_side=flattening["flattened_side"] if isinstance(flattening["flattened_side"], str) else None,
        self_intersection_count=self_intersections,
        convex_fill_ratio=_round_float(convex_fill_ratio, 6),
        repetition_score=_round_float(repetition["repetition_score"], 6),
        cycle_shape_similarity=_round_float(repetition["cycle_shape_similarity"], 6),
        one_x_point_count=len(points_1x),
        one_x_width=_round_float(bbox_1x["width"], 6),
        one_x_height=_round_float(bbox_1x["height"], 6),
        one_x_major_axis=_round_float(principal_1x["major_axis"], 6),
        one_x_minor_axis=_round_float(principal_1x["minor_axis"], 6),
        one_x_axis_ratio=_round_float(principal_1x["axis_ratio"], 6),
        one_x_eccentricity_ratio=_round_float(principal_1x["eccentricity_ratio"], 6),
        one_x_orbit_area=_round_float(area_1x, 6),
        one_x_closure_ratio=_round_float(closure_ratio_1x, 6),
        one_x_roundness_score=_round_float(roundness_1x, 6),
        one_x_principal_angle_deg=_round_float(principal_1x["principal_angle_deg"], 6),
        one_x_precession_direction=precession_1x,
        raw_to_one_x_angle_diff_deg=_round_float(raw_to_one_x_angle_diff, 6),
        one_x_center_x=_round_float(center_1x["center_x"], 6),
        one_x_center_y=_round_float(center_1x["center_y"], 6),
        one_x_center_offset_radius=_round_float(center_1x["center_offset_radius"], 6),
        two_x_point_count=len(points_2x),
        two_x_width=_round_float(bbox_2x["width"], 6),
        two_x_height=_round_float(bbox_2x["height"], 6),
        two_x_major_axis=_round_float(principal_2x["major_axis"], 6),
        two_x_minor_axis=_round_float(principal_2x["minor_axis"], 6),
        two_x_axis_ratio=_round_float(principal_2x["axis_ratio"], 6),
        two_x_orbit_area=_round_float(area_2x, 6),
        two_x_principal_angle_deg=_round_float(principal_2x["principal_angle_deg"], 6),
        two_x_to_one_x_ratio=_round_float(two_x_to_one_x_ratio, 6),
        one_x_dominant=one_x_dominant,
        two_x_significant=two_x_significant,
        two_x_self_intersection_count=int(two_x_metrics["self_intersection_count"] or 0),
        two_x_figure_eight_score=_round_float(two_x_metrics["figure_eight_score"] if isinstance(two_x_metrics["figure_eight_score"], (int, float)) else None, 6),
        two_x_double_lobe_score=_round_float(two_x_metrics["double_lobe_score"] if isinstance(two_x_metrics["double_lobe_score"], (int, float)) else None, 6),
        two_x_waist_width=_round_float(two_x_metrics["waist_width"] if isinstance(two_x_metrics["waist_width"], (int, float)) else None, 6),
        two_x_lobe_width=_round_float(two_x_metrics["lobe_width"] if isinstance(two_x_metrics["lobe_width"], (int, float)) else None, 6),
        two_x_waist_constriction_ratio=_round_float(two_x_metrics["waist_constriction_ratio"] if isinstance(two_x_metrics["waist_constriction_ratio"], (int, float)) else None, 6),
        two_x_banana_bending_score=_round_float(two_x_metrics["banana_bending_score"] if isinstance(two_x_metrics["banana_bending_score"], (int, float)) else None, 6),
        two_x_quadratic_bending=_round_float(two_x_metrics["quadratic_bending"] if isinstance(two_x_metrics["quadratic_bending"], (int, float)) else None, 9),
        two_x_lobe_area_asymmetry=_round_float(two_x_metrics["lobe_area_asymmetry"] if isinstance(two_x_metrics["lobe_area_asymmetry"], (int, float)) else None, 6),
        two_x_lobe_size_asymmetry=_round_float(two_x_metrics["lobe_size_asymmetry"] if isinstance(two_x_metrics["lobe_size_asymmetry"], (int, float)) else None, 6),
        two_x_shape_asymmetry_score=_round_float(two_x_metrics["shape_asymmetry_score"] if isinstance(two_x_metrics["shape_asymmetry_score"], (int, float)) else None, 6),
        shape_tags=shape_tags,
        centerline_tags=center_tags,
        one_x_tags=one_x_tags,
        two_x_tags=two_x_tags,
    )


def _build_shape_findings(detail: OrbitCenterlineFeatureDetail) -> list[str]:
    findings: list[str] = []

    if detail.width is not None and detail.height is not None:
        findings.append(f"原始轨迹宽度={detail.width}，高度={detail.height}")

    if detail.orbit_area is not None:
        findings.append(f"原始轨迹面积={detail.orbit_area}")

    if detail.principal_angle_deg is not None:
        findings.append(f"原始轨迹主轴方向角约为 {detail.principal_angle_deg}°")

    if detail.path_complexity is not None:
        findings.append(f"原始轨迹路径复杂度={detail.path_complexity}")

    if detail.flattening_score is not None:
        findings.append(f"原始轨迹削平得分={detail.flattening_score}")

    if detail.flattened_side:
        findings.append(f"原始轨迹主要削平方向={detail.flattened_side}")

    if detail.self_intersection_count > 0:
        findings.append(f"原始轨迹自交次数={detail.self_intersection_count}")

    if detail.convex_fill_ratio is not None:
        findings.append(f"原始轨迹凸包填充率={detail.convex_fill_ratio}")

    findings.extend(detail.shape_tags)
    return findings[:10]


def _build_centerline_findings(detail: OrbitCenterlineFeatureDetail) -> list[str]:
    findings: list[str] = []

    if detail.center_offset_radius is not None:
        findings.append(f"中心偏移半径={detail.center_offset_radius}")

    if detail.repetition_score is not None:
        findings.append(f"原始轨迹重复性得分={detail.repetition_score}")

    if detail.cycle_shape_similarity is not None:
        findings.append(f"周期间形状相似度={detail.cycle_shape_similarity}")

    findings.extend(detail.centerline_tags)
    return findings[:8]


def _build_one_x_findings(detail: OrbitCenterlineFeatureDetail) -> list[str]:
    findings: list[str] = []

    if detail.one_x_width is not None and detail.one_x_height is not None:
        findings.append(f"1X轨迹宽度={detail.one_x_width}，高度={detail.one_x_height}")

    if detail.one_x_axis_ratio is not None:
        findings.append(f"1X主副轴比={detail.one_x_axis_ratio}")

    if detail.one_x_principal_angle_deg is not None:
        findings.append(f"1X主轴方向角约为 {detail.one_x_principal_angle_deg}°")

    if detail.one_x_precession_direction is not None:
        findings.append(f"1X进动方向={detail.one_x_precession_direction}")

    if detail.raw_to_one_x_angle_diff_deg is not None:
        findings.append(f"原始轨迹与1X主轴夹角差={detail.raw_to_one_x_angle_diff_deg}°")

    findings.extend(detail.one_x_tags)
    return findings[:8]


def _build_two_x_findings(detail: OrbitCenterlineFeatureDetail) -> list[str]:
    findings: list[str] = []

    if detail.two_x_width is not None and detail.two_x_height is not None:
        findings.append(f"2X轨迹宽度={detail.two_x_width}，高度={detail.two_x_height}")

    if detail.two_x_to_one_x_ratio is not None:
        findings.append(f"2X/1X轨迹尺度比={detail.two_x_to_one_x_ratio}")

    if detail.two_x_figure_eight_score is not None:
        findings.append(f"2X 8字形得分={detail.two_x_figure_eight_score}")

    if detail.two_x_waist_constriction_ratio is not None:
        findings.append(f"2X腰部收缩比={detail.two_x_waist_constriction_ratio}")

    if detail.two_x_banana_bending_score is not None:
        findings.append(f"2X香蕉形弯曲得分={detail.two_x_banana_bending_score}")

    if detail.two_x_shape_asymmetry_score is not None:
        findings.append(f"2X形态不对称得分={detail.two_x_shape_asymmetry_score}")

    findings.extend(detail.two_x_tags)
    return findings[:8]


def _build_summary(detail: OrbitCenterlineFeatureDetail) -> list[str]:
    summary: list[str] = []

    if detail.one_x_dominant:
        summary.append("轨迹以 1X 同步分量主导")
    elif detail.two_x_significant:
        summary.append("轨迹中 2X 分量具有一定贡献")

    if detail.repetition_score is not None:
        if detail.repetition_score >= 0.8:
            summary.append("原始轨迹多周期重复性较好")
        elif detail.repetition_score <= 0.5:
            summary.append("原始轨迹多周期重复性一般")

    if detail.flattening_score is not None and detail.flattening_score >= 0.35:
        summary.append("原始轨迹存在一定削平/截断特征")

    if detail.self_intersection_count > 0:
        summary.append("原始轨迹存在自交或多环倾向")

    if detail.one_x_precession_direction is not None:
        summary.append(f"1X轨迹表现为{detail.one_x_precession_direction}")

    if detail.two_x_figure_eight_score is not None and detail.two_x_figure_eight_score >= 0.55:
        summary.append("2X轨迹呈现较明显的8字/双叶趋势")

    return summary[:8]


def _build_suspected_faults(detail: OrbitCenterlineFeatureDetail) -> list[str]:
    suspects: list[str] = []

    if detail.one_x_dominant and detail.one_x_axis_ratio is not None and detail.one_x_axis_ratio <= 2.5:
        suspects.append("疑似同步类振动问题，偏向不平衡方向")

    if detail.two_x_significant:
        suspects.append("2X 成分较明显，需关注不对中方向")

    if detail.self_intersection_count > 0 or (detail.path_complexity is not None and detail.path_complexity >= 5.0):
        suspects.append("轨迹复杂度或自交较明显，需关注松动、碰摩或非线性接触")

    if detail.flattening_score is not None and detail.flattening_score >= 0.45:
        suspects.append("轨迹存在削平/截断迹象，需排查接触限制、碰摩或单侧约束")

    if detail.center_offset_radius is not None and detail.one_x_major_axis not in (None, 0.0):
        if detail.center_offset_radius / detail.one_x_major_axis >= 0.25:
            suspects.append("中心偏移较明显，需关注偏心、热弯曲或载荷偏置")

    if detail.one_x_precession_direction == "反进动":
        suspects.append("1X表现为反进动，需关注油膜不稳定或异常转子动力学特征")

    if detail.two_x_figure_eight_score is not None and detail.two_x_figure_eight_score >= 0.55:
        suspects.append("2X呈8字/双叶，需关注不对中或耦合类问题")

    if detail.two_x_shape_asymmetry_score is not None and detail.two_x_shape_asymmetry_score >= 0.2:
        suspects.append("2X存在不对称，需关注非线性接触、偏置或不均匀支撑")

    deduped: list[str] = []
    seen: set[str] = set()
    for item in suspects:
        if item not in seen:
            seen.add(item)
            deduped.append(item)

    return deduped[:8]


async def extract_orbit_centerline_features_tool(
    machine_id: str,
    bearing_id: str,
    time: str | None = None,
    time_ms: str | None = None,
) -> dict[str, Any]:
    """
    提取轴心轨迹与中心偏置特征。

    输入格式：
    {
      "machine_id": ".",
      "bearing_id": "type_num/type_enum=70 的轴承 ID",
      "time": "趋势分析返回的异常毫秒时间戳，或可解析时间字符串",
      "time_ms": "趋势分析返回的异常毫秒时间戳，可选，优先于 time"
    }
    """
    payload_time = str(time_ms or time or "")
    if not payload_time:
        raise ValueError("time or time_ms is required")
    orbit_payload = await _get_orbit_data_impl(machine_id, bearing_id, payload_time)

    machine_id = str(orbit_payload.get("machine_id") or "")
    bearing_id = str(orbit_payload.get("bearing_id") or "")
    time_ms = str(orbit_payload.get("time_ms") or "")
    probe_ids = orbit_payload.get("probe_ids") or []
    data = orbit_payload.get("data") or {}
    if not isinstance(data, dict):
        data = {}

    rotation_direction = orbit_payload.get("rotation_direction")
    if rotation_direction is None:
        rotation_direction = data.get("rotation_direction")

    feature_details = _build_feature_detail(data, rotation_direction=rotation_direction)

    result = OrbitCenterlineAnalysisResult(
        machine_id=machine_id,
        bearing_id=bearing_id,
        time_ms=time_ms,
        summary=_build_summary(feature_details),
        shape_findings=_build_shape_findings(feature_details),
        centerline_findings=_build_centerline_findings(feature_details),
        one_x_findings=_build_one_x_findings(feature_details),
        two_x_findings=_build_two_x_findings(feature_details),
        suspected_faults=_build_suspected_faults(feature_details),
        feature_details=feature_details,
        probe_ids=probe_ids if isinstance(probe_ids, list) else [],
    )
    return result.model_dump()


async def main() -> None:
    """
    用法:
    python extract_orbit_centerline_features_tool.py <machine_id> <bearing_id> <time_or_time_ms>
    """
    if len(sys.argv) < 4:
        raise SystemExit(
            "用法: python extract_orbit_centerline_features_tool.py <machine_id> <bearing_id> <time_or_time_ms>"
        )

    result = await extract_orbit_centerline_features_tool(
        machine_id=sys.argv[1],
        bearing_id=sys.argv[2],
        time=sys.argv[3],
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
