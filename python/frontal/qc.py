# -*- coding: utf-8 -*-
"""qc.py — run43 최소 QC. **판정이 아니다.** 결과를 억지로 내지 않기 위한 장치만 둔다.

  1) 튐(jump)      : 관절 좌표가 0.2 s 이동중앙값에서 span 의 JUMP_DEV_RATIO 배 이상 벗어난 프레임 → 미검출로 취급
  2) 긴 공백(gap)  : 핵심 관절(골반·무릎·발목) 미검출이 MAX_GAP_S 이상 이어진 구간 → 보간값을 지표에 쓰지 않는다
  3) 경고 목록     : {code, severity(info/warn/error), message}. error 가 하나라도 있으면 result_usable=False

연속영상(fps 있음) 경로에서만 쓴다. 키프레임 경로(AI-Hub 검증)는 건드리지 않는다.
모든 상수는 잠정값 — OpenCap 45개(개발 데이터) 분포에서 여유를 두고 잡았다. run41·외부 11개 미사용.
numpy 만 쓴다 (웹 Pyodide 에서 scipy 를 받지 않기 위해).
"""
from __future__ import annotations

import numpy as np

from frontal import landmarks as L

CORE_JOINTS = ["Left Hip", "Right Hip", "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"]
JUMP_WINDOW_S = 0.2        # 이동중앙값 창
JUMP_DEV_RATIO = 0.25      # OpenCap 정면 최대 0.048, 전 각도(정상) 최대 0.22, 추적 붕괴 영상(subject8 Cam4) 최대 1.14
JUMP_FRAC_ERROR = 0.05     # 튐 프레임 비율이 이보다 크면 추적 자체를 믿지 않는다
MAX_GAP_S = 0.3            # 이보다 긴 핵심 관절 공백은 보간값을 지표로 쓰지 않는다
BOTTOM_GUARD_S = 0.1       # 저점 ±이 범위에 긴 공백이 걸치면 그 반복은 '사용 불가'
MISSING_FRAC_WARN = 0.2    # 핵심 관절 미검출 비율 경고선

SEVERITY = ("info", "warn", "error")


def _span(P):
    import warnings
    g = lambda n: P[:, L.IDX[n]]                                    # noqa: E731
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # 미검출 프레임의 빈 평균
        sh = np.nanmean([g("Left Shoulder")[:, 1], g("Right Shoulder")[:, 1]], axis=0)
        an = np.nanmean([g("Left Ankle")[:, 1], g("Right Ankle")[:, 1]], axis=0)
        s = float(np.nanmedian(np.abs(sh - an)))
    return s if np.isfinite(s) and s > 1e-6 else float("nan")


def _rolling_nanmedian(v, k):
    h = k // 2
    pad = np.r_[np.full(h, np.nan), v, np.full(h, np.nan)]
    W = np.lib.stride_tricks.sliding_window_view(pad, k)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmedian(W, axis=1)


def core_missing(P):
    idx = [L.IDX[n] for n in CORE_JOINTS]
    return ~np.isfinite(P[:, idx, :]).all(axis=(1, 2))


def jump_mask(P, fps, dev_ratio=JUMP_DEV_RATIO, window_s=JUMP_WINDOW_S):
    """원본(보간 전) 좌표에서 튄 프레임. 반환 (mask, info)."""
    T = P.shape[0]
    span = _span(P)
    if not np.isfinite(span) or T < 3:
        return np.zeros(T, bool), {"span": span, "max_dev_ratio": None}
    k = max(3, int(round(window_s * fps)) | 1)
    dev = np.zeros(T)
    for n in CORE_JOINTS:
        for ax in (0, 1):
            v = P[:, L.IDX[n], ax]
            d = np.abs(v - _rolling_nanmedian(v, k)) / span
            dev = np.fmax(dev, np.nan_to_num(d, nan=0.0))
    m = dev > dev_ratio
    return m, {"span": span, "max_dev_ratio": float(dev.max()), "window_frames": k}


def runs_of(mask):
    e = np.flatnonzero(np.diff(np.r_[0, np.asarray(mask, int), 0]))
    return [(int(a), int(b)) for a, b in zip(e[::2], e[1::2])]


def long_gap_mask(missing, fps, max_gap_s=MAX_GAP_S):
    need = max(1, int(round(max_gap_s * fps)))
    out = np.zeros(len(missing), bool)
    runs = [(a, b) for a, b in runs_of(missing) if b - a >= need]
    for a, b in runs:
        out[a:b] = True
    return out, runs


def rep_usable(bottoms, long_gap, fps, guard_s=BOTTOM_GUARD_S):
    g = max(0, int(round(guard_s * fps)))
    T = len(long_gap)
    return [not bool(long_gap[max(0, int(b) - g):min(T, int(b) + g + 1)].any()) for b in bottoms]


def warn(code, severity, message, **extra):
    assert severity in SEVERITY
    d = {"code": code, "severity": severity, "message": message}
    d.update(extra)
    return d


# ── run50: 저점 관절 붕괴 (좌우 다리가 한쪽으로 겹쳐 잡힘) ─────────────────────
# 실제 개발 촬영(검정 바지·역광, IMG_7595)에서 저점 3/8 이 붕괴했는데 튐 검사로는 안 잡혔다(여러 프레임 지속).
# 정면에서 골반폭은 스쿼트 중 거의 변하지 않는다: OpenCap 0°·±40° 133 저점에서 저점/서기 골반폭 0.80~1.12, 좌우 순서 항상 일치.
HIP_RATIO_MIN, HIP_RATIO_MAX = 0.70, 1.43     # 잠정 — 위 분포 밖으로 여유
COLLAPSE_FRAC_ERROR = 0.25                    # 세트의 이 비율 이상이 붕괴면 추적 자체를 믿지 않는다


def rep_geometry_ok(P, bottoms, stand_window):
    """저점마다 (골반폭비, 좌우 순서 일치) 로 붕괴 여부. 반환 [(ok, hip_ratio, order_ok)]."""
    a, b = stand_window
    li, ri = L.IDX["Left Hip"], L.IDX["Right Hip"]
    with np.errstate(all="ignore"):
        hw_s = float(np.nanmedian(np.abs(P[a:b, ri, 0] - P[a:b, li, 0])))
    out = []
    for t in bottoms:
        t = int(t)
        s = [np.sign(P[t, L.IDX[f"Right {j}"], 0] - P[t, L.IDX[f"Left {j}"], 0]) for j in ("Hip", "Knee", "Ankle")]
        order_ok = len(set(s)) == 1 and s[0] != 0
        hr = abs(P[t, ri, 0] - P[t, li, 0]) / hw_s if np.isfinite(hw_s) and hw_s > 1e-6 else float("nan")
        ok = bool(order_ok and np.isfinite(hr) and HIP_RATIO_MIN <= hr <= HIP_RATIO_MAX)
        out.append((ok, float(hr), bool(order_ok)))
    return out


# run56 — C1(정면 상체 길이) 표시 조건 (SERVICE_ANALYSIS_SPEC 2-3: 어깨 가시성 QC 통과 시에만 출력).
#   새 임계는 없다: 어깨가 '관측됨' = 가시성 ≥ landmarks.DEFAULT_MIN_VISIBILITY(0.5, 이 값 미만은 이미 NaN 처리),
#   저점 범위 = BOTTOM_GUARD_S(핵심 관절과 같은 ±0.1 s). 기준자세는 창의 절반 이상에서 관측돼야
#   중앙값이 실제 관측값에서 나온다(중앙값의 붕괴점).
SHOULDERS = ["Left Shoulder", "Right Shoulder"]


def shoulder_missing(P):
    """보간 전 좌표에서 두 어깨 중 하나라도 관측되지 않은(가시성 미달·미검출) 프레임."""
    idx = [L.IDX[n] for n in SHOULDERS]
    return ~np.isfinite(P[:, idx, :]).all(axis=(1, 2))


def c1_visible(sh_missing, bottoms, stand_window, fps, guard_s=BOTTOM_GUARD_S):
    """반환 (기준자세 어깨 관측 여부, [반복별 저점 ±guard 어깨 관측 여부])."""
    T = len(sh_missing)
    a, b = stand_window
    seg = sh_missing[max(0, int(a)):min(T, int(b))]
    stand_ok = bool(len(seg) > 0 and (~seg).mean() > 0.5)
    g = max(0, int(round(guard_s * fps)))
    reps = [not bool(sh_missing[max(0, int(t) - g):min(T, int(t) + g + 1)].any()) for t in bottoms]
    return stand_ok, reps
