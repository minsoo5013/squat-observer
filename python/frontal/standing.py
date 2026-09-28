# -*- coding: utf-8 -*-
"""standing.py — 정면 트랙의 기준자세(standing reference) 추출.

**개념은 3D 트랙과 공유하지만 구현은 공유하지 않는다.**
squat_core.capture_standing_reference() 는
  · 3D 무릎각(knee_mean)의 최댓값으로 직립 프레임을 고르고,
  · '시퀀스 초반 25%' 라는 규칙을 쓴다 (클립당 16 키프레임 전제).
정면 트랙에서는 둘 다 쓸 수 없다 — 무릎각은 정면 투영에서 무너지고,
30fps 연속 영상에는 '초반 25%' 가 의미가 없다.

그래서 경로를 셋으로 나눈다. **서비스의 정상 경로는 명시적 캘리브레이션이다.**
  from_calibration_segment()  ★ 서비스 정상 경로 — "정면을 보고 2초간 서 주세요" 구간을 그대로 받는다
  from_continuous()             fallback — 캘리브레이션 구간이 없거나 실패했을 때만 추정한다
  from_keyframes()              AI-Hub 같은 희소 키프레임 입력 (검증용)
셋 다 z 를 쓰지 않고 **골반 높이(y)** 로만 직립을 다룬다.

왜 명시적 캘리브레이션이 정상 경로인가:
  서비스는 촬영 시작 시점에 "평소 스쿼트 준비자세를 취하고 2초간 그대로 유지하세요" 라고 안내할 수 있다.
  그러면 영상 전체를 뒤져 '가장 높은 골반 위치' 를 추정할 이유가 없다.
  A2 / A45 / D1 은 전부 기준자세를 분모로 쓰므로, 분모가 추정값이 아니라 지정 구간일수록 안정된다.
  촬영 흐름:  정면에 서기 → **평소 스쿼트 준비자세** → 2초 유지 → 그대로 스쿼트 시작

  ※ 차렷 자세를 요구하지 않는다. 기준은 해부학적 중립자세가 아니라
    "이 사람이 이번 세트를 시작하기 직전의 자기 상태" 다.
    팔짱을 끼고 할 사람은 팔짱 낀 채로, 팔을 뻗을 사람은 뻗은 채로 2초.
    차렷 → 2초 → 팔짱 → 스쿼트 로 만들면 기준자세와 실제 수행 자세가 달라진다.
  ※ 핵심 지표(A1/A2/A45/A6/D1)와 저점 검출은 어깨·팔을 전혀 쓰지 않는다 —
    어깨를 완전히 지워도 값이 bit 단위로 같다 (frontal/check_shoulder_dependence.py).
    따라서 준비자세의 팔 위치는 이들에 영향을 주지 않는다.

※ 아래 상수는 전부 **잠정값**이다. 실촬영 데이터로 재측정하기 전까지 확정이 아니다.
   특히 프레임 수에 의존하는 값을 30fps 로 비례 환산하지 않는다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from frontal import landmarks as L

# ── 잠정 파라미터 ────────────────────────────────────────────
EARLY_FRACTION = 0.25        # 키프레임 경로: 시퀀스 초반 비율 (AI-Hub 16프레임 기준)
STAND_SECONDS = 2.0          # 연속영상 경로: '기준자세 2초' 유지 요구
NEAR_TOP_RATIO = 0.97        # 골반 높이가 최댓값의 몇 배 이상이면 직립으로 볼 것인가
MOTION_TOL_RATIO = 0.010     # 창 안에서 허용할 골반 높이 변동 (span 대비)
# run43 — search_standing (시간 단위 자동 탐색). OpenCap 45개(개발)에서 측정한 분포로 잡은 잠정값.
SEARCH_SECONDS = (1.0, 0.5, 0.3)   # 긴 창부터 찾는다. OpenCap 반복 사이 서 있는 구간 0.3~1.35 s
SEARCH_RELIABILITY = {1.0: "high", 0.5: "medium", 0.3: "low"}
SEARCH_TOP_PERCENTILE = 98         # '최고 골반 높이' 를 max 대신 98 백분위로 (튀는 프레임 1장에 끌려가지 않게)
ANKLE_TOL_RATIO = 0.05             # 창 안 발목폭 변동 허용 (span 대비). 걸어 들어오는 구간 배제용.
                                   # OpenCap 정면(Cam2) 서기 창 최대 0.005, 전 각도 p90 0.026


@dataclass
class StandingReference:
    """정면 기준자세. 모든 값은 등방 단위, 좌표계는 y 위쪽이 +."""
    index: int                  # 대표 프레임
    window: tuple               # (start, end) — 연속영상에서 잡힌 구간, 키프레임이면 (i, i+1)
    hip_ankle_dy: float         # 직립 골반-발목 수직거리  (A2·D1 의 분모)
    knee_w: float               # 직립 무릎폭              (A2 의 분모)
    ankle_w: float              # 직립 발목폭              (A1·A45 의 분모)
    shoulder_w: float
    span_sh_ank: float          # 어깨~발목 세로길이 (머리~발목을 대체한 정규화 분모)
    trunk_span: float           # 어깨중점~골반중점 세로길이 (C1 의 분모)
    method: str
    seconds_held: float | None = None
    reliability: str | None = None   # run43: 'high'/'medium'/'low'/'failed' (search_standing 경로만 채움)
    note: str = ("정면 전용 기준자세. squat_core.capture_standing_reference 와 다른 구현이며 "
                 "무릎각을 쓰지 않는다.")


def _hip_ankle_dy(P):
    g = lambda n: P[:, L.IDX[n]]                                   # noqa: E731
    hip = np.nanmean([g("Left Hip")[:, 1], g("Right Hip")[:, 1]], axis=0)
    ank = np.nanmean([g("Left Ankle")[:, 1], g("Right Ankle")[:, 1]], axis=0)
    return hip - ank


def _measure(P, idx, window, method, seconds=None):
    g = lambda n: P[idx, L.IDX[n]]                                  # noqa: E731
    ls, rs = g("Left Shoulder"), g("Right Shoulder")
    lh, rh = g("Left Hip"), g("Right Hip")
    lk, rk = g("Left Knee"), g("Right Knee")
    la, ra = g("Left Ankle"), g("Right Ankle")
    sh_mid_y = float(np.nanmean([ls[1], rs[1]]))
    ank_mid_y = float(np.nanmean([la[1], ra[1]]))
    span = abs(sh_mid_y - ank_mid_y)
    return StandingReference(
        index=int(idx), window=tuple(int(v) for v in window),
        hip_ankle_dy=float(np.nanmean([lh[1], rh[1]]) - ank_mid_y),
        knee_w=float(abs(rk[0] - lk[0])),
        ankle_w=float(abs(ra[0] - la[0])),
        shoulder_w=float(abs(rs[0] - ls[0])),
        span_sh_ank=float(span) if np.isfinite(span) and span > 1e-6 else float("nan"),
        trunk_span=float(abs(sh_mid_y - np.nanmean([lh[1], rh[1]]))),
        method=method, seconds_held=seconds)


def from_calibration_segment(P, fps, start_s=0.0, duration_s=STAND_SECONDS,
                            ok_mask=None, motion_tol_ratio=MOTION_TOL_RATIO):
    """★ 서비스 정상 경로 — 안내에 따라 **평소 준비자세로** 정지해 있던 지정 구간을 기준자세로 쓴다.

    추정하지 않는다. 구간을 받아서 그 구간이 실제로 쓸 만한지 **검사만** 한다.
    반환 StandingReference 의 method 는
      "calibration"            정상
      "calibration:unsteady"   구간 안에서 골반이 흔들렸다 (값은 계산하되 호출부가 재촬영을 안내할 수 있다)
      "calibration:occluded"   구간 안에 쓸 수 있는 프레임이 부족하다
    합격/불합격 선은 정하지 않는다 — motion_tol_ratio 는 잠정값이고 실촬영에서 재측정한다.
    """
    if not fps or fps <= 0:
        raise ValueError("캘리브레이션 경로에는 fps 가 필요하다.")
    T = P.shape[0]
    a = max(0, int(round(start_s * fps)))
    b = min(T, a + max(2, int(round(duration_s * fps))))
    if b - a < 2:
        return _measure(P, min(a, T - 1), (a, b), "calibration:occluded", 0.0)

    hd = _hip_ankle_dy(P)
    ok = np.ones(T, bool) if ok_mask is None else np.asarray(ok_mask, bool)
    valid = np.where(ok[a:b] & np.isfinite(hd[a:b]))[0] + a
    if len(valid) < max(2, (b - a) // 2):
        idx = int(valid[len(valid) // 2]) if len(valid) else min(a, T - 1)
        return _measure(P, idx, (a, b), "calibration:occluded", (b - a) / fps)

    g = lambda n: P[:, L.IDX[n]]                                    # noqa: E731
    sh_mid_y = np.nanmean([g("Left Shoulder")[:, 1], g("Right Shoulder")[:, 1]], axis=0)
    ank_mid_y = np.nanmean([g("Left Ankle")[:, 1], g("Right Ankle")[:, 1]], axis=0)
    span = float(np.nanmedian(np.abs(sh_mid_y - ank_mid_y)))
    tol = motion_tol_ratio * span if np.isfinite(span) and span > 1e-6 else np.inf

    seg = hd[valid]
    steady = float(seg.max() - seg.min()) <= tol
    idx = int(valid[len(valid) // 2])       # 구간 중앙 — 들어가고 나가는 양 끝을 피한다
    return _measure(P, idx, (a, b),
                    "calibration" if steady else "calibration:unsteady", (b - a) / fps)


def from_keyframes(P, ok_mask=None, early_fraction=EARLY_FRACTION):
    """희소 키프레임 입력(AI-Hub 등): 시퀀스 초반에서 골반이 가장 높은 프레임.

    3D 경로가 '무릎각 최대' 로 고르는 것을 '골반 높이 최대' 로 바꾼 것이다.
    골반 높이는 y 만 쓰므로 정면에서도 성립한다.
    """
    hd = _hip_ankle_dy(P)
    T = len(hd)
    ok = np.ones(T, bool) if ok_mask is None else np.asarray(ok_mask, bool)
    head = max(5, int(T * early_fraction))
    cand = np.where(ok[:head] & np.isfinite(hd[:head]))[0]
    if len(cand) == 0:
        cand = np.where(ok & np.isfinite(hd))[0]
    if len(cand) == 0:
        cand = np.arange(T)
    i = int(cand[np.nanargmax(hd[cand])])
    return _measure(P, i, (i, i + 1), "keyframes")


def from_continuous(P, fps, ok_mask=None, stand_seconds=STAND_SECONDS,
                    near_top_ratio=NEAR_TOP_RATIO, motion_tol_ratio=MOTION_TOL_RATIO):
    """**fallback 경로** — 캘리브레이션 구간을 못 받았을 때만 쓴다.

    골반이 최고 높이 근처에서 `stand_seconds` 이상 머문 구간을 영상 전체에서 **추정**한다.
    정상 경로는 from_calibration_segment() 다. 추정은 분모를 불안정하게 만든다.
    구간을 못 찾으면 요구 길이를 줄여가며 재시도하고, 끝내 없으면 method 에 남긴다
    — 호출부가 '기준자세를 잡지 못했다' 고 사용자에게 알릴 수 있어야 한다.

    반환된 구간의 **중앙 프레임**을 대표로 쓴다 (양 끝은 들어가고 나가는 구간이라 불안정).
    """
    if not fps or fps <= 0:
        raise ValueError("연속영상 경로에는 fps 가 필요하다. 희소 입력이면 from_keyframes 를 쓴다.")
    hd = _hip_ankle_dy(P)
    T = len(hd)
    ok = np.ones(T, bool) if ok_mask is None else np.asarray(ok_mask, bool)
    valid = ok & np.isfinite(hd)
    if valid.sum() == 0:
        return _measure(P, 0, (0, 1), "continuous:no_valid_frame")

    top = float(np.nanmax(hd[valid]))
    g = lambda n: P[:, L.IDX[n]]                                    # noqa: E731
    sh_mid_y = np.nanmean([g("Left Shoulder")[:, 1], g("Right Shoulder")[:, 1]], axis=0)
    ank_mid_y = np.nanmean([g("Left Ankle")[:, 1], g("Right Ankle")[:, 1]], axis=0)
    span = float(np.nanmedian(np.abs(sh_mid_y - ank_mid_y)))
    tol = motion_tol_ratio * span if np.isfinite(span) and span > 1e-6 else np.inf

    near = valid & (hd >= near_top_ratio * top)
    for req_sec in (stand_seconds, stand_seconds * 0.75, stand_seconds * 0.5):
        need = max(2, int(round(req_sec * fps)))
        best = None
        s = None
        for t in range(T + 1):
            if t < T and near[t]:
                if s is None:
                    s = t
                continue
            if s is not None:
                seg = hd[s:t]
                if (t - s) >= need and np.isfinite(seg).all() and (seg.max() - seg.min()) <= tol:
                    if best is None or (t - s) > (best[1] - best[0]):
                        best = (s, t)
                s = None
        if best is not None:
            mid = int((best[0] + best[1] - 1) // 2)
            return _measure(P, mid, best, "continuous", (best[1] - best[0]) / fps)

    # 끝내 못 찾음 — 가장 높은 프레임으로 대체하고 사실을 남긴다
    i = int(np.nanargmax(np.where(valid, hd, -np.inf)))
    return _measure(P, i, (i, i + 1), "continuous:fallback_single_frame", 0.0)


def _measure_window(P, window, method, seconds, reliability):
    """창 안 프레임별 값의 **중앙값**으로 기준자세를 만든다 (run43).

    단일 프레임은 MediaPipe 떨림을 그대로 분모에 싣는다 — OpenCap 정면 서기 창 안에서
    무릎폭이 프레임마다 3~20 % 흔들렸다. 대표 index 는 창 중앙 프레임.
    """
    a, b = window
    g = lambda n: P[a:b, L.IDX[n]]                                  # noqa: E731
    ls, rs = g("Left Shoulder"), g("Right Shoulder")
    lh, rh = g("Left Hip"), g("Right Hip")
    lk, rk = g("Left Knee"), g("Right Knee")
    la, ra = g("Left Ankle"), g("Right Ankle")
    med = lambda v: float(np.nanmedian(v))                          # noqa: E731
    sh_mid_y = np.nanmean([ls[:, 1], rs[:, 1]], axis=0)
    hip_mid_y = np.nanmean([lh[:, 1], rh[:, 1]], axis=0)
    ank_mid_y = np.nanmean([la[:, 1], ra[:, 1]], axis=0)
    span = med(np.abs(sh_mid_y - ank_mid_y))
    return StandingReference(
        index=int((a + b - 1) // 2), window=(int(a), int(b)),
        hip_ankle_dy=med(hip_mid_y - ank_mid_y),
        knee_w=med(np.abs(rk[:, 0] - lk[:, 0])),
        ankle_w=med(np.abs(ra[:, 0] - la[:, 0])),
        shoulder_w=med(np.abs(rs[:, 0] - ls[:, 0])),
        span_sh_ank=span if np.isfinite(span) and span > 1e-6 else float("nan"),
        trunk_span=med(np.abs(sh_mid_y - hip_mid_y)),
        method=method, seconds_held=seconds, reliability=reliability)


def search_standing(P, fps, bad_mask=None, seconds=SEARCH_SECONDS,
                    top_percentile=SEARCH_TOP_PERCENTILE, near_top_ratio=NEAR_TOP_RATIO,
                    motion_tol_ratio=MOTION_TOL_RATIO, ankle_tol_ratio=ANKLE_TOL_RATIO):
    """run43 — 영상 전체에서 '가만히 서 있는 구간' 을 시간 단위로 찾는다.

    조건 (창 전체가 만족해야 함)
      · 골반 높이 ≥ near_top_ratio × (골반 높이 98 백분위)
      · 창 안 골반 높이 변동 ≤ motion_tol_ratio × span
      · 창 안 발목폭 변동 ≤ ankle_tol_ratio × span   (걷는 중이면 발목폭이 크게 바뀐다)
      · bad_mask(긴 미검출·튐 프레임)가 창에 없음
    seconds 의 긴 창부터 시도하고, 같은 길이에서는 **가장 이른 창**을 쓴다
    (세트 시작 전 자기 상태가 기준 — 지친 뒤의 서기보다 앞선다).
    값은 창 안 중앙값. 끝내 못 찾으면 reliability='failed' 로 남긴다 (호출부가 경고).
    """
    if not fps or fps <= 0:
        raise ValueError("search_standing 에는 fps 가 필요하다.")
    hd = _hip_ankle_dy(P)
    T = len(hd)
    bad = np.zeros(T, bool) if bad_mask is None else np.asarray(bad_mask, bool)
    valid = np.isfinite(hd) & ~bad
    if valid.sum() == 0:
        return _with_rel(_measure(P, 0, (0, 1), "search:no_valid_frame"), "failed")
    g = lambda n: P[:, L.IDX[n]]                                    # noqa: E731
    sh_mid_y = np.nanmean([g("Left Shoulder")[:, 1], g("Right Shoulder")[:, 1]], axis=0)
    ank_mid_y = np.nanmean([g("Left Ankle")[:, 1], g("Right Ankle")[:, 1]], axis=0)
    span = float(np.nanmedian(np.abs(sh_mid_y - ank_mid_y)))
    if not (np.isfinite(span) and span > 1e-6):
        return _with_rel(_measure(P, 0, (0, 1), "search:no_span"), "failed")
    aw = np.abs(g("Right Ankle")[:, 0] - g("Left Ankle")[:, 0])
    top = float(np.nanpercentile(hd[valid], top_percentile))
    ok = valid & (hd >= near_top_ratio * top) & np.isfinite(aw)
    tol_h, tol_a = motion_tol_ratio * span, ankle_tol_ratio * span
    for sec in seconds:
        w = max(2, int(round(sec * fps)))
        if w > T:
            continue
        # 창 전체가 ok 인 시작점만 본다
        c = np.r_[0, np.cumsum(ok.astype(int))]
        starts = np.flatnonzero(c[w:] - c[:-w] == w)
        for i in starts:
            seg_h, seg_a = hd[i:i + w], aw[i:i + w]
            if np.ptp(seg_h) <= tol_h and np.ptp(seg_a) <= tol_a:
                rel = SEARCH_RELIABILITY.get(sec, "low")
                return _measure_window(P, (i, i + w), f"search:{sec:g}s", w / fps, rel)
    i = int(np.nanargmax(np.where(valid, hd, -np.inf)))
    return _with_rel(_measure(P, i, (i, i + 1), "search:not_found", 0.0), "failed")


def _with_rel(ref, rel):
    from dataclasses import replace
    return replace(ref, reliability=rel)
