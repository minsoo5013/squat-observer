# -*- coding: utf-8 -*-
"""signals.py — 정면 반복 검출.

squat_core 에서 가져다 쓰는 것은 **좌표를 받지 않는 1차원 신호 처리 함수뿐**이다.
좌표를 받는 함수(extract_features / capture_standing_reference / detect_bottoms /
compute_gate_features / run_sentinel_qc …)는 z 를 전제하므로 정면 트랙에서 부르지 않는다.

정면 전환에서 무너진 것은 저점 '신호'가 아니라 각도 기반 **유효반복 필터**였다.
  dev42 · 라벨 저점 153개 기준 (exploration/filter_replacement.py)
    현행 3D 각도 < 130°              ±1 일치 100.0%   반복수 정확 41/42
    정면 XY 에 같은 각도식 그대로      ±1 일치  40.5%   반복수 정확 15/42
    정면 XY 골반높이비 < 0.90         ±1 일치 100.0%   반복수 정확 40/42
그래서 필터를 무차원 수직변위(기준자세 대비 골반 높이비)로 바꾼다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import squat_core as sc              # noqa: E402
from frontal import landmarks as L            # noqa: E402

# 정면 트랙이 squat_core 에서 호출해도 되는 것 — 두 종류뿐이다.
#  (a) 좌표를 받지 않는 1차원 신호 처리 — 서비스 경로에서도 쓴다
ALLOWED_SIGNAL_FUNCS = ("smooth", "zscore_arr", "_estimate_K_refined",
                        "_detect_valley_first", "BEST_RULE")
#  (b) AI-Hub JSON 을 읽기 위한 순수 접근자 — **검증 경로(from_aihub_3d)에서만** 쓰고,
#      읽자마자 z 를 버린다. MediaPipe 서비스 경로에서는 호출되지 않는다.
ALLOWED_LOADER_FUNCS = ("load_input_frames", "get_frame_pts", "get_xyz")
ALLOWED_CORE_FUNCS = ALLOWED_SIGNAL_FUNCS + ALLOWED_LOADER_FUNCS

# ── 잠정 파라미터 ────────────────────────────────────────────
# 확정 임계값이 아니다. 위 실험에서 나온 후보값이며 실촬영 30fps 로 재측정해야 한다.
REP_FILTER_REL = 0.90        # 기준자세 대비 골반높이비가 이보다 낮아야 '유효 반복'
ADJ_MERGE_KEYFRAME = 2       # 인접 저점 병합 간격 — **프레임 수 의존**. 16키프레임 기준값 (키프레임 경로 전용)

# ── 연속 영상 경로 (run 43, 2026-09-27) ─────────────────────
# 이전: 키프레임용 K 구간 방식(squat_core._estimate_K_refined → 반복 수 상한 6, 구간마다 저점 1개 강제)을
#       연속 영상에도 써서 7회 이상이 6회로 잘렸다 (ISSUES I121·I123, OpenCap 이어붙임 10·15회 → 6 재현).
# 지금: 연속 영상은 '골반높이비 < REP_FILTER_REL 인 구간 = 반복 1회' 로 센다. 상한 없음. 값은 **초 단위**.
#   개발 데이터(OpenCap 9명 × 5카메라, squats1) 분포에서 정했다 (run 43 PLAN.md):
#     실제 반복 구간 길이 ≥ 0.5 s, 반복 사이 서기 구간 ≥ 0.4 s  /  문턱 근처 잡음 구간 ≤ 0.1 s, 잡음 틈 ≤ 0.15 s
#   → 두 분포 사이 빈 구간의 값으로 고정. run 41(holdout)·외부 11개는 보지 않았다.
#   병합 규칙(v1.1, 같은 날 수정): 두 구간 사이에서 골반높이비가 REP_EXIT_REL 이상으로 '다시 일어서지' 않았으면
#   같은 반복이다 (히스테리시스 0.90 진입 / 0.95 이탈). 시간 병합은 짧은 튐만 막는 보조(MERGE_GAP_S).
#   이유: 시간 병합만 0.25 s 로 두면 반복 사이 서기가 0.2 s 인 빠른 연속 반복(verify_pipeline 합성, 완전히 일어섬)이
#   1회로 합쳐졌다. OpenCap 49건(45 + 3배 이어붙임, 기준자세 찾은 것)은 네 변형 모두 49/49 (eval_merge_variants.json).
REP_EXIT_REL = 0.95          # 반복 사이에 이 높이 이상 올라와야 '다음 반복' 으로 센다
MERGE_GAP_S = 0.10           # 이보다 짧은 틈은 높이와 상관없이 같은 반복 (튀는 프레임)
MIN_REP_S = 0.25             # 이보다 짧은 구간은 반복이 아니다 (잡음)


def hip_height_ratio(P, standing_hip_ankle_dy):
    """기준자세 대비 골반 높이비. 1 = 직립, 낮을수록 깊게 앉음. y 만 쓴다."""
    g = lambda n: P[:, L.IDX[n]]                                    # noqa: E731
    hip = np.nanmean([g("Left Hip")[:, 1], g("Right Hip")[:, 1]], axis=0)
    ank = np.nanmean([g("Left Ankle")[:, 1], g("Right Ankle")[:, 1]], axis=0)
    d = float(standing_hip_ankle_dy)
    if not np.isfinite(d) or abs(d) < 1e-9:
        return np.full(len(hip), np.nan)
    return (hip - ank) / d


def build_hip_signal(P, smooth_window=None):
    """저점 검출용 신호 = 골반 하강량. squat_core 의 sig_hip 과 같은 정의다.

    sig_knee / sig_multi 는 쓰지 않는다 —
      sig_knee 는 3D 무릎각이라 정면 투영에서 무너지고,
      sig_multi 는 Back 관절이 필요한데 MediaPipe 에 없다.
    """
    if smooth_window is None:
        smooth_window = sc.BEST_RULE["smooth_window"]
    g = lambda n: P[:, L.IDX[n]]                                    # noqa: E731
    hip = np.nanmean([g("Left Hip")[:, 1], g("Right Hip")[:, 1]], axis=0)
    ank = np.nanmean([g("Left Ankle")[:, 1], g("Right Ankle")[:, 1]], axis=0)
    hip_inv = sc.smooth(-np.abs(hip - ank), smooth_window, 3)
    return sc.zscore_arr(hip_inv)


def detect_bottoms(P, standing_ref, fps=None, adj_merge=None,
                   rep_filter_rel=REP_FILTER_REL, smooth_window=None):
    """정면 저점 검출. 반환: (저점 인덱스 리스트, 진단 dict)

    fps 가 있으면(연속 영상) detect_bottoms_continuous 로 간다 — 아래 키프레임 코드는 fps=None 일 때만 쓴다
    (AI-Hub 키프레임 검증 경로, 결과 비트 동일 유지).

    adj_merge 는 **프레임 수에 의존**하므로 비례 환산하지 않는다.
      · 희소 키프레임 입력(fps=None)  → ADJ_MERGE_KEYFRAME(=2) 사용
      · 연속 영상(fps 지정)           → 호출부가 명시해야 한다. 없으면 진단에 경고를 남긴다.
    """
    if fps:
        return detect_bottoms_continuous(P, standing_ref, fps, rep_filter_rel=rep_filter_rel,
                                         exit_rel=REP_EXIT_REL)
    warnings = []
    if adj_merge is None:
        if fps:
            adj_merge = ADJ_MERGE_KEYFRAME
            warnings.append(
                "adj_merge 를 지정하지 않아 키프레임 기준값(2)을 썼다. "
                "연속 영상에서는 실측으로 정해야 한다 — 비례 환산 금지.")
        else:
            adj_merge = ADJ_MERGE_KEYFRAME

    sig = build_hip_signal(P, smooth_window)
    rel = hip_height_ratio(P, standing_ref.hip_ankle_dy)

    K = sc._estimate_K_refined(sig)
    det = sc._detect_valley_first(sig, K, sc.BEST_RULE["trough_distance_ratio"],
                                  sc.BEST_RULE["trough_prominence_factor"])
    raw = sorted(int(b) for b in det["pred"].tolist())

    kept = [b for b in raw if np.isfinite(rel[b]) and rel[b] < rep_filter_rel]
    merged = []
    for b in kept:
        if merged and (b - merged[-1]) < adj_merge:
            if rel[b] < rel[merged[-1]]:
                merged[-1] = b
        else:
            merged.append(b)

    return merged, {
        "signal": "hip_descent (sig_hip 동일 정의)",
        "rep_filter": f"기준자세 대비 골반높이비 < {rep_filter_rel} (잠정)",
        "adj_merge": int(adj_merge),
        "n_raw": len(raw), "n_kept": len(kept), "n_final": len(merged),
        "dropped_by_filter": len(raw) - len(kept),
        "rel_at_bottoms": [round(float(rel[b]), 3) for b in merged],
        "warnings": warnings,
    }


def detect_bottoms_continuous(P, standing_ref, fps, rep_filter_rel=REP_FILTER_REL,
                              merge_gap_s=MERGE_GAP_S, min_rep_s=MIN_REP_S,
                              exit_rel=None):
    """연속 영상 반복 검출. 반복 1회 = 기준자세 대비 골반높이비가 rep_filter_rel 아래로 내려간 구간.
    저점 = 그 구간에서 골반높이비가 가장 낮은 프레임 (키프레임 경로의 sig_hip 최댓값과 같은 정의).
    반복 수 상한·구간 수 강제 없음. 시간 값은 초 단위 → 프레임 수는 fps 로만 환산한다.
    """
    rel = hip_height_ratio(P, standing_ref.hip_ankle_dy)
    below = np.isfinite(rel) & (rel < rep_filter_rel)
    edges = np.flatnonzero(np.diff(np.r_[0, below.astype(int), 0]))
    runs = [[int(a), int(b)] for a, b in zip(edges[::2], edges[1::2])]      # [start, end)
    n_raw = len(runs)
    gap_f = max(1, int(round(merge_gap_s * fps)))
    merged = []
    for r in runs:
        stood_up = True
        if merged and exit_rel is not None:
            gseg = rel[merged[-1][1]:r[0]]
            gseg = gseg[np.isfinite(gseg)]
            stood_up = bool(gseg.size) and float(gseg.max()) >= exit_rel
        if merged and (r[0] - merged[-1][1] < gap_f or not stood_up):
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    min_f = max(1, int(round(min_rep_s * fps)))
    kept = [r for r in merged if r[1] - r[0] >= min_f]
    bots = [int(a + np.nanargmin(rel[a:b])) for a, b in kept]
    warnings = []
    if not np.isfinite(standing_ref.hip_ankle_dy) or standing_ref.hip_ankle_dy <= 0:
        warnings.append("기준자세 골반-발목 거리가 유효하지 않아 반복을 셀 수 없다.")
    return bots, {
        "signal": "hip_height_ratio (기준자세 대비 골반높이비)",
        "method": "continuous_runs_v1.1",
        "rep_filter": f"기준자세 대비 골반높이비 < {rep_filter_rel} 인 구간 = 반복 1회",
        "merge_gap_s": merge_gap_s, "min_rep_s": min_rep_s, "exit_rel": exit_rel,
        "adj_merge": None,
        "n_raw": n_raw, "n_merged": len(merged), "n_kept": len(kept), "n_final": len(bots),
        "dropped_short": len(merged) - len(kept),
        "dropped_by_filter": 0,
        "rep_windows": [[int(a), int(b)] for a, b in kept],
        "rel_at_bottoms": [round(float(rel[b]), 3) for b in bots],
        "warnings": warnings,
    }
