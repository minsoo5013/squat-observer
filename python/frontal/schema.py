# -*- coding: utf-8 -*-
"""schema.py — 최종 출력 스키마. 흩어진 값을 **5개 영역**으로 묶기만 한다.

  무릎 정렬 Knee alignment · 스쿼트 깊이 Squat depth · 상체 안정성 Trunk stability
  · 반복 일관성 Rep consistency   + 촬영 품질 Capture QC

판정하지 않는다. judgement 는 모든 층에서 None 이다.
계산을 새로 하지 않는다 — engine 이 이미 만든 값을 재배치할 뿐이다.

각 값에는 **지금 어디까지 쓸 수 있는지**를 level 로 붙인다.
  service_now  지금 정면 키포인트만 있으면 바로 낼 수 있다 (AI-Hub XY·view3 로 확인됨)
  needs_mp4    실제 30fps 영상이 들어와야 의미가 생긴다 (시간축·촬영 QC 합격선)
  exploratory  아직 연구용이다. 사용자 화면에 숫자로 내보내지 않는다.
"""
from __future__ import annotations

SCHEMA_VERSION = "5axis-v1"

SERVICE_NOW, NEEDS_MP4, EXPLORATORY = "service_now", "needs_mp4", "exploratory"

AREAS = ("knee_alignment", "squat_depth", "trunk_stability", "rep_consistency", "capture_qc")
AREA_TITLE = {
    "knee_alignment": "무릎 정렬",
    "squat_depth": "스쿼트 깊이",
    "trunk_stability": "상체 안정성",
    "rep_consistency": "반복 일관성",
    "capture_qc": "촬영 품질",
}

# key → (영역, level, 설명)
METRIC_SPEC = {
    "A2_knee_w_rel_stand":  ("knee_alignment", SERVICE_NOW,  "기준자세 대비 무릎폭 (1보다 작을수록 모임)"),
    "A1_knee_ankle_w":      ("knee_alignment", SERVICE_NOW,  "무릎폭 ÷ 발목폭"),
    "A45_knee_in_mean":     ("knee_alignment", SERVICE_NOW,  "기준자세 대비 무릎 안쪽 이동 (좌우 평균)"),
    "A4_knee_in_L":         ("knee_alignment", SERVICE_NOW,  "왼무릎 안쪽 이동 — mirror 확정 시에만"),
    "A5_knee_in_R":         ("knee_alignment", SERVICE_NOW,  "오른무릎 안쪽 이동 — mirror 확정 시에만"),
    "A6_knee_in_asym":      ("knee_alignment", EXPLORATORY,  "좌우 비대칭 — view3 재현성 ρ=0.550, 아직 사용자에게 내지 않는다"),
    "A6c1_norm_asym":       ("knee_alignment", EXPLORATORY,  "A6 후보① 정규화 비대칭"),
    "A6c2_baseline_corrected": ("knee_alignment", EXPLORATORY, "A6 후보② 기준자세·흔들림 보정 비대칭"),
    "A6c3_across_rep":      ("knee_alignment", EXPLORATORY,  "A6 후보③ 반복 간 부호 일관 비대칭"),
    "A6c4_bc_across_rep":   ("knee_alignment", EXPLORATORY,  "A6 후보④ ②+③ 조합"),

    "D1_hip_ankle_rel":     ("squat_depth",    SERVICE_NOW,  "기준자세 대비 골반 높이 (낮을수록 깊음)"),
    "hip_knee_dy":          ("squat_depth",    EXPLORATORY,  "골반−무릎 높이차 ÷ 어깨~발목 — diagnostic 전용"),
    "parallel_reached":     ("squat_depth",    EXPLORATORY,  "골반이 무릎보다 낮았는가 — diagnostic 전용, 합격판정 아님"),

    "C1_trunk_span_rel":    ("trunk_stability", EXPLORATORY, "상체 apparent shortening — 어깨 의존이 있다"),
    "B3_pelvis_tilt_deg_r": ("trunk_stability", EXPLORATORY, "저점 골반 기울기(도)"),
    "B4_pelvis_tilt_rng_r": ("trunk_stability", EXPLORATORY, "세트 전체 골반 기울기 변동폭"),
    "B7_hip_drop_asym_r":   ("trunk_stability", EXPLORATORY, "좌우 골반 하강 비대칭"),
    "B8_knee_drop_asym_r":  ("trunk_stability", EXPLORATORY, "좌우 무릎 하강 비대칭"),

    "B9_ankle_x_drift":     ("capture_qc",     NEEDS_MP4,    "발 좌우 이동량 — 합격선은 실촬영 후"),
}

# 영역별로 '이 값이 무엇을 위한 것인지'를 한 줄로 — 사용자 화면 문구가 아니라 개발 기준이다
AREA_NOTE = {
    "knee_alignment": "핵심 축. A2/A1/A45 는 지금 낼 수 있다. A6 계열은 전부 연구용이다.",
    "squat_depth": "D1 이 주 지표다. hip_knee_dy·parallel_reached 는 diagnostic 이며 깊이 판정에 쓰지 않는다.",
    "trunk_stability": "대응 라벨이 없어 AUC 로 걸러내지 않았다. 전부 연구용으로 보존 중이다.",
    "rep_consistency": "이미 계산해 놓고 버리던 per_rep 을 살린 층. 세트 내 흩어짐·전후반 변화·순서 추세.",
    "capture_qc": "입력이 쓸 만한지 보는 층. 합격선은 실촬영 데이터로 정한다.",
}


def _put(areas, key, value):
    spec = METRIC_SPEC.get(key)
    if spec is None:
        return
    area, level, desc = spec
    areas[area]["metrics"][key] = {"value": value, "level": level, "desc": desc}


def build(result):
    """engine.analyze_frontal 의 결과 dict → 5영역 스키마. 값을 새로 만들지 않는다."""
    areas = {a: {"title": AREA_TITLE[a], "note": AREA_NOTE[a],
                 "metrics": {}, "judgement": None} for a in AREAS}

    tiers = result.get("features") or {}
    for tier in ("primary", "dynamic", "exploratory", "auxiliary", "qc_signals", "side_specific"):
        for k, v in (tiers.get(tier) or {}).items():
            _put(areas, k, v)

    dd = result.get("depth_diagnostics") or {}
    for k in ("hip_knee_dy", "parallel_reached"):
        if k in dd:
            _put(areas, k, dd[k])
    if dd:
        areas["squat_depth"]["diagnostic_detail"] = {
            k: dd[k] for k in ("per_rep_hip_knee_dy", "parallel_rep_fraction",
                               "numerically_comparable_to_3d", "usage", "not_for", "note")
            if k in dd}

    a6 = result.get("a6_candidates") or {}
    for k in ("A6c1_norm_asym", "A6c2_baseline_corrected",
              "A6c3_across_rep", "A6c4_bc_across_rep"):
        if k in a6:
            _put(areas, k, a6[k])
    if a6:
        areas["knee_alignment"]["a6_candidate_detail"] = {
            k: a6[k] for k in ("sign_agreement_base", "sign_agreement_bc",
                               "median_over_mad_base", "median_over_mad_bc",
                               "A6_recomputed_reference", "reference_max_abs_diff_vs_engine",
                               "note") if k in a6}

    rc = result.get("rep_consistency")
    if rc:
        areas["rep_consistency"]["per_key"] = rc["per_key"]
        areas["rep_consistency"]["rep_deviation_score"] = rc["rep_deviation_score"]
        areas["rep_consistency"]["rep_rank_by_deviation"] = rc["rep_rank_by_deviation"]
        areas["rep_consistency"]["level"] = SERVICE_NOW
        areas["rep_consistency"]["consistency_note"] = rc["note"]

    rh = result.get("rhythm")
    if rh:
        areas["rep_consistency"]["rhythm"] = rh
        areas["rep_consistency"]["rhythm_level"] = NEEDS_MP4

    qc = result.get("capture_qc") or {}
    areas["capture_qc"].update({
        "signals": qc,
        "level": NEEDS_MP4,
        "cutoffs_defined": False,
    })

    return {
        "schema_version": SCHEMA_VERSION,
        "judgement": None,
        "n_reps": result.get("n_reps"),
        "bottoms": result.get("bottoms"),
        "source": result.get("source"),
        "side_labels_usable": (result.get("capture_qc") or {}).get("side_labels_usable"),
        "side_note": result.get("side_note"),
        "areas": areas,
        "levels": {"service_now": SERVICE_NOW, "needs_mp4": NEEDS_MP4,
                   "exploratory": EXPLORATORY},
        "note": ("5영역 재배치 전용. 판정·threshold 없음. "
                 "level=exploratory 인 값은 사용자 화면에 숫자로 내보내지 않는다."),
    }
