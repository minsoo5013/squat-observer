# -*- coding: utf-8 -*-
"""features.py — 정면 지표 계산층. **판정하지 않는다. threshold 가 없다.**

계층 (Track 1 선별 결과 기준)
  PRIMARY      A2 개인 기준자세 대비 무릎폭  · A1 무릎-발목 기하 비율
  DYNAMIC      A45 기준자세 대비 무릎 안쪽 이동 (사용자 설명용)
  EXPLORATORY  A6 좌우 비대칭 · B3/B4/B7/B8 골반·좌우 안정성 · C1 상체 apparent shortening
  AUXILIARY    D1 상대 깊이
  QC           가시성 · 전신 프레임 점유 · 발 보임 · B9 발 이동 · mirror · roll

왜 이 구성인가 (dev42+val70, 106세트 · participant-aware LOPO):
  A 군 9개 지표는 PC1 83.3% + PC2 10.0% 의 두 축뿐이고, PC2 는 A6 가 단독(적재 -0.99)이다.
  A2 단독 LOPO 0.969 · A2+A45 0.971 · A 군 9개 전부 0.912(최악) → 최소 조합이 맞다.
  A1 은 참가자 내부에서 A2 보다 오히려 낫고(0.925 vs 0.918) 실패 방식이 달라 함께 둔다.
  B/C 군은 대응 라벨이 없어 AUC 로 제거하지 않았다 — exploratory 로 분리 보존한다.

어깨(11,12) 의존도 — 112세트 실측 (frontal/check_shoulder_dependence.py)
  어깨를 떨게 하거나(span 5%), 아래로 치우치게 하거나, 아예 지워도:
    A1 · A2 · A45 · A6 · D1 · B3_r · B4_r · 저점검출 · 반복수  → **변화 0.0000 (bit 동일)**
    C1_trunk_span_rel                                        → 0.043~0.054 (자체 IQR 의 30~40%)
    B7_r · B8_r                                              → 잡음에는 둔감(0.0002)하나 어깨가 없으면 **계산 불가**
                                                                (span = 어깨~발목 을 분모로 쓰기 때문)
    capture_qc 의 frame_fill_ratio                            → 같은 이유로 계산 불가
  => 팔 자세·바벨 가림의 위험은 **C1 과 span 을 분모로 쓰는 것들에 한정**된다.
     핵심 판정 지표는 준비자세의 팔 위치와 무관하다.
     (B7/B8 의 어깨 의존은 분모 때문이므로, 필요하면 골반~발목 분모로 바꾸면 사라진다 — 아직 바꾸지 않았다.)

mirror
  아래 MIRROR_SAFE 에 든 지표는 좌우반전과 무관하다.
  그 밖(A4/A5 의 좌우 배정, 사용자에게 말하는 '왼쪽/오른쪽')은 mirror 가 확정된 경우에만 낸다.
"""
from __future__ import annotations

import numpy as np

from frontal import landmarks as L

PRIMARY = ["A2_knee_w_rel_stand", "A1_knee_ankle_w"]
DYNAMIC = ["A45_knee_in_mean"]
SIDE_SPECIFIC = ["A4_knee_in_L", "A5_knee_in_R"]
EXPLORATORY = ["A6_knee_in_asym", "B3_pelvis_tilt_deg_r", "B4_pelvis_tilt_rng_r",
               "B7_hip_drop_asym_r", "B8_knee_drop_asym_r", "C1_trunk_span_rel"]
AUXILIARY = ["D1_hip_ankle_rel"]
QC_SIGNALS = ["B9_ankle_x_drift"]

# 좌우반전에 영향받지 않는 지표 — 폭·절댓값·평균만 쓰기 때문
MIRROR_SAFE = set(PRIMARY + DYNAMIC + EXPLORATORY + AUXILIARY + QC_SIGNALS)

# 촬영 거리 — **확정 조건이 아니다.** 합성 perturbation 관찰값에서 나온 초기 가이드 후보일 뿐이며
# 사람 키·렌즈 화각·세로/가로 촬영에 따라 달라진다. 실제 QC 는 아래 framing 지표로 한다.
PROVISIONAL_DISTANCE_GUIDE_M = (2.5, 4.0)


def _safe(a, b):
    b = float(b)
    return float(a) / b if np.isfinite(b) and abs(b) > 1e-9 else float("nan")


def _tilt_deg(dy, dx):
    if not (np.isfinite(dy) and np.isfinite(dx)) or abs(dx) < 1e-9:
        return float("nan")
    return float(np.degrees(np.arctan2(dy, abs(dx))))


def _rng(v):
    v = np.asarray(v, float)
    return float(np.nanmax(v) - np.nanmin(v)) if np.isfinite(v).any() else float("nan")


def _at(P, t, n):
    return P[t, L.IDX[n]]


def rep_features(P, Pr, bi, ref, si):
    """저점 프레임 하나에 대한 지표. Pr 은 발목선 roll 보정된 좌표."""
    ls, rs = _at(P, bi, "Left Shoulder"), _at(P, bi, "Right Shoulder")
    lh, rh = _at(P, bi, "Left Hip"), _at(P, bi, "Right Hip")
    lk, rk = _at(P, bi, "Left Knee"), _at(P, bi, "Right Knee")
    la, ra = _at(P, bi, "Left Ankle"), _at(P, bi, "Right Ankle")
    lk_s, rk_s = _at(P, si, "Left Knee"), _at(P, si, "Right Knee")
    la_s, ra_s = _at(P, si, "Left Ankle"), _at(P, si, "Right Ankle")

    knee_w_b = abs(float(rk[0] - lk[0]))
    ank_w_b = abs(float(ra[0] - la[0]))
    hip_mid = np.nanmean([lh, rh], axis=0)
    sh_mid = np.nanmean([ls, rs], axis=0)
    ank_mid_y = float(np.nanmean([la[1], ra[1]]))
    mid_x_s = float(np.nanmean([la_s[0], ra_s[0]]))
    inw_L = float(np.sign(mid_x_s - lk_s[0])) or 1.0
    inw_R = float(np.sign(mid_x_s - rk_s[0])) or -1.0

    m = {}
    # PRIMARY
    m["A2_knee_w_rel_stand"] = _safe(knee_w_b, ref.knee_w)
    m["A1_knee_ankle_w"] = _safe(knee_w_b, ank_w_b)
    # DYNAMIC / SIDE
    a4 = _safe((float(lk[0]) - float(lk_s[0])) * inw_L, ref.ankle_w)
    a5 = _safe((float(rk[0]) - float(rk_s[0])) * inw_R, ref.ankle_w)
    m["A4_knee_in_L"] = a4
    m["A5_knee_in_R"] = a5
    m["A45_knee_in_mean"] = float(np.nanmean([a4, a5]))
    m["A6_knee_in_asym"] = abs(a4 - a5)
    # AUXILIARY
    m["D1_hip_ankle_rel"] = _safe(float(hip_mid[1]) - ank_mid_y, ref.hip_ankle_dy)
    # EXPLORATORY — 상체
    m["C1_trunk_span_rel"] = _safe(abs(float(sh_mid[1] - hip_mid[1])), ref.trunk_span)
    # EXPLORATORY — 좌우 (발목선 보정 좌표에서 계산)
    lhr, rhr = _at(Pr, bi, "Left Hip"), _at(Pr, bi, "Right Hip")
    lkr, rkr = _at(Pr, bi, "Left Knee"), _at(Pr, bi, "Right Knee")
    lhs, rhs = _at(Pr, si, "Left Hip"), _at(Pr, si, "Right Hip")
    lks, rks = _at(Pr, si, "Left Knee"), _at(Pr, si, "Right Knee")
    span = ref.span_sh_ank
    m["B3_pelvis_tilt_deg_r"] = abs(_tilt_deg(float(rhr[1] - lhr[1]), float(rhr[0] - lhr[0])))
    m["B7_hip_drop_asym_r"] = _safe(
        abs((float(lhs[1]) - float(lhr[1])) - (float(rhs[1]) - float(rhr[1]))), span)
    m["B8_knee_drop_asym_r"] = _safe(
        abs((float(lks[1]) - float(lkr[1])) - (float(rks[1]) - float(rkr[1]))), span)
    return m


def set_features(P, Pr, ref):
    """세트 전체 프레임이 필요한 지표."""
    g = lambda Q, n: Q[:, L.IDX[n]]                                  # noqa: E731
    lhr, rhr = g(Pr, "Left Hip"), g(Pr, "Right Hip")
    la, ra = g(P, "Left Ankle"), g(P, "Right Ankle")
    p_tilt = np.degrees(np.arctan2(rhr[:, 1] - lhr[:, 1], np.abs(rhr[:, 0] - lhr[:, 0])))
    ank_mid_x = np.nanmean([la[:, 0], ra[:, 0]], axis=0)
    return {
        "B4_pelvis_tilt_rng_r": _rng(p_tilt),
        "B9_ankle_x_drift": _safe(_rng(ank_mid_x), ref.ankle_w),
    }


def capture_qc(seq, ref, min_visibility=None):
    """촬영 품질 — **판정이 아니라 입력이 쓸 만한지**를 본다. 합격선은 정하지 않는다.

    거리(m)를 기준으로 삼지 않는다. 사람 키·화각·세로/가로에 따라 달라지기 때문이다.
    대신 화면에서 몸이 차지하는 비율과 발이 보이는지를 본다 — 그게 실제로 필요한 조건이다.
    """
    V = seq.visibility
    H = seq.image_size[1] if seq.image_size else None
    need = ["Left Shoulder", "Right Shoulder", "Left Hip", "Right Hip",
            "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"]
    vis = {n: float(np.nanmedian(V[:, L.IDX[n]])) for n in need}
    foot = {n: float(np.nanmedian(V[:, L.IDX[n]]))
            for n in ("Left Heel", "Right Heel") if n in L.IDX}

    # 전신 화면 점유율 — 어깨~발목 세로길이가 화면 높이의 몇 %인가
    frame_fill = _safe(ref.span_sh_ank, H) if H else float("nan")

    # 프레임 밖으로 나간 관절 (픽셀 좌표는 y 를 뒤집어 -H..0 범위)
    out_of_frame = {}
    if seq.image_size:
        W, Ht = seq.image_size
        for n in need:
            p = seq.P[:, L.IDX[n]]
            bad = np.isfinite(p[:, 0]) & ((p[:, 0] < 0) | (p[:, 0] > W)
                                          | (p[:, 1] > 0) | (p[:, 1] < -Ht))
            if bad.any():
                out_of_frame[n] = int(bad.sum())

    return {
        "median_visibility": vis,
        "foot_visibility": foot,
        "frame_fill_ratio": frame_fill,      # 전신 화면 점유 — 1차 QC
        "out_of_frame_frames": out_of_frame,  # 전신 포함 여부
        "mirror_known": seq.mirrored is not None,
        "side_labels_usable": seq.side_labels_usable,
        "roll_corrected": True,
        "standing_method": ref.method,
        "standing_seconds_held": ref.seconds_held,
        "note": ("합격/불합격 선은 아직 없다. 실촬영 데이터로 정한다. "
                 f"촬영 거리 {PROVISIONAL_DISTANCE_GUIDE_M[0]}~{PROVISIONAL_DISTANCE_GUIDE_M[1]}m 는 "
                 "확정 조건이 아니라 초기 가이드 후보다."),
    }
