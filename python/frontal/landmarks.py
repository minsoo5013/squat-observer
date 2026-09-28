# -*- coding: utf-8 -*-
"""landmarks.py — MediaPipe Pose 33 landmark 매핑과 좌표 변환.

MediaPipe Pose Landmarker 출력 규격 (Google AI Edge 문서):
  pose_landmarks       x, y 는 이미지 **폭(x)·높이(y)로 각각** 0~1 정규화. z 는 비계량.
  pose_world_landmarks 미터 단위 — 단일 카메라에서는 모델 추정값이므로 **쓰지 않는다**.
  visibility           가시성 확률.

이 모듈이 책임지는 것은 딱 두 가지다.
  1) 33개 중 우리가 쓰는 12개만 골라낸다.
  2) 정규화 좌표를 **등방(isotropic) 픽셀 좌표**로 되돌리고 y 축을 위로 뒤집는다.

(2)를 건너뛰면 안 되는 이유 — 112 세트 실측 (세로형 1080x1920 가정):
  가로÷가로 / 세로÷세로 비율     오차 0.00%   (그대로 써도 값이 같다)
  각도                          오차 43.7%
  가로÷세로 혼합                 오차 77.8%
  발목선 roll 보정을 거친 지표     오차 최대 147%   ← 회전은 등방 좌표에서만 성립한다
지표마다 따지지 않고 입력 단계에서 한 번에 픽셀로 되돌린다.
"""
from __future__ import annotations

import numpy as np

# ── MediaPipe Pose 33 landmark 중 실제로 쓰는 것 ──────────────
MP_INDEX = {
    "Left Shoulder": 11, "Right Shoulder": 12,
    "Left Hip": 23, "Right Hip": 24,
    "Left Knee": 25, "Right Knee": 26,
    "Left Ankle": 27, "Right Ankle": 28,
    "Left Heel": 29, "Right Heel": 30,
    "Left Foot Index": 31, "Right Foot Index": 32,
    "Nose": 0,
}
# 미사용: 얼굴 1~10, 팔 13~22 — 바벨 스쿼트 정면 분석에 필요 없다.

# 지표 계산에 쓰는 관절 순서 (배열 인덱스)
JOINTS = ["Left Shoulder", "Right Shoulder", "Left Hip", "Right Hip",
          "Left Knee", "Right Knee", "Left Ankle", "Right Ankle",
          "Left Heel", "Right Heel", "Nose"]
IDX = {n: i for i, n in enumerate(JOINTS)}

# 반드시 있어야 하는 관절 — 없으면 분석 자체가 불가능하다
REQUIRED_JOINTS = ["Left Shoulder", "Right Shoulder", "Left Hip", "Right Hip",
                   "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"]
# 있으면 쓰고 없으면 대체하는 관절
#   Heel  : 발목선 대신 쓸 수 있는 바닥 기준선 (없으면 발목선을 쓴다)
#   Nose  : 프레이밍 참고용
OPTIONAL_JOINTS = [j for j in JOINTS if j not in REQUIRED_JOINTS]

# AI-Hub 에 있었으나 MediaPipe 에 없는 것 — 대체 정의는 features.py 참조
MISSING_VS_AIHUB = {
    "Head":  "Nose(0) 로 대체. 머리끝이 아니므로 span 정의가 달라진다",
    "Neck":  "어깨 중점으로 대체",
    "Back":  "어깨 중점으로 대체 (몸통 상단)",
    "Waist": "골반 중점으로 대체 (몸통 하단)",
}

DEFAULT_MIN_VISIBILITY = 0.5      # 프레임 채택 기준 — 잠정값, 실촬영에서 재측정

# ── 좌우반전(mirror) ─────────────────────────────────────────
# 스마트폰 전면카메라는 미리보기가 좌우반전이고, 저장 영상도 기기·앱에 따라 반전될 수 있다.
#
# 자동 판별이 왜 안 되는가:
#   MediaPipe 는 landmark 의 좌/우 정체성을 **이미지에서 학습된 겉모습**으로 정한다.
#   정면을 보고 선 대칭 자세에서는 반전된 영상도 '정상적인 사람'으로 보이므로,
#   모델은 일관되게 '거울상 라벨링'을 내놓는다. 기하만으로는 두 경우가 구분되지 않는다.
#   (nose 의 좌우 치우침 같은 단서는 yaw 가 0 에 가까울수록 사라져서 쓸 수 없다.)
# => mirror 는 **촬영 계층에서 알려주거나, 촬영 시 한 번 캘리브레이션**해야 한다.
#    알 수 없으면 좌우 구분이 필요한 출력을 억제한다. features.py 의 MIRROR_SAFE 참조.
MIRROR_UNKNOWN = None


def to_pixel_xy(frames_landmarks, width, height, min_visibility=DEFAULT_MIN_VISIBILITY):
    """MediaPipe 정규화 landmark 시퀀스 → (T, J, 2) 등방 픽셀 좌표 + (T, J) 가시성.

    frames_landmarks : 프레임별 landmark 리스트. 각 원소는 33개 landmark 로,
                       landmark 는 .x/.y/.visibility 속성 또는 같은 키의 dict.
    반환 P : x = 오른쪽 +, y = **위쪽 +** (이미지 y 는 아래로 증가하므로 뒤집는다)
             단위는 픽셀. 가시성 미달 좌표는 NaN.
    """
    T = len(frames_landmarks)
    P = np.full((T, len(JOINTS), 2), np.nan)
    V = np.zeros((T, len(JOINTS)))
    for t, lms in enumerate(frames_landmarks):
        if lms is None:
            continue
        for name, j in IDX.items():
            k = MP_INDEX[name]
            if k >= len(lms):
                continue
            lm = lms[k]
            x = lm["x"] if isinstance(lm, dict) else lm.x
            y = lm["y"] if isinstance(lm, dict) else lm.y
            vis = (lm.get("visibility", 1.0) if isinstance(lm, dict)
                   else getattr(lm, "visibility", 1.0))
            V[t, j] = float(vis)
            if vis is not None and float(vis) < min_visibility:
                continue
            # 등방 픽셀로 되돌리고 y 축을 위로 뒤집는다
            P[t, j] = (float(x) * width, -float(y) * height)
    return P, V


def fill_gaps(P):
    """미검출/가시성 미달로 생긴 NaN 프레임을 관절·축별 선형보간으로 메운다.

    왜 필요한가: 저점 검출은 1차원 신호를 보는데, 중간에 NaN 이 끼면
    core 의 valley 탐색이 All-NaN slice 로 죽는다. 3D 엔진도 같은 이유로
    extract_features 에서 interpolate(limit_direction="both") 를 한다 — 같은 처리를 한다.

    반환 (P_filled, info). info 에 몇 프레임을 메웠고 가장 긴 공백이 몇 프레임인지 남긴다.
    **보간된 프레임의 지표값은 만들어낸 값이다** — 호출부가 알 수 있게 진단에 싣는다.
    관절 하나가 전 구간 NaN 이면 메울 수 없다. info["all_nan_joints"] 로 알린다.
    """
    Q = P.copy()
    T = Q.shape[0]
    t = np.arange(T, dtype=float)
    filled_any = np.zeros(T, bool)
    all_nan, per_joint = [], {}
    for j, name in enumerate(JOINTS):
        miss_j = 0
        for ax in (0, 1):
            v = Q[:, j, ax]
            ok = np.isfinite(v)
            if not ok.any():
                if ax == 0 and name not in all_nan:
                    all_nan.append(name)
                continue
            if ok.all():
                continue
            Q[:, j, ax] = np.interp(t, t[ok], v[ok])      # 양 끝은 최근접값으로 유지된다
            filled_any |= ~ok
            miss_j = max(miss_j, int((~ok).sum()))
        if miss_j:
            per_joint[name] = miss_j

    gaps, run = [], 0
    for f in filled_any:
        run = run + 1 if f else 0
        if run:
            gaps.append(run)
    return Q, {
        "n_frames_interpolated": int(filled_any.sum()),
        "longest_gap_frames": int(max(gaps)) if gaps else 0,
        "per_joint_missing_frames": per_joint,
        "all_nan_joints": all_nan,
        "all_nan_required": [j for j in all_nan if j in REQUIRED_JOINTS],
        "all_nan_optional": [j for j in all_nan if j in OPTIONAL_JOINTS],
        "interpolated_mask": filled_any,
        "note": "보간된 프레임의 지표값은 실제 관측이 아니다",
    }


def flip_x(P):
    """좌우반전을 되돌린다. x 부호만 뒤집으면 되고 지표는 전부 상대량이라 원점은 무관하다.

    되돌린 뒤에는 MediaPipe 의 'Left ~' 가 실제 사람의 왼쪽을 가리킨다.
    """
    Q = P.copy()
    Q[..., 0] = -Q[..., 0]
    return Q


def ankle_line_angle(P, use_heel=False):
    """발목선(=바닥선)이 수평에서 벗어난 각도(rad). 전 프레임 중앙값.

    발은 바닥에 붙어 있으므로 발목선은 바닥의 대리 기준선이 된다.
    뒤꿈치(29/30)가 발목(27/28)보다 바닥에 가까우므로 보이면 그쪽이 낫다.
    """
    L, R = ("Left Heel", "Right Heel") if use_heel else ("Left Ankle", "Right Ankle")
    la, ra = P[:, IDX[L]], P[:, IDX[R]]
    th = np.arctan2(ra[:, 1] - la[:, 1], ra[:, 0] - la[:, 0])
    th = np.where(np.abs(th) > np.pi / 2, th - np.sign(th) * np.pi, th)   # 좌우 순서 무관
    t = float(np.nanmedian(th))
    return t if np.isfinite(t) else 0.0


def align_roll(P, use_heel=False):
    """발목선을 수평으로 맞춰 카메라 roll 을 제거한다.

    검증: 합성 perturbation 에서 roll 10° 의 영향이 **정확히 0** 이 된다.
          개인차(eta^2)도 같이 내려간다 — 좌우 무릎높이 차 0.67 → 0.07.
    주의: 진짜로 한쪽 발이 높은 경우(바닥 경사, 뒤꿈치 들림)도 같이 지워진다.
    """
    t = ankle_line_angle(P, use_heel=use_heel)
    c, s = np.cos(-t), np.sin(-t)
    return P @ np.array([[c, -s], [s, c]]).T
