# -*- coding: utf-8 -*-
"""diagnostics.py — 정면 트랙 깊이 diagnostic (hip_knee_dy · parallel_reached).

*** 이 값들은 diagnostic / reference 전용이다. 자세 합격·불합격 판정에 쓰지 않는다. ***

이식 원본 (src/squat_core.py::compute_depth_auxiliary — **읽기만 했고 수정하지 않았다**)
    hip_knee_dy = ( mean(골반 y) − mean(무릎 y) ) / skel_span      ← 저점 프레임에서
    세트값       = 반복별 hip_knee_dy 의 중앙값
    parallel_reached = 세트값 ≤ 0.0                                ← 골반이 무릎보다 낮음
  원본 docstring 이 남긴 경고를 그대로 승계한다:
    · "diagnostic / reference 전용이며 depth warning 에 연결하지 않는다"
    · AI-Hub 112세트 중 조건 만족은 14.3%(16/112) 뿐이다 — 대부분의 정상 스쿼트가 '미달' 로 나온다
    · 카메라 pitch 에 따라 **부호가 뒤집힌다**

정면 이식에서 달라진 것 — 분모 하나
    3D:   skel_span   = 머리~발목 세로길이
    정면: span_sh_ank = **어깨~발목** 세로길이   (MediaPipe 에 머리끝 관절이 없다)
  어깨~발목 < 머리~발목 이므로 **같은 자세라도 정면 hip_knee_dy 의 절댓값이 더 크다.**
  => 3D 값과 숫자를 직접 비교하지 말 것. 임계값 0 을 옮겨쓰지 말 것.
  다만 분모가 양수이므로 **부호는 보존된다** → parallel_reached 자체는 두 트랙에서 같은 뜻이다.

정면에서 추가로 위험한 점
  · 카메라가 무릎보다 높거나 낮으면 골반·무릎의 화면상 y 차이가 직접 바뀐다 (원본 pitch 경고와 동일).
  · 정면에서는 대퇴가 화면 깊이 방향으로 누우므로, 실제 'thigh parallel' 과 화면상 y 비교는 같지 않다.
  => 그래서 사용자에게 "깊이가 부족합니다" 로 번역하지 않는다. 깊이 판정은 D1 과 함께 실촬영 후 결정한다.
"""
from __future__ import annotations

import numpy as np

from frontal import landmarks as L

# 원본과 같은 값. **판정 임계값이 아니라 정의상의 기준선(골반=무릎 높이)이다.**
PARALLEL_LEVEL = 0.0

FORBIDDEN_USE = (
    "자세 합격/불합격 판정", "depth warning", "gate", "사용자 피드백 문구 직접 생성",
)


def _mid_y(P, t, a, b):
    return float(np.nanmean([P[t, L.IDX[a]][1], P[t, L.IDX[b]][1]]))


def hip_knee_dy_at(P, t, span):
    """한 프레임의 (골반y − 무릎y) / span. y 는 위쪽이 + (adapter 규약)."""
    span = float(span)
    if not np.isfinite(span) or abs(span) < 1e-9:
        return float("nan")
    d = _mid_y(P, t, "Left Hip", "Right Hip") - _mid_y(P, t, "Left Knee", "Right Knee")
    return float(d) / span if np.isfinite(d) else float("nan")


def depth_diagnostics(P, bottoms, ref):
    """저점 프레임들에 대한 깊이 diagnostic. 판정하지 않는다.

    반환
      per_rep_hip_knee_dy   반복별 값 (반복 일관성층에 그대로 넘길 수 있다)
      hip_knee_dy           세트 대표값 = 반복별 값의 중앙값 (원본과 같은 집계)
      parallel_reached      세트 대표값 ≤ 0  — **diagnostic 표시 전용**
      parallel_rep_fraction 반복 중 ≤0 인 비율 (원본에 없던 보조 표시. 판정 아님)
    """
    span = ref.span_sh_ank
    vals = [hip_knee_dy_at(P, int(b), span) for b in bottoms]
    fin = [v for v in vals if np.isfinite(v)]
    med = float(np.median(fin)) if fin else float("nan")
    return {
        "per_rep_hip_knee_dy": [None if not np.isfinite(v) else round(float(v), 4) for v in vals],
        "hip_knee_dy": round(med, 4) if np.isfinite(med) else float("nan"),
        "parallel_reached": bool(np.isfinite(med) and med <= PARALLEL_LEVEL),
        "parallel_rep_fraction": (round(float(np.mean([v <= PARALLEL_LEVEL for v in fin])), 3)
                                  if fin else float("nan")),
        "denominator": "span_sh_ank (어깨~발목)",
        "denominator_3d_equivalent": "skel_span (머리~발목)",
        "numerically_comparable_to_3d": False,
        "usage": "diagnostic / reference 전용",
        "not_for": list(FORBIDDEN_USE),
        "note": ("3D 원본과 분모가 다르므로 숫자를 직접 비교하지 않는다(정면 값의 절댓값이 더 크다). "
                 "부호는 보존되므로 parallel_reached 의 의미는 같다. "
                 "카메라 높이(pitch)에 따라 부호가 뒤집힐 수 있다 — 원본 경고를 그대로 승계한다. "
                 "자세 합격/불합격 판정에 쓰지 않는다."),
    }
