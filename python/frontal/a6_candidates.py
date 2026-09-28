# -*- coding: utf-8 -*-
"""a6_candidates.py — A6(좌우 비대칭) 개선 **후보**. 기존 A6 를 대체하지 않는다.

*** 보호 규칙 ***
  · features.py 의 A4 / A5 / A6 정의를 건드리지 않는다. 이 파일은 **추가 계산**만 한다.
  · 후보는 전부 exploratory 다. 운영 판정에 연결하지 않는다.

왜 A6 가 흔들리는가 — 지금까지 측정된 것
  view3 실카메라 재현성 : A4 ρ=0.958 · A5 ρ=0.975 · 평균(A45) ρ=0.977 인데 **A6(차이) ρ=0.550**
  크기 비교             : |A4|,|A5| 중앙값 0.141 vs A6 중앙값 0.044 → 차이는 신호의 31%
  즉 A6 는 큰 두 값의 뺄셈이라 **catastrophic cancellation** 이 일어난다.
  (한때 '합성 yaw 민감도' 때문이라고 적었던 것은 틀렸다. off-center ρ=−0.126 p=0.18,
   A6 크기 ρ=+0.429 p<0.001 — 카메라 정렬 보정으로는 해결되지 않는다.)

추가로 발견한 기하학적 증폭 (후보 ②의 근거)
  A4 = (무릎L 이동)·inw_L,  A5 = (무릎R 이동)·inw_R 이고 inw_R = −inw_L 이다.
  따라서 몸 전체가 옆으로 δ 만큼 흔들리면  A4 += δ·inw_L,  A5 −= δ·inw_L 이 되어
  **A4 − A5 에는 2δ·inw_L 이 통째로 들어간다.** 좌우 흔들림이 비대칭으로 둔갑한다.

후보 셋
  ① A6c1_norm_asym          |A4−A5| / (|A4|+|A5|)
       공통성분(전체 무릎 모음 크기)으로 나눠 비대칭 '비율' 만 본다. 무차원 0~1.
       약점: 두 값이 모두 0 근처면 불안정하다 → denom 하한을 두고 부족하면 NaN.
  ② A6c2_baseline_corrected  기준자세 원점을 **창 중앙값**으로 바꾸고, 저점의 발목 중심선 이동 δ 를 뺀다.
       단일 기준 프레임 잡음과 위의 좌우 흔들림 증폭을 동시에 제거한다.
  ③ A6c3_across_rep         반복별 **부호 있는** 차이 s_i = A4_i − A5_i 를 먼저 모으고
       |median(s)| 를 쓴다. 기존은 |s_i| 를 먼저 취해 중앙값 → 잡음이 전부 양수로 쌓인다.
       sign_agreement(같은 쪽이 반복해서 나오는 비율)를 신뢰도로 함께 낸다.
  ④ A6c4_bc_across_rep      ②+③ 조합. 참고용으로 같이 계산한다.

평가 기준 (**사용자 지시**)
  AI-Hub 무릎 조건은 좌우 비대칭 라벨이 아니므로 AUC/지시조건 분리도를 선별 기준으로 쓰지 않는다.
  view3 재현성 · baseline 안정성 · 반복 간 방향 일관성 · perturbation robustness 로만 비교한다.
"""
from __future__ import annotations

import numpy as np

from frontal import landmarks as L

CANDIDATES = ["A6c1_norm_asym", "A6c2_baseline_corrected",
              "A6c3_across_rep", "A6c4_bc_across_rep"]

NORM_DENOM_FLOOR = 0.02      # |A4|+|A5| 가 이보다 작으면 ① 을 내지 않는다 (잠정값)
_EPS = 1e-9


def _safe(a, b):
    b = float(b)
    return float(a) / b if np.isfinite(b) and abs(b) > 1e-9 else float("nan")


def _x(P, t, n):
    return float(P[t, L.IDX[n]][0])


def _window_baseline(P, si, window=None):
    """기준자세 원점. window=None 이면 단일 프레임 si — 기존 A4/A5 와 **완전히 같다**."""
    if window is None:
        a, b = int(si), int(si) + 1
    else:
        a, b = int(window[0]), int(window[1])
        a = max(0, min(a, P.shape[0] - 1))
        b = max(a + 1, min(b, P.shape[0]))
    lk = np.nanmedian(P[a:b, L.IDX["Left Knee"], 0])
    rk = np.nanmedian(P[a:b, L.IDX["Right Knee"], 0])
    mid = np.nanmedian(np.nanmean([P[a:b, L.IDX["Left Ankle"], 0],
                                   P[a:b, L.IDX["Right Ankle"], 0]], axis=0))
    return float(lk), float(rk), float(mid)


def signed_components(P, bi, si, ankle_w, window=None, translation_correct=False):
    """저점 bi 에서의 (A4, A5) 부호 있는 성분.

    window=None, translation_correct=False → features.rep_features 의 A4/A5 와 **수치 동일**.
    (exploration/a6_candidate_compare.py 가 매 실행마다 이 동일성을 검사한다.)
    """
    x_L0, x_R0, mid0 = _window_baseline(P, si, window)
    inw_L = float(np.sign(mid0 - x_L0)) or 1.0
    inw_R = float(np.sign(mid0 - x_R0)) or -1.0
    d = 0.0
    if translation_correct:
        mid_b = float(np.nanmean([_x(P, int(bi), "Left Ankle"), _x(P, int(bi), "Right Ankle")]))
        d = mid_b - mid0
    a4 = _safe((_x(P, int(bi), "Left Knee") - x_L0 - d) * inw_L, ankle_w)
    a5 = _safe((_x(P, int(bi), "Right Knee") - x_R0 - d) * inw_R, ankle_w)
    return a4, a5


def norm_asym(a4, a5, floor=NORM_DENOM_FLOOR):
    """① 정규화 비대칭. 공통 성분으로 나눈다."""
    if not (np.isfinite(a4) and np.isfinite(a5)):
        return float("nan")
    den = abs(a4) + abs(a5)
    return float(abs(a4 - a5) / den) if den >= floor else float("nan")


def _sign_agreement(s):
    s = np.asarray([v for v in s if np.isfinite(v) and abs(v) > _EPS], float)
    if len(s) == 0:
        return float("nan")
    p = float(np.mean(s > 0))
    return float(max(p, 1.0 - p))


def _set_level(signed):
    fin = [v for v in signed if np.isfinite(v)]
    if not fin:
        return float("nan"), float("nan"), float("nan")
    med = float(np.median(fin))
    mad = float(np.median(np.abs(np.asarray(fin) - med))) * 1.4826
    ratio = float(abs(med) / mad) if mad > _EPS else float("nan")
    return abs(med), _sign_agreement(fin), ratio


def a6_candidates(P, bottoms, ref, per_rep=None):
    """세트 하나에 대한 A6 후보 전부. 기존 A6 는 건드리지 않고 참고용으로 재계산해 함께 낸다."""
    si, aw = ref.index, ref.ankle_w
    win = tuple(ref.window) if ref.window is not None else None

    base_signed, bc_signed = [], []
    c1, c2 = [], []
    for b in bottoms:
        a4, a5 = signed_components(P, b, si, aw, window=None, translation_correct=False)
        b4, b5 = signed_components(P, b, si, aw, window=win, translation_correct=True)
        base_signed.append(a4 - a5)
        bc_signed.append(b4 - b5)
        c1.append(norm_asym(a4, a5))
        c2.append(abs(b4 - b5) if np.isfinite(b4) and np.isfinite(b5) else float("nan"))

    def _med(v):
        f = [x for x in v if np.isfinite(x)]
        return float(np.median(f)) if f else float("nan")

    c3_val, c3_sign, c3_ratio = _set_level(base_signed)
    c4_val, c4_sign, c4_ratio = _set_level(bc_signed)

    out = {
        "A6_recomputed_reference": _med([abs(s) for s in base_signed]),  # = 기존 A6 (검산용)
        "A6c1_norm_asym": _med(c1),
        "A6c2_baseline_corrected": _med(c2),
        "A6c3_across_rep": c3_val,
        "A6c4_bc_across_rep": c4_val,
        "sign_agreement_base": c3_sign,
        "sign_agreement_bc": c4_sign,
        "median_over_mad_base": c3_ratio,
        "median_over_mad_bc": c4_ratio,
        "per_rep": {
            "signed_base": [round(float(v), 4) if np.isfinite(v) else None for v in base_signed],
            "signed_bc": [round(float(v), 4) if np.isfinite(v) else None for v in bc_signed],
            "A6c1_norm_asym": [round(float(v), 4) if np.isfinite(v) else None for v in c1],
            "A6c2_baseline_corrected": [round(float(v), 4) if np.isfinite(v) else None for v in c2],
        },
        "tier": "exploratory",
        "judgement": None,
        "note": ("A6 개선 후보. 기존 A6/A4/A5 정의는 그대로 두었다. "
                 "AUC·지시조건 분리도를 선별 기준으로 쓰지 않는다 — "
                 "AI-Hub 무릎 조건은 좌우 비대칭 라벨이 아니다."),
    }
    if per_rep is not None:
        # 기존 엔진이 이미 계산한 A6 와 검산 (정의 변경이 없었는지 확인)
        old = [d.get("A6_knee_in_asym", float("nan")) for d in per_rep]
        new = [abs(s) for s in base_signed]
        diff = [abs(a - b) for a, b in zip(old, new) if np.isfinite(a) and np.isfinite(b)]
        out["reference_max_abs_diff_vs_engine"] = float(max(diff)) if diff else float("nan")
    return out
