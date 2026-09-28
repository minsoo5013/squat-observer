# -*- coding: utf-8 -*-
"""consistency.py — Rep consistency layer (반복 일관성층). **판정하지 않는다. threshold 가 없다.**

왜 필요한가
  기존 운영 엔진은 반복별 값(per_rep)을 계산해 놓고도 **중앙값 하나만** 내보냈다.
  그래서 "세트 후반에 무너졌다" 같은, 사용자가 실제로 알고 싶은 정보가 전부 버려졌다.
  이 모듈은 이미 계산된 per_rep 을 **다시 쓰기만** 한다 — 새 feature 를 만들지 않는다.

무엇을 내는가 (지표별로)
  세트 내 흩어짐        sd · range · iqr · cv
  전반 vs 후반          early_mean · late_mean · late_minus_early
  반복 순서 추세        slope_per_rep (최소제곱) · rank_trend_rho (순위상관)
  반복별 이탈 순위      deviation_z (robust z) · 가장 많이 벗어난 반복

무엇을 하지 않는가
  · 합격/불합격, 좋다/나쁘다를 말하지 않는다. cutoff 가 없다.
  · deviation ranking 은 **상대 순위**이지 '문제 반복' 판정이 아니다.
    (서비스 흐름에서 evidence frame 을 고르는 후보 정렬용이다.)
  · 반복 수가 적으면 통계를 만들어내지 않고 부족하다고 표시한다.

주의 — 이 층의 안정성은 저점 검출의 안정성에 종속된다.
  저점이 ±1 프레임 흔들리면 여기 SD 도 흔들린다. 실촬영 30fps 에서 재확인해야 한다.
"""
from __future__ import annotations

import numpy as np

# 우선 대상 — 전부 **이미 존재하는** 지표다. 새로 정의한 것이 없다.
CONSISTENCY_KEYS = [
    "A2_knee_w_rel_stand",    # 무릎 정렬 (기준자세 대비 무릎폭)
    "A45_knee_in_mean",       # 무릎 안쪽 이동
    "D1_hip_ankle_rel",       # 깊이
    "C1_trunk_span_rel",      # 상체 (exploratory — 어깨 의존이 있다)
]

# ── 후속 측정으로 밝혀진 축 중복 (2026-09-19, exploration/rep_axis_separation.py) ──
#   **세트 안에서** A2 와 A45 는 Pearson r = −1.000000 (112/112 세트, 예외 없음) 이다.
#   둘 다 '양 무릎이 안쪽으로 이동한 양' 의 다른 표기이며, 분모(기준 무릎폭 vs 발목폭)와
#   부호만 다르다. 따라서 **반복 일관성 화면에 둘을 나란히 보여주면 같은 값을 두 번 보여주는 것**이다.
#   (세트 '사이' 비교에서는 분모가 사람마다 달라 서로 다른 값이 되므로, features.py 의
#    두 지표를 합치거나 지우지 않는다. 여기서는 표시 중복만 기록한다.)
#   D1 ↔ hip_knee_dy 도 세트 내 추세 상관 +0.922 로 사실상 같은 축(깊이)이다.
#   서로 독립적인 축은 셋이다:  무릎(A2 또는 A45) · 깊이(D1 또는 hip_knee_dy) · 상체(C1).
#   축 사이 추세 상관은 그 밖에는 |ρ| 0.13~0.24 로 낮다 — 따로 보여줄 근거가 있다.
#   아래 CONSISTENCY_KEYS 는 **계산 목록이지 표시 목록이 아니다.** 값은 그대로 다 낸다.

MIN_REPS_FOR_SPREAD = 3      # sd/range 를 말하려면 최소 3회
MIN_REPS_FOR_TREND = 4       # 전반/후반·추세를 말하려면 최소 4회
_EPS = 1e-9


def _rank(v):
    """평균 순위 (동점은 평균). scipy 없이 계산한다 — 서비스 경로 의존성을 늘리지 않는다."""
    v = np.asarray(v, float)
    order = np.argsort(v, kind="mergesort")
    r = np.empty(len(v), float)
    r[order] = np.arange(1, len(v) + 1, dtype=float)
    # 동점 평균 처리
    sv = v[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = np.mean(r[order[i:j + 1]])
        i = j + 1
    return r


def _pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3:
        return float("nan")
    sa, sb = a.std(), b.std()
    if sa < _EPS or sb < _EPS:
        return float("nan")
    return float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))


def _robust_z(v):
    """중앙값·MAD 기반 z. MAD 가 0 이면 표준편차로 대체하고, 그것도 0 이면 전부 0."""
    v = np.asarray(v, float)
    med = float(np.nanmedian(v))
    mad = float(np.nanmedian(np.abs(v - med))) * 1.4826
    if not np.isfinite(mad) or mad < _EPS:
        s = float(np.nanstd(v, ddof=1)) if np.isfinite(v).sum() > 1 else 0.0
        mad = s if s > _EPS else 0.0
    if mad < _EPS:
        return np.zeros_like(v), med, 0.0
    return (v - med) / mad, med, mad


def series_stats(values):
    """반복 순서대로 늘어선 값 하나에 대한 일관성 통계. 판정 없음."""
    v = np.asarray(values, float)
    finite = np.isfinite(v)
    n = int(finite.sum())
    out = {
        "n_reps_valid": n,
        "values": [None if not np.isfinite(x) else round(float(x), 4) for x in v],
        "mean": float("nan"), "median": float("nan"), "sd": float("nan"),
        "range": float("nan"), "iqr": float("nan"), "cv": float("nan"),
        "early_mean": float("nan"), "late_mean": float("nan"),
        "late_minus_early": float("nan"),
        "slope_per_rep": float("nan"), "rank_trend_rho": float("nan"),
        "first_rep": float("nan"), "last_rep": float("nan"),
        "deviation_z": [], "most_deviant_rep": None,
        "sufficiency": None,
    }
    if n == 0:
        out["sufficiency"] = "유효 반복 0회 — 계산하지 않는다."
        return out

    idx = np.where(finite)[0]
    x = v[finite]
    out["mean"] = float(np.mean(x))
    out["median"] = float(np.median(x))
    out["first_rep"] = float(x[0])
    out["last_rep"] = float(x[-1])

    if n < MIN_REPS_FOR_SPREAD:
        out["sufficiency"] = (f"반복 {n}회 — 흩어짐(sd/range)을 말하기에 부족하다 "
                              f"(최소 {MIN_REPS_FOR_SPREAD}회).")
        return out

    out["sd"] = float(np.std(x, ddof=1))
    out["range"] = float(np.max(x) - np.min(x))
    out["iqr"] = float(np.percentile(x, 75) - np.percentile(x, 25))
    m = abs(out["mean"])
    out["cv"] = float(out["sd"] / m) if m > _EPS else float("nan")

    z, _, scale = _robust_z(x)
    dz = [None] * len(v)
    for k, i in enumerate(idx):
        dz[i] = round(float(z[k]), 3)
    out["deviation_z"] = dz
    out["most_deviant_rep"] = int(idx[int(np.argmax(np.abs(z)))]) if scale > _EPS else None

    if n < MIN_REPS_FOR_TREND:
        out["sufficiency"] = (f"반복 {n}회 — 전반/후반 변화와 추세는 내지 않는다 "
                              f"(최소 {MIN_REPS_FOR_TREND}회).")
        return out

    half = n // 2
    out["early_mean"] = float(np.mean(x[:half]))
    out["late_mean"] = float(np.mean(x[n - half:]))
    out["late_minus_early"] = out["late_mean"] - out["early_mean"]

    t = idx.astype(float)                      # 반복 순서 (원래 인덱스 유지)
    out["slope_per_rep"] = float(np.polyfit(t, x, 1)[0])
    out["rank_trend_rho"] = _pearson(_rank(t), _rank(x))
    out["sufficiency"] = "ok"
    return out


def rep_consistency(per_rep, keys=None, extra_series=None):
    """운영 엔진의 per_rep 리스트 → 반복 일관성층.

    extra_series: {key: [반복별 값]} — per_rep 바깥에서 계산된 값(예: 깊이 diagnostic)을
                  같은 통계로 태우고 싶을 때 쓴다.
    """
    keys = list(CONSISTENCY_KEYS if keys is None else keys)
    n = len(per_rep)
    series = {k: [d.get(k, float("nan")) for d in per_rep] for k in keys}
    if extra_series:
        for k, v in extra_series.items():
            series[k] = list(v)
            if k not in keys:
                keys.append(k)

    per_key = {k: series_stats(series[k]) for k in keys}

    # 반복별 종합 이탈도 — 지표별 robust z 의 평균 절댓값. **순위일 뿐 판정이 아니다.**
    zmat = []
    for k in keys:
        dz = per_key[k]["deviation_z"]
        if dz:
            zmat.append([np.nan if x is None else abs(x) for x in dz])
    if zmat:
        agg = np.nanmean(np.asarray(zmat, float), axis=0)
        rank_order = [int(i) for i in np.argsort(-np.nan_to_num(agg, nan=-1.0))]
        rep_dev = [None if not np.isfinite(a) else round(float(a), 3) for a in agg]
    else:
        rep_dev, rank_order = [None] * n, []

    return {
        "n_reps": n,
        "keys": keys,
        "per_key": per_key,
        "rep_deviation_score": rep_dev,       # 반복별 |robust z| 평균
        "rep_rank_by_deviation": rank_order,  # 많이 벗어난 반복부터 — evidence frame 후보 정렬용
        "judgement": None,
        "note": ("반복 일관성 값만 낸다. 합격선·좋다/나쁘다 판정이 없다. "
                 "deviation ranking 은 상대 순위이며 '문제 반복' 판정이 아니다. "
                 "이 층의 안정성은 저점 검출 안정성에 종속된다 — 실촬영 30fps 에서 재확인 필요."),
    }
