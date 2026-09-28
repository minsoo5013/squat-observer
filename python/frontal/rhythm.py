# -*- coding: utf-8 -*-
"""rhythm.py — 리듬(템포) 층. ***인터페이스만 만든다. 성능검증·튜닝은 하지 않는다.***

왜 인터페이스만인가 — AI-Hub 에서는 검증이 불가능하다
  AI-Hub 클립은 한 세트를 **16개 키프레임**으로 희소 샘플링한 것이다.
  실측 저점 간격은 {2,3,4,5,6} 프레임, 중앙값 3.0, SD 0.84 —
  이건 사람의 템포가 아니라 **샘플러의 격자**다. 여기서 SD 를 재면 격자를 재는 것이다.
  그래서 이 모듈은 fps 가 없으면 **아무 값도 내지 않고** 이유를 남긴다.
  실제 30fps MP4 가 들어오기 전까지 threshold 도, 정상범위도 만들지 않는다.

무엇을 계산하는가 (fps 가 있을 때)
  bottom_time_sec        저점 시각
  rep_interval_sec       저점 간 간격 = 반복 주기
  interval sd / cv / range
  early vs late 평균 간격 (세트 후반에 느려졌는가)
  순서 추세 slope
무엇을 계산하지 않는가
  하강/상승 국면 분리, 저점 체류시간(pause), 템포 합격선.
  하강·상승 시간은 저점만으로 나눌 수 없다 — 상단 전환점 검출이 필요하고, 그건 실영상 뒤에 만든다.
"""
from __future__ import annotations

import numpy as np

from frontal import consistency as C

MIN_INTERVALS_FOR_SPREAD = 3       # 간격 3개 = 반복 4회
VALIDATED = False                  # 실촬영 데이터로 검증된 적 없음
_EPS = 1e-9

UNAVAILABLE_SPARSE = (
    "희소 키프레임 입력(fps 없음) — 프레임 간격이 실제 시간과 무관하다. "
    "AI-Hub 16키프레임에서 리듬을 재면 사람의 템포가 아니라 샘플러 격자를 재게 된다. "
    "실제 30fps 영상에서만 계산한다.")


def rhythm(bottoms, fps=None, n_frames=None, source=None):
    """저점 인덱스 → 리듬 값. 판정하지 않는다. threshold 가 없다."""
    b = [int(x) for x in (bottoms or [])]
    base = {
        "available": False, "reason": None, "validated": VALIDATED,
        "n_reps": len(b), "fps": fps, "source": source,
        "bottom_time_sec": None, "rep_interval_sec": None,
        "interval_stats": None, "judgement": None,
        "note": ("리듬층은 인터페이스만 구현되어 있다. "
                 "AI-Hub 희소 키프레임으로 성능검증·파라미터 튜닝을 하지 않는다. "
                 "합격선·정상 템포 범위는 실촬영 후에 정한다."),
        "not_implemented_yet": ["하강/상승 국면 분리", "저점 체류시간(pause)", "템포 합격선"],
    }
    if not fps or fps <= 0:
        base["reason"] = UNAVAILABLE_SPARSE
        return base
    if len(b) < 2:
        base["reason"] = f"반복 {len(b)}회 — 간격을 만들 수 없다 (최소 2회)."
        base["bottom_time_sec"] = [round(x / float(fps), 3) for x in b]
        return base

    t = np.asarray(b, float) / float(fps)
    iv = np.diff(t)
    st = C.series_stats(iv)

    out = dict(base)
    out["available"] = True
    out["bottom_time_sec"] = [round(float(x), 3) for x in t]
    out["rep_interval_sec"] = [round(float(x), 3) for x in iv]
    out["interval_stats"] = st
    out["n_intervals"] = int(len(iv))
    out["mean_interval_sec"] = float(np.mean(iv))
    out["reps_per_min"] = float(60.0 / np.mean(iv)) if np.mean(iv) > _EPS else float("nan")
    if len(iv) < MIN_INTERVALS_FOR_SPREAD:
        out["reason"] = (f"간격 {len(iv)}개 — 변동성을 말하기엔 부족하다 "
                         f"(최소 {MIN_INTERVALS_FOR_SPREAD}개 = 반복 4회). 원값만 낸다.")
    return out
