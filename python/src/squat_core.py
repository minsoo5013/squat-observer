# -*- coding: utf-8 -*-
"""squat_core.py — **웹 배포용 경량판** (Copyright © 2026 이민수. All rights reserved.)

원본 `src/squat_core.py`(3D 트랙 코어)는 웹에 싣지 않는다. 정면 엔진(frontal/)이 이 모듈에서 쓰는 것은
키프레임(AI-Hub 검증) 경로의 1차원 신호 함수뿐이고, 웹은 연속 영상 경로만 쓰므로 실제로는 호출되지 않는다.
아래 세 가지는 원본과 **같은 코드**(import 가 깨지지 않게 남김). 나머지 이름은 호출되면 명확히 실패한다.
동일성 검증: output/runs/2026-09-28_49_web-slim-core (원본 사본과 엔진 출력 비교).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

BEST_RULE = {"signal_name": "sig_knee", "smooth_window": 0,
             "trough_distance_ratio": 0.3, "trough_prominence_factor": 0.10}


def smooth(x, window=5, poly=3):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 5:
        return x.copy()
    w = min(window, n if n % 2 == 1 else n - 1)
    if w < poly + 2:
        return x.copy()
    if w % 2 == 0:
        w -= 1
    return savgol_filter(x, w, poly)


def zscore_arr(x):
    s = np.std(x)
    return (x - np.mean(x)) / (s + 1e-9)


def _web_unavailable(name):
    def f(*a, **k):
        raise NotImplementedError(f"웹 빌드에는 키프레임/AI-Hub 검증 경로({name})가 포함되어 있지 않습니다.")
    f.__name__ = name
    return f


_estimate_K_refined = _web_unavailable("_estimate_K_refined")
_detect_valley_first = _web_unavailable("_detect_valley_first")
load_input_frames = _web_unavailable("load_input_frames")
get_frame_pts = _web_unavailable("get_frame_pts")
get_xyz = _web_unavailable("get_xyz")
