# -*- coding: utf-8 -*-
"""frontal — 정면 단일카메라(MediaPipe) 트랙.

3D 트랙(src/squat_core.py)과 **판정 로직이 완전히 분리**되어 있다.
공유하는 것은 좌표를 받지 않는 1차원 신호 처리 함수뿐이다 (signals.ALLOWED_CORE_FUNCS).
gate v2 는 3D 전용이며 이 패키지는 gate 를 import 하지 않는다.
"""
from frontal import adapter, features, landmarks, signals, standing  # noqa: F401
