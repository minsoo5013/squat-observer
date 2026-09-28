# -*- coding: utf-8 -*-
"""adapter.py — 정면 입력을 내부 표현으로 바꾸는 경계층.

이 층을 지나면 데이터는 **2차원뿐**이다. z 는 저장하지도, 전달하지도 않는다.

두 입력을 같은 표현으로 받는다.
  from_mediapipe()  실제 서비스 경로. 정규화 landmark → 등방 픽셀 (x·W, y·H), y 위로 뒤집음.
  from_aihub_3d()   검증 경로. AI-Hub 3D 라벨에서 z 를 **버려서** 정면 관측을 모사한다.
                    두 경로가 같은 지표 코드를 타야 정면 로직을 AI-Hub 로 검증할 수 있다.

=====================================================================
 설계 규칙 1 — z 를 정면 트랙에 들여보내지 않는다
=====================================================================
· FrontalSequence 는 (T, J, 2) 만 갖는다. pose_landmarks.z 도 world z 도 읽지 않는다.
· squat_core 가 먹는 frames(dict with x/y/z) 형식을 **만들지 않는다.**
  z=0 을 채워 넣으면 core 의 angle_3d 가 '유효한 3D 각'인 척 값을 돌려준다 — 가장 위험한 형태다.
  (정면 투영 무릎각은 실제로 무너진다: 저점 검출 정확도 100% → 40.5%)
· 따라서 정면 트랙이 squat_core 에서 쓰는 것은 **좌표를 받지 않는 1차원 신호 처리 함수뿐**이다.
  signals.py 의 ALLOWED_CORE_FUNCS 참조.

=====================================================================
 설계 규칙 2 — gate v2 는 정면 입력에 절대 태우지 않는다
=====================================================================
squat_core.trunk_endpoints() 는 Back/Waist 가 없으면 **조용히** 어깨중점~골반중점으로
대체한다. 그 상태로 evaluate_gate() 를 부르면 3D 로 보정된 trunk_lean_max <= 70° 가
정의가 다른 값에 적용된다 — 에러 없이 틀린 판정이 나온다.
또 spread_med = |la.z - ra.z| 는 정면에 존재하지 않는다.
=> 이 패키지는 gate 관련 함수를 import 하지 않는다. 정면 트랙의 입력 적정성 검사는
   실촬영 데이터를 본 뒤 별도로 만든다.

=====================================================================
 설계 규칙 3 — 좌우반전(mirror) 을 여기서 표준화한다
=====================================================================
mirrored=True 면 x 를 뒤집어 'Left ~' 가 실제 사람의 왼쪽을 가리키게 만든다.
mirrored=None(알 수 없음) 이면 뒤집지 않고 **좌우 구분이 필요한 출력을 억제**한다.
자동 판별이 불가능한 이유는 landmarks.py 상단 주석 참조.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frontal import landmarks as L            # noqa: E402


@dataclass
class FrontalSequence:
    """정면 시퀀스 하나. 등방 좌표, y 는 위쪽이 +, **z 없음**."""
    P: np.ndarray                    # (T, J, 2)
    visibility: np.ndarray           # (T, J)
    source: str                      # "mediapipe" | "aihub_3d_xy"
    mirrored: bool | None = None     # None = 알 수 없음 → 좌우 구분 출력 억제
    roll_corrected: bool = False
    fps: float | None = None
    image_size: tuple | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.P.ndim != 3 or self.P.shape[2] != 2:
            raise ValueError(f"정면 트랙은 (T,J,2) 만 받는다. 받은 모양: {self.P.shape}")

    @property
    def n_frames(self):
        return int(self.P.shape[0])

    @property
    def side_labels_usable(self):
        """'왼쪽 무릎' 처럼 좌우를 특정하는 출력을 내도 되는가."""
        return self.mirrored is not None

    def _replace(self, **kw):
        d = dict(P=self.P, visibility=self.visibility, source=self.source,
                 mirrored=self.mirrored, roll_corrected=self.roll_corrected,
                 fps=self.fps, image_size=self.image_size, meta=dict(self.meta))
        d.update(kw)
        return FrontalSequence(**d)

    def rolled(self, use_heel=False):
        """발목선 기준으로 roll 을 제거한 사본. 합성 검증에서 roll 영향이 정확히 0 이 된다."""
        if self.roll_corrected:
            return self
        return self._replace(P=L.align_roll(self.P, use_heel=use_heel), roll_corrected=True)


def standardize_mirror(seq: FrontalSequence, mirrored: bool | None):
    """좌우반전 표준화. 한 번만 부른다.

    mirrored=True   전면카메라 등으로 반전된 영상 → x 를 되돌린다.
    mirrored=False  이미 정상.
    mirrored=None   알 수 없음 → 좌표는 그대로 두고 좌우 구분 출력을 억제한다.

    mirror 에 **영향받지 않는** 지표: A1, A2, A45(평균), A6(|L-R|), D1, C1, B4, B7, B8 …
    mirror 에 **영향받는** 것: A4/A5 의 좌우 배정, 사용자에게 말하는 '왼쪽/오른쪽'.
    """
    if mirrored is True:
        return seq._replace(P=L.flip_x(seq.P), mirrored=True,
                            meta={**seq.meta, "mirror_applied": True})
    return seq._replace(mirrored=mirrored)


def from_mediapipe(frames_landmarks, width, height, fps=None, mirrored=None,
                   min_visibility=L.DEFAULT_MIN_VISIBILITY):
    """MediaPipe pose_landmarks 시퀀스 → FrontalSequence.

    pose_world_landmarks 는 읽지 않는다 — 단일 카메라에서는 모델이 추정한 깊이다.
    mirrored 는 촬영 계층이 알려줘야 한다 (전면카메라 여부 / 촬영 시 캘리브레이션).
    """
    P, V = L.to_pixel_xy(frames_landmarks, width, height, min_visibility)
    seq = FrontalSequence(P, V, "mediapipe", None, False, fps, (width, height),
                          {"min_visibility": min_visibility})
    return standardize_mirror(seq, mirrored)


def from_aihub_3d(source, mirrored=False):
    """AI-Hub 3D 라벨 JSON → z 를 버린 FrontalSequence (검증 전용).

    실제 정면 카메라가 아니라 '정사영으로 본 정면' 이다.
    AI-Hub 좌표계는 반전이 없으므로 mirrored=False 로 둔다.
    """
    from src import squat_core as sc          # 로더만 쓴다. 지표 계산에는 쓰지 않는다.
    frames = sc.load_input_frames(str(source))
    T = len(frames)
    P = np.full((T, len(L.JOINTS), 2), np.nan)
    V = np.ones((T, len(L.JOINTS)))
    for t, fr in enumerate(frames):
        pts = sc.get_frame_pts(fr)
        for name, j in L.IDX.items():
            key = "Head" if name == "Nose" else name
            v = sc.get_xyz(pts, key)[:2]        # z 를 여기서 버린다
            P[t, j] = v
            if np.any(np.isnan(v)):
                V[t, j] = 0.0
    return FrontalSequence(P, V, "aihub_3d_xy", mirrored, False, None, None,
                           {"path": str(source)})
