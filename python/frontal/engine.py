# -*- coding: utf-8 -*-
"""engine.py — 정면 트랙 진입점. **판정하지 않는다. threshold 가 없다. gate 를 부르지 않는다.**

흐름
  입력(MediaPipe 또는 AI-Hub XY)
    → adapter    : 등방 픽셀화 · y 뒤집기 · mirror 표준화   (여기서 z 가 사라진다)
    → standing   : 정면 전용 기준자세 (키프레임 / 연속영상 경로 분리)
    → signals    : 골반 하강 신호 + 골반높이비 필터로 저점 검출
    → features   : PRIMARY / DYNAMIC / EXPLORATORY / AUXILIARY / QC 값만 산출
  판정(frontal profile)은 아직 만들지 않는다. 실촬영 데이터를 본 뒤에 만든다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frontal import a6_candidates as _a6        # noqa: E402
from frontal import adapter as _adapter        # noqa: E402
from frontal import consistency as _cons       # noqa: E402
from frontal import diagnostics as _diag       # noqa: E402
from frontal import features as _feat          # noqa: E402
from frontal import qc as _qc                   # noqa: E402
from frontal import landmarks as _lm           # noqa: E402
from frontal import rhythm as _rhythm          # noqa: E402
from frontal import schema as _schema          # noqa: E402
from frontal import signals as _sig            # noqa: E402
from frontal import standing as _stand         # noqa: E402

# 3D 트랙 전용 — 정면 입력으로 절대 부르면 안 되는 것들 (adapter.py 설계규칙 2 참조)
EARLY_SEARCH_S = 5.0     # run51: 캘리브레이션 실패 시 먼저 이 시간 안에서 서기 구간을 찾는다 (잠정)

FORBIDDEN_FOR_FRONTAL = (
    "squat_core.compute_gate_features", "squat_core.evaluate_gate",
    "squat_core.GATE_V2", "squat_core.analyze_set",
    "squat_core.extract_features", "squat_core.capture_standing_reference",
    "squat_core.detect_bottoms", "squat_core.compute_depth_auxiliary",
    "squat_core.compute_trunk_shortening", "squat_core.compute_frontal_knee_valgus_ratio",
)


def analyze_frontal(seq, fps=None, adj_merge=None, use_heel_for_roll=False,
                    calibration=None):
    """정면 시퀀스 → 지표값. 판정·피드백은 포함하지 않는다.

    기준자세 경로 (우선순위)
      1) calibration=(start_s, duration_s)  ★ 서비스 정상 경로.
         "정면을 보고 2초간 서 주세요" 안내에 해당하는 **지정 구간**을 그대로 쓴다.
      2) fps 만 주어짐                        fallback. 영상 전체에서 직립 구간을 추정한다.
      3) fps 없음                             희소 키프레임 입력 (AI-Hub 검증 경로).
    """
    fps = fps if fps is not None else seq.fps
    warnings = []
    qc_info = None
    continuous = bool(fps)

    # run43 — 연속영상이면 튄 프레임을 먼저 미검출로 바꾼다 (보간 전에). 키프레임 경로는 그대로.
    if continuous:
        jumps, jinfo = _qc.jump_mask(seq.P, fps)
        if jumps.any():
            Pj = seq.P.copy()
            Pj[jumps] = np.nan
            seq = seq._replace(P=Pj)
        missing = _qc.core_missing(seq.P)
        sh_missing = _qc.shoulder_missing(seq.P)      # run56 — C1 표시 조건용 (보간 전)
        long_gap, gap_runs = _qc.long_gap_mask(missing, fps)
        qc_info = {"jump_frames": int(jumps.sum()), "jump_frac": float(jumps.mean()),
                   "jump_max_dev_ratio": jinfo.get("max_dev_ratio"),
                   "core_missing_frac": float(missing.mean()),
                   "long_gaps_s": [((a / fps), (b / fps)) for a, b in gap_runs],
                   "params": {"JUMP_DEV_RATIO": _qc.JUMP_DEV_RATIO, "MAX_GAP_S": _qc.MAX_GAP_S,
                              "BOTTOM_GUARD_S": _qc.BOTTOM_GUARD_S}}
        if jumps.mean() > _qc.JUMP_FRAC_ERROR:
            warnings.append(_qc.warn("TRACKING_UNSTABLE", "error",
                                     "관절 위치가 자주 튀어 이 영상은 분석 결과를 내지 않습니다. "
                                     "정면에서 전신이 보이도록 다시 촬영해 주세요."))
        elif jumps.any():
            warnings.append(_qc.warn("TRACKING_JUMPS", "info",
                                     f"관절 위치가 튄 {int(jumps.sum())}프레임은 제외했습니다."))
        if missing.mean() > _qc.MISSING_FRAC_WARN:
            warnings.append(_qc.warn("LOW_DETECTION", "warn",
                                     "골반·무릎·발목이 잘 보이지 않는 장면이 많습니다."))

    # 미검출 프레임을 먼저 메운다. 그러지 않으면 저점 탐색이 All-NaN slice 로 죽는다.
    P_filled, gap = _lm.fill_gaps(seq.P)
    if gap["all_nan_required"]:
        raise ValueError(
            "전 구간 검출되지 않은 **필수** 관절이 있어 분석할 수 없다: "
            f"{gap['all_nan_required']}. 전신이 프레임 안에 들어와 있는지, "
            "발까지 보이는지 확인하라.")
    # 선택 관절(발뒤꿈치·코)이 없는 것은 정상이다 — 발목선으로 대체된다
    seq = seq._replace(P=P_filled)
    rolled = seq.rolled(use_heel=use_heel_for_roll)
    P, Pr = seq.P, rolled.P

    if calibration is not None:
        if not fps:
            raise ValueError("calibration 을 쓰려면 fps 가 필요하다.")
        start_s, dur_s = calibration
        ref = _stand.from_calibration_segment(P, fps, start_s=start_s, duration_s=dur_s)
        if ref.method == "calibration":
            # run43 — 가운데 한 프레임 대신 구간 중앙값 (서기 중 무릎폭 프레임 떨림 3~20 %, OpenCap 정면)
            ref = _stand._measure_window(P, ref.window, "calibration", ref.seconds_held, "high")
        else:
            # run51 — 먼저 영상 앞부분(EARLY_SEARCH_S)에서 서 있는 구간을 찾는다 (세트 시작 전 자기 상태가 기준).
            #   Session 1: 11개 중 10개가 앞 0~2 s 에서 실패 — 녹화 누른 직후 자리 잡는 움직임(앞 0.5 s 변동 0.02~0.05 span)
            #   이나 2 s 전에 시작. 하지만 그중 대부분은 앞 5 s 안에 조용한 0.5~1 s 가 있었다.
            early = missing.copy()
            early[int(round(EARLY_SEARCH_S * fps)):] = True
            alt = _stand.search_standing(P, fps, bad_mask=early, seconds=(1.0, 0.5))
            if alt.reliability != "failed":
                from dataclasses import replace as _dc_replace
                alt = _dc_replace(alt, method=alt.method.replace("search:", "early_search:"))
            else:
                # run43 — 앞부분에도 없으면 영상 전체에서 서 있는 구간을 찾는다
                alt = _stand.search_standing(P, fps, bad_mask=missing)
            if alt.reliability != "failed" and alt.method.startswith("early_search:"):
                ref = alt
            elif alt.reliability != "failed":
                warnings.append(_qc.warn(
                    "STANDING_CALIBRATION_FALLBACK", "info",
                    "앞부분에서 가만히 선 자세를 찾지 못해, 영상 중 서 있는 구간을 기준으로 썼습니다.",
                    calibration_method=ref.method))
                ref = alt
    elif fps:
        ref = _stand.search_standing(P, fps, bad_mask=missing)
    else:
        ref = _stand.from_keyframes(P)
    if continuous:
        if ref.reliability == "failed" or ref.method.startswith("calibration:"):
            warnings.append(_qc.warn("STANDING_NOT_FOUND", "error",
                                     "기준이 될 선 자세를 찾지 못했습니다. 시작할 때 1~2초 정도 "
                                     "가만히 선 뒤 스쿼트를 해 주세요.", standing_method=ref.method))
        elif ref.reliability == "low":
            warnings.append(_qc.warn("STANDING_SHORT", "warn",
                                     "가만히 선 구간이 짧아 기준 자세가 덜 안정적입니다.",
                                     standing_method=ref.method))

    bots, det = _sig.detect_bottoms(P, ref, fps=fps, adj_merge=adj_merge)

    per_rep = [_feat.rep_features(P, Pr, int(b), ref, ref.index) for b in bots]
    usable = None
    if continuous:
        usable = _qc.rep_usable(bots, long_gap, fps)
        # run50 — 저점 관절 붕괴(좌우 다리 겹침) 반복도 값으로 쓰지 않는다
        geo = _qc.rep_geometry_ok(P, bots, ref.window)
        collapsed = [i for i, (ok, _, _) in enumerate(geo) if not ok]
        usable = [u and g[0] for u, g in zip(usable, geo)]
        qc_info["collapsed_reps"] = [i + 1 for i in collapsed]
        qc_info["hip_ratio_at_bottoms"] = [round(g[1], 3) for g in geo]
        if bots and len(collapsed) / len(bots) >= _qc.COLLAPSE_FRAC_ERROR:
            warnings.append(_qc.warn("TRACKING_COLLAPSE", "error",
                                     "여러 반복에서 다리 관절이 겹쳐 잡혀 이 영상은 결과를 내지 않습니다. "
                                     "창문이나 조명을 등지지 말고, 다리 윤곽이 보이게 다시 촬영해 주세요.",
                                     reps=[i + 1 for i in collapsed]))
        for d, u in zip(per_rep, usable):
            d["usable"] = bool(u)
            if not u:      # 긴 공백을 보간한 값은 지표로 쓰지 않는다
                for k, v in list(d.items()):
                    if isinstance(v, (float, np.floating)):
                        d[k] = float("nan")
        # run56 — 어깨가 관측되지 않았으면 C1(정면 상체 길이)만 비운다. 다른 지표는 어깨를 쓰지 않아 그대로.
        c1_stand_ok, c1_rep_ok = _qc.c1_visible(sh_missing, bots, ref.window, fps)
        c1_hidden = [i + 1 for i, ok in enumerate(c1_rep_ok) if not (ok and c1_stand_ok) and usable[i]]
        for i, d in enumerate(per_rep):
            if not (c1_stand_ok and c1_rep_ok[i]):
                d["C1_trunk_span_rel"] = float("nan")
        qc_info["c1_standing_shoulders_ok"] = c1_stand_ok
        qc_info["c1_hidden_reps"] = c1_hidden
        if c1_hidden:
            warnings.append(_qc.warn("TRUNK_NOT_SHOWN", "info",
                                     ("준비자세에서 어깨가 잘 보이지 않아 상체 길이는 표시하지 않았습니다."
                                      if not c1_stand_ok else
                                      f"{len(c1_hidden)}회는 가장 낮은 자세에서 어깨가 가려져 상체 길이를 표시하지 않았습니다."),
                                     reps=c1_hidden))
        n_bad = int(len(usable) - sum(usable))
        if len(bots) == 0:
            warnings.append(_qc.warn("NO_REPS", "error",
                                     "앉았다 일어선 동작을 찾지 못했습니다."))
        elif n_bad == len(usable):
            warnings.append(_qc.warn("ALL_REPS_UNUSABLE", "error",
                                     "모든 반복의 가장 낮은 자세에서 몸이 가려져 값을 낼 수 없습니다."))
        elif n_bad:
            warnings.append(_qc.warn("REPS_EXCLUDED", "warn",
                                     f"{n_bad}회는 가장 낮은 자세에서 몸이 가려졌거나 관절 인식이 흔들려 값에서 뺐습니다.",
                                     reps=[i + 1 for i, u in enumerate(usable) if not u]))
    setv = _feat.set_features(P, Pr, ref)

    def med(k):
        v = [d[k] for d in per_rep if d.get(k) == d.get(k)]
        return float(np.median(v)) if v else float("nan")

    tiers = {
        "primary": {k: med(k) for k in _feat.PRIMARY},
        "dynamic": {k: med(k) for k in _feat.DYNAMIC},
        "exploratory": {**{k: med(k) for k in _feat.EXPLORATORY if k != "B4_pelvis_tilt_rng_r"},
                        "B4_pelvis_tilt_rng_r": setv["B4_pelvis_tilt_rng_r"]},
        "auxiliary": {k: med(k) for k in _feat.AUXILIARY},
        "qc_signals": {"B9_ankle_x_drift": setv["B9_ankle_x_drift"]},
    }
    # 좌우 구분 값은 mirror 가 확정된 경우에만 낸다
    if seq.side_labels_usable:
        tiers["side_specific"] = {k: med(k) for k in _feat.SIDE_SPECIFIC}
        side_note = None
    else:
        tiers["side_specific"] = None
        side_note = ("좌우반전 여부를 모른다 → '왼쪽/오른쪽' 을 특정하는 값과 피드백을 내지 않는다. "
                     "A6(좌우 비대칭 크기)은 부호가 없어 그대로 유효하다.")

    # ── 추가층 (Track 5) — 기존 값은 하나도 바꾸지 않는다 ───────────────────
    #   깊이 diagnostic : 3D compute_depth_auxiliary 정의를 정면 좌표계로 이식 (분모만 다름)
    depth_diag = _diag.depth_diagnostics(P, bots, ref)
    #   반복 일관성     : 이미 계산된 per_rep 을 재사용한다. 새 feature 없음.
    rep_cons = _cons.rep_consistency(
        per_rep, extra_series={"hip_knee_dy": [
            np.nan if v is None else v for v in depth_diag["per_rep_hip_knee_dy"]]})
    #   A6 후보         : 기존 A4/A5/A6 를 건드리지 않고 별도 계산만 한다 (exploratory)
    a6_cand = _a6.a6_candidates(P, bots, ref, per_rep=per_rep)
    #   리듬            : 인터페이스만. fps 가 없으면 아무 값도 내지 않는다.
    rhy = _rhythm.rhythm(bots, fps=fps, n_frames=seq.n_frames, source=seq.source)

    interp = gap["interpolated_mask"]
    out = {
        "source": seq.source,
        "n_frames": seq.n_frames,
        "gap_fill": {k: v for k, v in gap.items() if k != "interpolated_mask"},
        "bottom_on_interpolated_frame": [bool(interp[int(b)]) for b in bots],
        "standing": ref,
        "bottoms": bots,
        "n_reps": len(bots),
        "detection": det,
        "per_rep": per_rep,
        "features": tiers,
        "capture_qc": _feat.capture_qc(seq, ref),
        "side_note": side_note,
        "depth_diagnostics": depth_diag,
        "rep_consistency": rep_cons,
        "a6_candidates": a6_cand,
        "rhythm": rhy,
        "qc": qc_info,
        "reps_usable": usable,
        "warnings": warnings if continuous else None,
        "result_usable": (not any(w["severity"] == "error" for w in warnings)) if continuous else None,
        "judgement": None,       # 판정 없음 — frontal profile 은 아직 만들지 않았다
        "note": "값 산출 전용. threshold·판정·gate 없음. 3D 트랙과 완전히 분리되어 있다.",
    }
    out["schema"] = _schema.build(out)     # 5영역 재배치 (값을 새로 만들지 않는다)
    return out


def analyze_mediapipe(frames_landmarks, width, height, fps, mirrored=None, adj_merge=None,
                      calibration=(0.0, _stand.STAND_SECONDS)):
    """실제 서비스 경로.

    calibration 기본값 = 영상 맨 앞 2초 — 촬영 안내("2초간 서 주세요")와 같은 구간이다.
    캘리브레이션 없이 찍힌 영상이면 calibration=None 을 주어 fallback 추정으로 넘긴다.
    """
    seq = _adapter.from_mediapipe(frames_landmarks, width, height, fps=fps, mirrored=mirrored)
    return analyze_frontal(seq, fps=fps, adj_merge=adj_merge, calibration=calibration)


def analyze_aihub_xy(source):
    """검증 경로 — AI-Hub 3D 라벨에서 z 를 버리고 같은 코드를 태운다."""
    return analyze_frontal(_adapter.from_aihub_3d(source), fps=None)
