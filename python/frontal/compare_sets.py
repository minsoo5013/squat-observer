# -*- coding: utf-8 -*-
"""compare_sets.py — Retry/Verify: 직전 세트 vs 새 세트, **같은 지표**만 비교해 개선/유지/악화를 낸다. (run44)

판정(정상/비정상)이 아니다. 같은 사람이 방금 한 세트와 비교한 **변화 방향**만 말한다.
  Δ      = 새 세트 중앙값 − 직전 세트 중앙값   (각 세트의 '사용 가능한' 반복만)
  유지폭 = max(FLOOR[지표], 2 × √(s₁²/n₁ + s₂²/n₂))     s = 반복값 1.4826·MAD (세트 안 흔들림)
           ※ 통계 검정이 아니다. 반복 3~10회 세트의 경험적 '최소 변화량' (heuristic). 화면은 DISPLAY 문구만.
  |Δ| ≤ 유지폭 → 유지 / 좋아지는 방향으로 넘음 → 개선 / 반대 → 악화
  둘 중 하나라도 result_usable=False 이거나 사용 가능한 반복 < MIN_REPS 면 → 비교불가

FLOOR 근거 (run44 set_noise_current.json, 개발 데이터 OpenCap 정면 9명):
  같은 영상의 반복을 홀/짝으로 나누고 서로 다른 서기 창을 기준으로 삼은 '가짜 두 세트' 차이의 90 백분위.
  같은 사람·같은 동작이므로 이 차이는 잡음이다. ±40° 15개는 FLOOR 를 정하는 데 쓰지 않았다(오경보 시험용).
어느 쪽이 '좋아지는 방향' 인지는 피드백 규칙(Phase 2)이 정한다 — 여기서는 인자로 받는다.
numpy 만 쓴다 (웹 Pyodide).
"""
from __future__ import annotations

import numpy as np

FLOOR = {                       # 잠정값 (p90 을 소수 셋째 자리 올림) — 새 개발 촬영(연속 두 세트)으로 재확인
    "A1_knee_ankle_w": 0.060,       # p90 0.0595
    "A2_knee_w_rel_stand": 0.089,   # p90 0.0888
    "A45_knee_in_mean": 0.031,      # p90 0.0304
    "D1_hip_ankle_rel": 0.031,      # p90 0.0307
}
SE_MULT = 2.0
MIN_REPS = 3
# 촬영 방향이 크게 바뀌었는지 (주의 표시만, 비교는 한다): 서기 어깨폭/몸길이 비율의 상대 변화.
# run44 eval_view_change.json — 같은 카메라 앞/뒤 절반 최대 0.113, 0°↔±40° 최소 0.131 → 사이값.
# 이 값으로는 40° 안팎의 큰 변화만 잡힌다 (15° 회전은 비율 3 % 로 잡음 안에 묻힘).
VIEW_CHANGE_REL = 0.12
# run52 — 발 간격(서기 발목폭/몸길이) 변화: 주의 표시만(비교는 한다).
#   검증 Session 1: 발 간격 10 % 넓을 때 A2 ≈ −0.10 (r −0.59) → 허용폭(0.089) 수준. A1 은 발 간격과 무관(r 0.07).
#   OpenCap 같은 영상 안 자연 변동 최대 1.5 %(0°)·3.8 %(±40°). → 8 % (A2 예상 변화 ≈ 0.08) 에서 주의. 잠정.
STANCE_CHANGE_REL = 0.08

LABEL = {                       # 사용자 문구용 이름 (판정어 없음)
    "A1_knee_ankle_w": "가장 낮은 자세의 무릎 간격 (발목 간격 대비)",   # run59: 세트 간 무릎 비교 지표
    "A2_knee_w_rel_stand": "준비자세 대비 무릎 간격",
    "A45_knee_in_mean": "무릎이 안쪽으로 움직인 정도",
    "D1_hip_ankle_rel": "가장 낮은 자세의 골반 높이",
}
VERDICTS = ("개선", "유지", "악화", "비교불가")   # 내부 코드. 화면 문구는 DISPLAY 를 쓴다
# 화면 표시 (run44 프로젝트 결정): 통계 검정이 아니다 — 2×SE 는 '내부 변화 허용폭' 일 뿐.
# '통계적으로 유의' · '확실히 개선' 같은 표현 금지.
DISPLAY = {"개선": "개선 경향", "유지": "비슷함", "악화": "악화 경향", "비교불가": "비교 불가"}   # 사용자 확정 2026-09-27


def _rep_values(result, key):
    per = result.get("per_rep") or []
    use = result.get("reps_usable") or [True] * len(per)
    v = np.array([d.get(key, np.nan) for d, u in zip(per, use) if u], float)
    return v[np.isfinite(v)]


def _robust_sd(v):
    if len(v) < 2:
        return float("nan")
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def compare(prev, new, key, better, floor=None, se_mult=SE_MULT, min_reps=MIN_REPS):
    """prev·new = engine.analyze_* 결과 dict. better = 'higher' | 'lower' (문제 지표가 좋아지는 방향)."""
    if better not in ("higher", "lower"):
        raise ValueError("better 는 'higher' 또는 'lower'")
    out = {"metric": key, "label": LABEL.get(key, key), "better": better, "verdict": "비교불가",
           "reasons": []}
    for name, r in (("직전 세트", prev), ("새 세트", new)):
        if r.get("result_usable") is False:
            out["reasons"].append(f"{name} 결과를 쓸 수 없음 ({', '.join(w['code'] for w in r.get('warnings') or [] if w['severity'] == 'error')})")
    a, b = _rep_values(prev, key), _rep_values(new, key)
    if len(a) < min_reps or len(b) < min_reps:
        out["reasons"].append(f"비교에 쓸 수 있는 반복이 부족함 (직전 {len(a)}회 · 새 {len(b)}회, 각 {min_reps}회 이상 필요)")
    out.update(n_prev=int(len(a)), n_new=int(len(b)))
    if out["reasons"]:
        return out
    out["cautions"] = []
    vc = view_change(prev, new)
    out["view_change_rel"] = vc
    if vc is not None and vc > VIEW_CHANGE_REL:
        out["cautions"].append("두 세트의 촬영 방향이 달라 보여 비교가 덜 정확할 수 있습니다")
    sc = stance_change(prev, new)
    out["stance_change_rel"] = sc
    out["stance_changed"] = bool(sc is not None and sc > STANCE_CHANGE_REL)
    if out["stance_changed"]:
        out["cautions"].append("발 간격이 직전 세트와 달라 직접 비교에 주의가 필요합니다")
    ma, mb = float(np.median(a)), float(np.median(b))
    sa, sb = _robust_sd(a), _robust_sd(b)
    se = float(np.sqrt(sa ** 2 / len(a) + sb ** 2 / len(b)))
    fl = FLOOR.get(key, 0.0) if floor is None else floor
    band = max(fl, se_mult * se)
    d = mb - ma
    good = d if better == "higher" else -d
    verdict = "유지" if abs(d) <= band else ("개선" if good > 0 else "악화")
    out.update(verdict=verdict, prev_median=ma, new_median=mb, delta=d, band=band,
               band_source="floor" if band == fl else "within_set_spread",
               prev_sd=sa, new_sd=sb)
    return out


def view_change(prev, new):
    def g(r, k):                      # dataclass 또는 JSON(dict) 둘 다 받는다 (웹은 dict)
        return r[k] if isinstance(r, dict) else getattr(r, k)
    try:
        ra, rb = prev["standing"], new["standing"]
        ga = g(ra, "shoulder_w") / g(ra, "span_sh_ank")
        gb = g(rb, "shoulder_w") / g(rb, "span_sh_ank")
        v = abs(gb / ga - 1.0)
        return float(v) if np.isfinite(v) else None
    except (KeyError, AttributeError, ZeroDivisionError, TypeError):
        return None


def compare_with_consistency(prev, new, key, better, checks, min_reps=MIN_REPS):
    """run47 — 주 지표(key)의 개선/악화를 **보조 지표 방향과 맞을 때만** 인정한다. 복합 점수가 아니다.

    checks = [(보조키, 보조방향), ...]. 주 지표가 개선(또는 악화)인데 보조 지표가 **반대 방향**으로 움직였으면
    (부호만 본다 — A2 와 A45 는 같은 무릎 움직임의 다른 표현이라 진짜 변화면 함께 움직인다)
    → verdict 를 '유지' 로 낮추고 consistency_conflict=True, 주의 문구를 붙인다.
    보조 지표를 계산할 수 없으면 주 지표 판단을 그대로 두고 그 사실만 남긴다.
    """
    r = compare(prev, new, key, better, min_reps=min_reps)
    r["consistency"] = []
    r["consistency_conflict"] = False
    if r["verdict"] not in ("개선", "악화"):
        return r
    main_sign = 1 if r["verdict"] == "개선" else -1
    for ck, cb in checks:
        c = compare(prev, new, ck, cb, min_reps=min_reps)
        if c.get("delta") is None:
            r["consistency"].append({"metric": ck, "status": "계산 불가"})
            continue
        c_good = c["delta"] if cb == "higher" else -c["delta"]
        agree = (c_good * main_sign) > 0
        r["consistency"].append({"metric": ck, "delta": c["delta"], "agree": bool(agree), "verdict": c["verdict"]})
        if not agree:
            r["consistency_conflict"] = True
    if r["consistency_conflict"]:
        r["verdict_before_check"] = r["verdict"]
        r["verdict"] = "유지"
        r.setdefault("cautions", []).append(
            "무릎 간격과 무릎 안쪽 이동의 방향이 엇갈려 변화로 보지 않았습니다")
    return r


def stance_change(prev, new):
    """서기 발목폭/몸길이 의 상대 변화 (발 간격이 달라졌는가)."""
    def g(r, k):
        return r[k] if isinstance(r, dict) else getattr(r, k)
    try:
        ra, rb = prev["standing"], new["standing"]
        v = abs((g(rb, "ankle_w") / g(rb, "span_sh_ank")) / (g(ra, "ankle_w") / g(ra, "span_sh_ank")) - 1.0)
        return float(v) if np.isfinite(v) else None
    except (KeyError, AttributeError, ZeroDivisionError, TypeError):
        return None


def compare_many(prev, new, targets):
    """targets = [(key, better), ...] 또는 [(key, better, [(보조키, 보조방향), ...]), ...]
    — 직전 세트 피드백이 가리킨 지표들 (feedback_rules.retry_targets 출력을 그대로 받는다)."""
    out = []
    for t in targets:
        if len(t) >= 3 and t[2]:
            out.append(compare_with_consistency(prev, new, t[0], t[1], t[2]))
        else:
            out.append(compare(prev, new, t[0], t[1]))
    return out


def summary_text(r):
    """사용자 문구 초안 (판정어·교정·위험 표현 없음). UI 가 그대로 쓰거나 바꿔 쓴다."""
    if r["verdict"] == "비교불가":
        return f"[{DISPLAY['비교불가']}] {r['label']}: " + "; ".join(r["reasons"])
    way = {"개선": "직전 세트보다 목표한 방향으로 달라진 경향이 보입니다",
           "유지": "직전 세트와 비슷합니다",
           "악화": "직전 세트보다 목표와 반대 방향으로 달라진 경향이 보입니다"}[r["verdict"]]
    cautions = list(r.get("cautions") or [])
    if r.get("consistency_conflict"):
        way = "다만 무릎 간격과 무릎 안쪽 이동의 방향이 엇갈려 변화로 판단하지 않았습니다"
        cautions = [c for c in cautions if "엇갈려" not in c]
    t = f"[{DISPLAY[r['verdict']]}] {r['label']}: {r['prev_median']:.2f} → {r['new_median']:.2f}. {way}."
    if cautions:
        t += " (" + "; ".join(cautions) + ")"
    return t
