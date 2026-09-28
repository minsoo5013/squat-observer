# -*- coding: utf-8 -*-
"""feedback_rules.py — Phase 2 피드백 계기 규칙 (run46, **잠정**). 판정이 아니다.

원칙 (사용자 결정 2026-09-27)
  · population 절대 threshold 없음 ("A2 가 1.4 미만이면 문제" 같은 규칙 금지)
  · **같은 사람·같은 세트 안의 상대 변화**만 본다:
      KNEE_LATE   후반 반복의 무릎 간격이 초반 반복보다 지속적으로 좁아짐   (A2 ↓, A45 ↑ 함께)
      KNEE_REP    특정 반복의 무릎 간격이 나머지 반복보다 크게 좁음
      DEPTH_LATE  후반 반복이 초반보다 지속적으로 얕아짐                     (D1 ↑)
      DEPTH_REP   특정 반복이 나머지 반복보다 크게 얕음
  · 문턱은 '변화 허용폭' = max(FLOOR, k × 세트 안 흔들림). 통계 검정이 아니다.
  · 각 규칙은 Retry/Verify 에 넘길 (지표, 좋아지는 방향) 을 함께 낸다.
FLOOR 근거: OpenCap 정면(0°) 9명 정상 수행(지시 없음) 분포 — run46 diag_within_set.json.
  민수 님 controlled 촬영(좁게/넓게·얕게·후반 변화)으로 민감도를 확인한 뒤 조정한다. run41·외부 11개 미사용.
A6·내전근·둔근 활성화는 자동 추천 대상이 아니다 (여기서 다루지 않음). numpy 만 사용.
"""
from __future__ import annotations

import numpy as np

A2, A45, D1 = "A2_knee_w_rel_stand", "A45_knee_in_mean", "D1_hip_ankle_rel"
MIN_REPS = 4            # 초반/후반 또는 '나머지' 를 만들려면 최소 4회
GROUP = 3               # 초반·후반 묶음 크기 (반복 수가 적으면 n//2)
SE_MULT = 2.0           # 초반·후반 비교 허용폭의 흔들림 배수
REP_MULT = 3.0          # 한 반복 이탈 허용폭의 흔들림 배수
FLOOR_LATE = {A2: 0.089, A45: 0.031, D1: 0.031}   # = compare_sets.FLOOR (세트 간 잡음 p90)
FLOOR_REP = {A2: 0.19, D1: 0.06}                  # 0° 정상 수행 한 반복 최대 이탈 p90 (A2 0.187, D1 0.057)

# run50 — 깊이와 무릎 간격의 결합: 얕게 앉으면 무릎 간격(A2)도 자연히 줄어든다.
# 개발 촬영(민수 Session 1) '얕게' 영상 2개에서 무릎은 그대로 두고 깊이만 바꿨는데 KNEE_LATE 가 함께 걸렸다
# (D1 +0.17~0.24 일 때 A2 −0.2 안팎, 기울기 약 −0.9~−1.2). → 깊이 변화로 설명되는 만큼은 무릎 변화로 보지 않는다.
LATE_STRONG = 2.0       # 후반 경향을 한 반복보다 먼저 보여줄 세기(허용폭 배수). 잠정
DEPTH_COUPLING = 1.5    # A2 감소 허용 = 1.5 × (그 반복들이 더 얕아진 D1 양). 관측 기울기보다 보수적. 잠정

PRIORITY = ["KNEE_LATE", "KNEE_REP", "DEPTH_LATE", "DEPTH_REP"]   # 화면에는 한 가지 포인트(첫 번째)
CONTENT = {"KNEE_LATE": "knee_width_decrease", "KNEE_REP": "knee_width_decrease",
           "DEPTH_LATE": "relative_shallow_rep", "DEPTH_REP": "relative_shallow_rep"}


def _rsd(v):
    v = np.asarray(v, float)
    return float(1.4826 * np.median(np.abs(v - np.median(v)))) if len(v) >= 2 else float("nan")


def _vals(result, key, exclude=()):
    per = result.get("per_rep") or []
    use = result.get("reps_usable") or [True] * len(per)
    idx, v = [], []
    for i, (d, u) in enumerate(zip(per, use)):
        x = d.get(key)
        if u and (i + 1) not in exclude and x is not None and np.isfinite(float(x)):
            idx.append(i); v.append(float(x))
    return np.array(idx, int), np.array(v, float)


def _late_vs_early(result, key, sign, exclude=()):
    """sign=-1: 값이 작아지면 발동(A2), +1: 커지면 발동(A45·D1). exclude = 뺄 회차(1부터)."""
    idx, v = _vals(result, key, exclude)
    n = len(v)
    if n < MIN_REPS:
        return None
    g = min(GROUP, n // 2)
    e, l = v[:g], v[-g:]
    d = float(np.median(l) - np.median(e))
    se = float(np.sqrt(_rsd(e) ** 2 / g + _rsd(l) ** 2 / g)) if g >= 2 else float("nan")
    band = max(FLOOR_LATE[key], SE_MULT * se if np.isfinite(se) else 0.0)
    return {"fired": bool(sign * d > band), "delta": d, "band": band, "early": float(np.median(e)),
            "late": float(np.median(l)), "early_reps": (idx[:g] + 1).tolist(), "late_reps": (idx[-g:] + 1).tolist()}


def _rep_outlier(result, key, sign):
    idx, v = _vals(result, key)
    n = len(v)
    if n < MIN_REPS:
        return None
    best = None
    for j in range(n):
        rest = np.delete(v, j)
        d = float(v[j] - np.median(rest))
        band = max(FLOOR_REP[key], REP_MULT * _rsd(rest))
        if sign * d > band and (best is None or sign * d > sign * best["delta"]):
            best = {"rep": int(idx[j] + 1), "delta": d, "band": band, "value": float(v[j]), "others": float(np.median(rest))}
    return {"fired": best is not None, **(best or {})}


def evaluate(result):
    """engine 결과(dataclass 든 웹 JSON 이든) → 발동한 피드백 목록 (PRIORITY 순). 결과를 쓸 수 없으면 빈 목록."""
    if result.get("result_usable") is False:
        return []
    out = []
    # run50: 한 반복만 튄 경우(REP)가 '후반 경향'(LATE)으로도 잡히지 않게, LATE 는 REP 로 잡힌 반복을 빼고 본다
    kr = _rep_outlier(result, A2, -1)
    if kr and kr["fired"]:                                  # 그 반복이 더 얕았던 만큼은 빼고 본다
        idx, dv = _vals(result, D1)
        if kr["rep"] - 1 in idx.tolist():
            j = idx.tolist().index(kr["rep"] - 1)
            d1_dev = float(dv[j] - np.median(np.delete(dv, j))) if len(dv) >= 2 else 0.0
            kr["depth_explained"] = DEPTH_COUPLING * max(0.0, d1_dev)
            kr["fired"] = (-kr["delta"] - kr["depth_explained"]) > kr["band"]
    dr = _rep_outlier(result, D1, +1)
    k_ex = (kr["rep"],) if kr and kr["fired"] else ()
    d_ex = (dr["rep"],) if dr and dr["fired"] else ()
    kl, ka = _late_vs_early(result, A2, -1, k_ex), _late_vs_early(result, A45, +1, k_ex)
    dl = _late_vs_early(result, D1, +1, d_ex)
    dl_all = _late_vs_early(result, D1, +1, k_ex)
    if kl and kl["fired"] and dl_all:                       # 얕아져서 줄어든 만큼은 빼고 본다
        explained = DEPTH_COUPLING * max(0.0, dl_all["delta"])
        kl["depth_explained"] = explained
        kl["fired"] = (-kl["delta"] - explained) > kl["band"]
    if kl and ka and kl["fired"] and ka["fired"]:          # A2·A45 가 같은 방향일 때만 (둘은 같은 정보의 다른 표현)
        out.append({"rule_id": "KNEE_LATE", "metric": A2, "better": "higher", "reps": kl["late_reps"],
                    "evidence": {"A2": kl, "A45": ka},
                    "text": (f"후반 반복({', '.join(map(str, kl['late_reps']))}회차)에서 무릎 간격이 초반보다 좁아지는 경향이 보였습니다 "
                             f"(준비자세 대비 {kl['early']:.2f} → {kl['late']:.2f}).")})
    if kr and kr["fired"]:
        out.append({"rule_id": "KNEE_REP", "metric": A2, "better": "higher", "reps": [kr["rep"]], "evidence": kr,
                    "text": (f"{kr['rep']}회차에서 무릎 간격이 다른 반복보다 눈에 띄게 좁았습니다 "
                             f"(준비자세 대비 {kr['value']:.2f}, 다른 반복 {kr['others']:.2f}).")})
    if dl and dl["fired"]:
        out.append({"rule_id": "DEPTH_LATE", "metric": D1, "better": "lower", "reps": dl["late_reps"], "evidence": dl,
                    "text": (f"후반 반복({', '.join(map(str, dl['late_reps']))}회차)이 초반보다 얕아지는 경향이 보였습니다 "
                             f"(가장 낮은 자세의 골반 높이 {dl['early']:.2f} → {dl['late']:.2f}).")})
    if dr and dr["fired"]:
        out.append({"rule_id": "DEPTH_REP", "metric": D1, "better": "lower", "reps": [dr["rep"]], "evidence": dr,
                    "text": (f"{dr['rep']}회차가 다른 반복보다 눈에 띄게 얕았습니다 "
                             f"(골반 높이 {dr['value']:.2f}, 다른 반복 {dr['others']:.2f}).")})
    for f in out:
        f["content_group"] = CONTENT[f["rule_id"]]
        ev = f["evidence"]["A2"] if f["rule_id"] == "KNEE_LATE" else f["evidence"]
        f["strength"] = round((abs(ev["delta"]) - ev.get("depth_explained", 0.0)) / ev["band"], 2)   # 허용폭의 몇 배인가
    # run50: 무릎 > 깊이 순은 유지. 같은 계열 안에서는
    #   후반 경향(LATE)이 허용폭의 LATE_STRONG 배 이상이면 LATE 먼저 (점점 변함), 아니면 한 반복(REP) 먼저.
    #   근거: Session 1 — 한 반복만 좁힘(7598): LATE 1.3배·REP 2.1배 / 점점 좁힘(7601): LATE 2.2배. 잠정(개발 영상 2개).
    fam = lambda f: 0 if f["rule_id"].startswith("KNEE") else 1                         # noqa: E731
    def rank(f):
        if f["rule_id"].endswith("LATE"):
            return 0 if f["strength"] >= LATE_STRONG else 2
        return 1
    out.sort(key=lambda f: (fam(f), rank(f), -f["strength"]))
    for i, f in enumerate(out):
        f["primary"] = i == 0
    return out


CONSISTENCY = {A2: [(A45, "lower")]}   # run47: A2 개선/악화는 A45 가 같은 방향일 때만 인정 (primary=A2, A45=확인용)


def retry_targets(feedback):
    """Retry/Verify 에 넘길 [(지표, 방향, [(보조지표, 방향)])] — 중복 제거, 우선순위 순.
    compare_sets.compare_many 에 그대로 넘긴다."""
    seen, t = set(), []
    for f in feedback:
        if f["metric"] not in seen:
            seen.add(f["metric"]); t.append((f["metric"], f["better"], CONSISTENCY.get(f["metric"], [])))
    return t
