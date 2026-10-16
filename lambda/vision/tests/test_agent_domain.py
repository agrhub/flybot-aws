"""Domain rules mirrored from FlyBot (surgeService, sprayPlanService, degreeDayService)."""

from datetime import UTC, datetime, timedelta

from sentinel.agent import domain
from sentinel.agent.policy import load_policy

P = load_policy()


def test_surge_sticky_peak_rule_matches_surge_service():
    daily = {"2026-10-12": {"peak": 4}, "2026-10-15": {"peak": 9}}
    s = domain.detect_surge(daily, "2026-10-15", P, "sticky")
    assert s["previous"] == 4 and s["count"] == 9 and s["pct"] == 125
    # each condition of surgeService: min count 5, +3, x1.5
    assert domain.detect_surge({"2026-10-12": {"peak": 1}, "2026-10-15": {"peak": 4}}, "2026-10-15", P) is None
    assert domain.detect_surge({"2026-10-12": {"peak": 6}, "2026-10-15": {"peak": 8}}, "2026-10-15", P) is None
    assert domain.detect_surge({"2026-10-12": {"peak": 10}, "2026-10-15": {"peak": 14}}, "2026-10-15", P) is None
    # base 0 -> pct 100 like FlyBot
    assert domain.detect_surge({"2026-10-12": {"peak": 0}, "2026-10-15": {"peak": 6}}, "2026-10-15", P)["pct"] == 100
    # missing base day -> None
    assert domain.detect_surge({"2026-10-15": {"peak": 50}}, "2026-10-15", P) is None


def test_surge_skipped_after_card_change_sticky_only():
    daily = {"2026-10-12": {"peak": 2, "total": 2, "scans": 1}, "2026-10-15": {"peak": 9, "total": 9, "scans": 1}}
    assert domain.detect_surge(daily, "2026-10-15", P, "sticky", ["2026-10-13"]) is None
    assert domain.detect_surge(daily, "2026-10-15", P, "sticky", ["2026-10-01"]) is not None
    assert domain.detect_surge(daily, "2026-10-15", P, "lure", ["2026-10-13"]) is not None  # no card on a lure trap


def test_surge_lure_uses_daily_average():
    daily = {"2026-10-12": {"peak": 9, "total": 12, "scans": 6}, "2026-10-15": {"peak": 9, "total": 14, "scans": 2}}
    s = domain.detect_surge(daily, "2026-10-15", P, "lure")
    assert s["basis"] == "daily_average" and s["previous"] == 2 and s["count"] == 7
    assert domain.detect_surge(daily, "2026-10-15", P, "sticky") is None  # same peaks: no sticky surge


def test_with_today_folds_current_count():
    d = domain.with_today({"2026-10-15": {"peak": 3, "total": 3, "scans": 1}}, "2026-10-15", 7)
    assert d["2026-10-15"] == {"peak": 7, "total": 10, "scans": 2}


def test_disagreement_threshold():
    assert not domain.disagreement({"npu": 10, "opencv": 12, "production": None}, P)["disagree"]  # 2 <= 3
    assert domain.disagreement({"npu": 30, "opencv": 18}, P)["disagree"]  # 12 > max(3, 6)
    assert not domain.disagreement({"npu": 100, "opencv": 85}, P)["disagree"]  # 15 <= 20
    assert not domain.disagreement({"npu": 5, "opencv": None, "production": None}, P)["disagree"]


def test_spray_suggest_mirrors_spray_plan_service():
    now = datetime(2026, 10, 15, 3, 0, tzinfo=UTC)
    base = {"sprays": [], "wave": None, "outlook": None, "today": "2026-10-15", "now": now, "policy": P}
    assert domain.spray_suggest(at_threshold=1, last_treatment=None, **base)["action"] == "treat_now"
    recent = {"at": now - timedelta(days=3)}
    r = domain.spray_suggest(at_threshold=1, last_treatment=recent, **base)
    assert r["action"] == "wait_check" and r["waitUntil"] == "2026-10-19"
    w = domain.spray_suggest(at_threshold=0, last_treatment=None, **{**base, "wave": {"date": "2026-10-22", "inDays": 7}})
    assert w["action"] == "bait_before_wave" and w["date"] == "2026-10-19"
    rain = {"level": None, "days": [{"day": "2026-10-15", "rain": 25}, {"day": "2026-10-16", "rain": 12}]}
    r = domain.spray_suggest(at_threshold=2, last_treatment=None, **{**base, "outlook": rain})
    assert r["date"] == "2026-10-17" and r["rainMoved"]
    sprays = [{"activeIngredient": "Abamectin"}, {"activeIngredient": "abamectin "}]
    assert domain.spray_suggest(at_threshold=1, last_treatment=None, **{**base, "sprays": sprays})["rotateFrom"] == "Abamectin"
    assert domain.spray_suggest(at_threshold=0, last_treatment=None, **base)["action"] == "monitor"


def test_degree_days():
    assert domain.day_degrees(20, 30) == 13
    assert domain.day_degrees(30, 40) == 20.5  # upper cutoff 35: (30 + 35) / 2 - 12
    assert domain.day_degrees(5, 10) == 0
    temps = [{"day": domain.add_days("2026-10-15", i), "min": 24, "max": 33} for i in range(-30, 8)]
    out = domain.next_wave(temps, {"2026-10-01": 12.0}, "2026-10-15")
    # 16.5 DD/day, 350 DD -> the generation emerges 22 days after the anchor
    assert out["nextWave"]["date"] == "2026-10-23" and out["anchor"]["day"] == "2026-10-01"
    assert domain.next_wave(temps, {"2026-10-01": 2.0}, "2026-10-15")["nextWave"] is None  # below MIN_WAVE
