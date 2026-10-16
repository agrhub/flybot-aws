"""Op 4 on lure traps reaches the agent as activity."""

from pathlib import Path

import pytest

from sentinel.agent.planner import activity
from sentinel.agent.vision import PipelineAnalyzer, new_catch_of
from sentinel.common.schemas import Counts, Observation, PhotoEvent

FIX = Path(__file__).parent / "fixtures"


def ev(image, kind="lure"):
    return PhotoEvent(event_id="e1", camera="BE4089629", source="ezviz", trap_kind=kind, image=str(image))


def test_real_pipeline_reports_lure_activity_with_registration():
    pytest.importorskip("sentinel.vision.pipeline")
    obs = PipelineAnalyzer().analyze(ev(FIX / "vision_be4089629_cur.jpg"), str(FIX / "vision_be4089629_prev.jpg"))
    nc = obs.new_catch
    assert nc is not None and nc.meaning == "activity"
    assert nc.inliers is not None and nc.inliers > 0  # op 3 ran on the lure pair
    assert nc.skipped is not None or nc.new_count is not None  # matched (detector present) or skipped with a reason
    assert nc.card_change_suspected is False  # no card on a lure trap


def test_no_previous_image_no_diff():
    pytest.importorskip("sentinel.vision.pipeline")
    obs = PipelineAnalyzer().analyze(ev(FIX / "vision_be4089629_cur.jpg"), None)
    assert obs.new_catch is None


def test_new_catch_mapping_both_kinds():
    lure = new_catch_of({"trap_kind": "lure", "meaning": "activity", "new_count": 4, "disappeared_count": 1,
                         "persisted_count": 2, "registration_ok": True, "card_change_suspected": True},
                        {"ok": True, "inliers": 212})
    assert (lure.meaning, lure.new_count, lure.disappeared_count, lure.inliers) == ("activity", 4, 1, 212)
    assert lure.card_change_suspected is False  # ignored outside sticky cards
    sticky = new_catch_of({"new_count": 3, "card_change_suspected": True, "card_change_reasons": ["x"]}, None, "sticky")
    assert sticky.meaning == "catches" and sticky.card_change_suspected
    skipped = new_catch_of({"status": "skipped", "reason": "no boxes", "change_ratio": 0.02}, {"ok": True, "inliers": 90},
                           "lure")
    assert skipped.skipped == "no boxes" and skipped.inliers == 90 and skipped.new_count is None
    obs = Observation(event_id="e", camera="c", source="ezviz", trap_kind="lure", counts=Counts(opencv=5), new_catch=lure)
    assert activity(obs).startswith("activity since the previous photo: +4 arrivals, -1 departures")
