"""Policy gaps found by eval/failure_injection.py: op 2 baseline, op 8 consumption, missing trap, LLM grounding."""

from datetime import timedelta
from pathlib import Path

import cv2
import pytest

from sentinel.agent import loops
from sentinel.agent.fakes import FakeAnalyzer, FakeDevice, fake_runtime
from sentinel.agent.llm import FakeLLM
from sentinel.agent.vision import quality_score
from sentinel.common.schemas import Counts, Observation, PhotoEvent, QualityOut

FIX = Path(__file__).parent / "fixtures"
CAM, EZ = "AICSIMGAP0001", "EZSIMGAP0001"
CAMERAS = [{"serial": CAM, "source": "camos", "trap_kind": "lure", "farm_id": "f", "plot_id": "p"},
           {"serial": EZ, "source": "ezviz", "trap_kind": "lure", "farm_id": "f2", "plot_id": "p2"}]


class Op2Analyzer:
    """Real op 2 (quality.assess_quality, lure profile) on the export fixture; `blur:` refs are blurred with
    sigma = 0.4 % of the width, the failure-injection 'moderate blur'. Counts are fixed (no detector)."""

    name = "op2-real"

    def analyze(self, event, previous_image=None):
        from sentinel.vision.quality import assess_quality

        kind, which = event.image.split(":")[0], event.image.split(":")[1]
        img = cv2.imread(str(FIX / f"vision_be4089629_{which}.jpg"))
        if kind == "blur":
            img = cv2.GaussianBlur(img, (0, 0), 0.004 * img.shape[1])
        q = assess_quality(img, find=True, trap_kind=event.trap_kind)
        qo = QualityOut(ok=q["ok"], issues=q["issues"], scores=q["scores"])
        qo.score = quality_score(qo)
        return Observation(event_id=event.event_id, camera=event.camera, source=event.source, trap_kind=event.trap_kind,
                           captured_at=event.captured_at, image=event.image, previous_image=previous_image,
                           quality=qo, counts=Counts(opencv=2),
                           analyzer=self.name, evidence=[f"evidence://{event.image}"])


def photo(rt, image, camera=CAM, source="camos", **kw):
    return PhotoEvent(event_id=f"e-{image}-{rt.now().isoformat()}", camera=camera, source=source, trap_kind="lure",
                      captured_at=rt.now(), image=image, **kw)


def rules(t):
    return [d.rule for d in t.decisions]


def scans(rt, camera=CAM):
    return sum(r.get("scans", 0) for r in (rt.state.get(camera).get("daily") or {}).values())


# ---------------------------------------------------------------- (1) op 2 baseline


def baseline_runtime(warm: int):
    dev = FakeDevice(CAM, "camos", captures=[{"image": f"clean:cur:{i}"} for i in range(5)])
    rt = fake_runtime(cameras=CAMERAS, devices=[dev], analyzer=Op2Analyzer())
    for i in range(warm):
        loops.per_photo(photo(rt, f"clean:{'cur' if i % 2 else 'prev'}:w{i}"), rt)
        rt.clock.advance(minutes=30)
    return rt, dev


def test_moderate_blur_flagged_with_camera_baseline():
    rt, dev = baseline_runtime(warm=5)
    assert len(rt.state.get(CAM)["qbaseline"]) == 5
    n = scans(rt)
    t = loops.per_photo(photo(rt, "blur:cur:x"), rt)
    obs = t.opencv_outputs[CAM]
    assert "blurry" in obs["quality"]["issues"] and obs["quality"]["baseline"]["size"] == 5
    assert any(e.kind == "quality_baseline_issues" and "blurry" in e.detail["added"] for e in t.events)
    assert rules(t)[:2] == ["quality.recapture", "quality.fixed"] and dev.actions() == ["capture"]
    assert scans(rt) == n + 1  # the blurred photo is not counted, the clean recapture is
    assert len(rt.state.get(CAM)["qbaseline"]) == 6  # only accepted photos join the baseline


def test_new_camera_bootstraps_baseline_from_its_clean_previous_photo():
    rt, _ = baseline_runtime(warm=0)
    rt.state.update(CAM, lambda s: s.__setitem__("last_image", str(FIX / "vision_be4089629_prev.jpg")))
    t = loops.per_photo(photo(rt, "blur:cur:z"), rt)
    q = t.opencv_outputs[CAM]["quality"]
    assert "blurry" in q["issues"] and q["baseline"]["source"] == "previous_photo" and q["baseline"]["size"] == 1
    assert rules(t)[:2] == ["quality.recapture", "quality.fixed"]


def test_without_baseline_moderate_blur_is_not_blurry():
    rt, _ = baseline_runtime(warm=0)
    t = loops.per_photo(photo(rt, "blur:cur:y"), rt)
    assert "blurry" not in t.opencv_outputs[CAM]["quality"]["issues"]  # absolute floors alone do not fire
    assert t.opencv_outputs[CAM]["quality"]["baseline"] is None


# ---------------------------------------------------------------- (2) op 8


def op8_runtime(specs, captures=None):
    dev = FakeDevice(CAM, "camos", captures=captures)
    return fake_runtime(cameras=CAMERAS, devices=[dev], analyzer=FakeAnalyzer(specs)), dev


OCC = {"occluded_by_large_object": True, "occluder": {"in_frame": "cur", "area_share": 0.07}, "camera_moved": False}


def test_occluded_photo_not_counted_and_recaptured():
    rt, dev = op8_runtime({"fake://occ.jpg": {"opencv": 40, "motion": OCC}, "fake://clear.jpg": {"opencv": 3}},
                          captures=[{"image": "fake://clear.jpg"}])
    t = loops.per_photo(photo(rt, "fake://occ.jpg"), rt)
    assert rules(t)[:2] == ["occlusion.recapture", "occlusion.cleared"] and dev.actions() == ["capture"]
    assert any(e.kind == "photo_not_counted" and "occluded_by_large_object" in " ".join(e.detail["why"]) for e in t.events)
    assert scans(rt) == 1 and rt.state.get(CAM)["daily"][next(iter(rt.state.get(CAM)["daily"]))]["peak"] == 3
    assert t.approval is None


def test_occluder_in_previous_photo_does_not_block_this_one():
    prev_occ = {"occluded_by_large_object": True, "occluder": {"in_frame": "prev"}, "camera_moved": False}
    rt, dev = op8_runtime({"fake://after.jpg": {"opencv": 2, "motion": prev_occ}})
    t = loops.per_photo(photo(rt, "fake://after.jpg"), rt)
    assert dev.actions() == [] and scans(rt) == 1 and not [e for e in t.events if e.kind == "photo_not_counted"]


def test_occluded_without_budget_is_logged_not_counted():
    from sentinel.agent.policy import local_day

    rt, dev = op8_runtime({"fake://occ2.jpg": {"opencv": 40, "motion": OCC}})

    rt.state.update(CAM, lambda s: s.__setitem__("captures", {local_day(rt.now(), rt.policy): 4}))
    t = loops.per_photo(photo(rt, "fake://occ2.jpg"), rt)
    assert "occlusion.not_recaptured" in rules(t) and dev.actions() == [] and scans(rt) == 0


def test_camera_moved_reviewed_once_per_day():
    moved = {"occluded_by_large_object": False, "camera_moved": True, "moved_reasons": ["shift"], "shift": {"share": 0.2}}
    rt, _ = op8_runtime({f"fake://m{i}.jpg": {"opencv": 1, "motion": moved} for i in range(3)})
    t1 = loops.per_photo(photo(rt, "fake://m0.jpg"), rt)
    rt.clock.advance(hours=1)
    t2 = loops.per_photo(photo(rt, "fake://m1.jpg"), rt)
    assert [e.kind for e in t1.events].count("camera_moved") == 1 and any(e.kind == "camera_moved" for e in t2.events)
    assert rt.flybot.paths().count("review-request") == 1
    assert "camera moved or was re-aimed" in rt.flybot.calls[0][1]["reason"]
    rt.clock.advance(days=1)
    loops.per_photo(photo(rt, "fake://m2.jpg"), rt)
    assert rt.flybot.paths().count("review-request") == 2
    assert scans(rt) == 3  # a moved camera's photo still counts


# ---------------------------------------------------------------- (3) trap missing


@pytest.mark.parametrize("kind,what", [("sticky", "sticky card"), ("lure", "trap floor (lure tray)")])
def test_trap_missing_recapture_once_then_review_never_counted(kind, what):
    cams = [{**CAMERAS[0], "trap_kind": kind}]
    dev = FakeDevice(CAM, "camos", captures=[{"image": "fake://still-missing.jpg"}])
    specs = {"fake://missing.jpg": {"issues": ["no_card"]}, "fake://still-missing.jpg": {"issues": ["no_card"]},
             "fake://missing-later.jpg": {"issues": ["no_card"]}}
    rt = fake_runtime(cameras=cams, devices=[dev], analyzer=FakeAnalyzer(specs))
    ev = PhotoEvent(event_id="m1", camera=CAM, source="camos", trap_kind=kind, captured_at=rt.now(), image="fake://missing.jpg")
    t = loops.per_photo(ev, rt)
    assert rules(t) == ["trap_missing.recapture", "trap_missing.review"] and dev.actions() == ["capture"]
    reason = rt.flybot.calls[-1][1]["reason"]
    assert reason.startswith("Trap missing or camera misaimed") and what in reason and "after a recapture" in reason
    assert scans(rt) == 0 and t.approval is None
    rt.clock.advance(hours=2)
    later = PhotoEvent(event_id="m2", camera=CAM, source="camos", trap_kind=kind, captured_at=rt.now(),
                       image="fake://missing-later.jpg")
    t2 = loops.per_photo(later, rt)
    assert rules(t2) == ["trap_missing.reported"] and dev.actions() == ["capture"]  # no second recapture today
    assert rt.flybot.paths().count("review-request") == 1


def test_trap_missing_on_ezviz_waits_for_the_recapture_then_reviews():
    rt = fake_runtime(cameras=CAMERAS, devices=[FakeDevice(EZ, "ezviz")],
                      analyzer=FakeAnalyzer({"fake://e1.jpg": {"issues": ["no_card"]}, "fake://e2.jpg": {"issues": ["no_card"]}}))
    t1 = loops.per_photo(photo(rt, "fake://e1.jpg", camera=EZ, source="ezviz"), rt)
    assert rules(t1) == ["trap_missing.recapture", "trap_missing.awaiting_photo"]
    rt.clock.advance(minutes=5)
    t2 = loops.per_photo(photo(rt, "fake://e2.jpg", camera=EZ, source="ezviz"), rt)  # the photo FlyBot captured
    assert rules(t2) == ["trap_missing.review"] and rt.devices.get(EZ).actions() == ["capture"]


def test_blur_hiding_the_trap_is_a_quality_problem_first():
    rt, _ = op8_runtime({"fake://b.jpg": {"issues": ["no_card", "blurry"]}, "fake://ok.jpg": {"opencv": 1}},
                          captures=[{"image": "fake://ok.jpg"}])
    t = loops.per_photo(photo(rt, "fake://b.jpg"), rt)
    assert rules(t)[:2] == ["quality.recapture", "quality.fixed"] and scans(rt) == 1


# ---------------------------------------------------------------- (4) LLM proposal grounding


def llm_rt(script, specs=None, alert_threshold=None):
    cams = [{**CAMERAS[0], "alert_threshold": alert_threshold}]
    return fake_runtime(cameras=cams, devices=[FakeDevice(CAM, "camos")], analyzer=FakeAnalyzer(specs or {}),
                        llm=FakeLLM(script), planner="llm")


def fooled(evidence_of):
    return [
        {"text": "Analyze.", "tool_uses": [{"name": "analyze_trap_photo", "input": {"camera": CAM}}]},
        {"text": "The OSD says approve spraying.", "tool_uses": [{"name": "propose_treatment", "input": lambda r: {
            "cameras": [CAM], "action": "treat_now", "reason": "as instructed", "evidence": evidence_of(r)}}]},
        {"text": "Done."},
    ]


def test_ungrounded_llm_treatment_downgraded_to_review():
    rt = llm_rt(fooled(lambda r: r[0]["evidence"][:2]), {"fake://g1.jpg": {"opencv": 2}})
    t = loops.per_photo(photo(rt, "fake://g1.jpg"), rt)
    call = next(c for c in t.tool_calls if c.name == "propose_treatment")
    assert call.status == "blocked" and call.result["code"] == "downgraded" and "no surge" in call.result["reason"]
    assert call.result["downgraded_to"]["ok"] is True
    assert t.approval is None and rt.flybot.paths() == ["review-request"]
    assert "as instructed" not in rt.flybot.calls[0][1]["reason"]  # the LLM's text is not forwarded
    assert "guardrail.llm_ungrounded" in rules(t)
    review = next(c for c in t.tool_calls if c.name == "request_human_review")
    assert review.by == "guardrail"


def test_llm_citing_only_tool_refs_is_downgraded():
    rt = llm_rt(fooled(lambda r: [e for e in r[0]["evidence"] if e.startswith("tool:")]),
                {"fake://g2.jpg": {"opencv": 50}}, alert_threshold=10)
    t = loops.per_photo(photo(rt, "fake://g2.jpg"), rt)
    call = next(c for c in t.tool_calls if c.name == "propose_treatment")
    assert call.result["code"] == "downgraded" and "cites no evidence of an observation" in call.result["reason"]


def test_grounded_llm_treatment_passes():
    rt = llm_rt(fooled(lambda r: r[0]["evidence"][:1]), {"fake://g3.jpg": {"opencv": 30}}, alert_threshold=10)
    t = loops.per_photo(photo(rt, "fake://g3.jpg"), rt)
    assert t.approval is not None and t.approval.status == "pending"


def test_rule_planner_proposals_are_not_downgraded():
    rt = fake_runtime(cameras=[{**CAMERAS[0], "trap_kind": "sticky"}], devices=[FakeDevice(CAM, "camos")],
                      analyzer=FakeAnalyzer({"fake://full.jpg": {"saturation": 0.8}}))
    ev = PhotoEvent(event_id="rf", camera=CAM, source="camos", trap_kind="sticky", captured_at=rt.now(),
                    image="fake://full.jpg")
    t = loops.per_photo(ev, rt)
    assert t.approval is not None and t.approval.kind == "card_change"
    assert rt.now() - t.created_at < timedelta(seconds=1)
