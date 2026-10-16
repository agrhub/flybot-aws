"""The FlyBot -> vision Lambda contract (docs/CONTRACTS.md section 1) with a fake detector and an S3 stub."""

from __future__ import annotations

import json
import threading
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

from sentinel import handler as h
from sentinel.agent.fakes import FakeAnalyzer
from sentinel.agent.llm import FakeLLM
from sentinel.vision import detector

FIX = Path(__file__).parent / "fixtures"
CUR, PREV = FIX / "vision_be4089629_cur.jpg", FIX / "vision_be4089629_prev.jpg"
CAM = {"id": "BE4089629", "name": "Củ Chi", "trapKind": "lure", "platform": "ezviz", "location": {"lat": 10.97, "lng": 106.49}}


class FakeDetector:
    """Three boxes in the top-left quarter, scores 0.9 / 0.8 / 0.7."""

    name = "fake-yolo"
    engine_used = "new"
    onnx_path = "fake.onnx"

    def __init__(self, n: int = 3):
        self.n, self.calls = n, 0
        self.meta = {"version": "test-1"}

    def detect(self, img):
        self.calls += 1
        hgt, wid = img.shape[:2]
        boxes = [{"xywh": [10 + 30 * i, 10 + 20 * i, 24, 30], "score": round(0.9 - 0.1 * i, 2), "class_id": 0}
                 for i in range(self.n) if 10 + 30 * i + 24 < wid / 2 and 10 + 20 * i + 30 < hgt / 2]
        return {"count": len(boxes), "boxes": boxes, "forward_ms": 1.5, "output_shape": [1, 300, 6], "output_finite": True}


class FakeS3:
    def __init__(self, objects: dict[str, bytes]):
        self.objects, self.puts = dict(objects), []

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            from botocore.exceptions import ClientError

            raise ClientError({"Error": {"Code": "NoSuchKey", "Message": Key}}, "GetObject")
        return {"Body": SimpleNamespace(read=lambda n=-1: self.objects[Key])}

    def put_object(self, Bucket, Key, Body, **kw):
        self.puts.append((Bucket, Key, len(Body)))
        return {}


@pytest.fixture
def fake_det(monkeypatch):
    det = FakeDetector()
    # monkeypatch first so it records the real DETECTOR value: set_detector() writes os.environ itself, and the
    # old order made monkeypatch "restore" DETECTOR=fake, which left test_vision_pipeline without a detector
    monkeypatch.setenv("DETECTOR", "fake")
    detector.set_detector(det, "fake")
    yield det
    detector._DETECTORS.clear()
    detector.DEFAULTS.pop("fake", None)


@pytest.fixture
def no_bucket(monkeypatch, tmp_path):
    monkeypatch.delenv("PHOTOS_BUCKET", raising=False)
    monkeypatch.setenv("EVIDENCE_DIR", str(tmp_path / "evidence"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("SENTINEL_PLANNER", raising=False)
    return tmp_path / "evidence"


def request(**over):
    base = {"v": 1, "mode": "detect", "photo": {"url": CUR.as_uri()}, "camera": CAM,
            "capturedAt": "2026-10-16T08:00:00Z", "locale": "vi"}
    base.update(over)
    return base


def test_detect_mode_returns_the_contract(fake_det, no_bucket):
    out = h.analyze(request())
    assert out["v"] == 1 and out["traceId"].startswith("tr_20261016_080000_")
    assert out["model"] == {"name": "fake-yolo", "version": "test-1", "engine": f"opencv-{out['opencv']}-dnn-new", "onnx": "fake.onnx"}
    assert out["counts"] == {"total": 3, "byClass": {"fruit_fly": 3}}
    assert len(out["boxes"]) == 3 and out["boxes"][0]["cls"] == "fruit_fly" and out["boxes"][0]["score"] == 0.9
    b = out["boxes"][0]
    assert 0 <= b["x"] < 0.5 and 0 < b["w"] < 0.5 and all(0 <= b[k] <= 1 for k in ("x", "y", "w", "h"))
    assert out["density"]["gridRows"] == 4 and sum(map(sum, out["density"]["cells"])) == 3
    assert out["density"]["cells"][0][0] == 3  # all three in the top-left cell
    assert set(out["quality"]) >= {"ok", "flags", "blur", "exposure", "issues", "scores"}
    assert all(f in ("blurry", "dark", "overexposed", "occluded", "camera_moved", "trap_missing") for f in out["quality"]["flags"])
    assert out["diff"] is None  # no prevPhoto
    assert out["trace"]["planner"] == "rule" and out["trace"]["steps"][1]["tool"] == "analyze_trap_photo"
    assert out["trace"]["tokens"] == {"in": 0, "cachedIn": 0, "out": 0, "llmCalls": 0}
    assert out["timings"]["dnnMs"] == 1.5 and out["timings"]["totalMs"] > 0
    assert isinstance(out["proposals"], list) and "answer" not in out
    assert out["evidence"]["annotatedS3Key"] is None  # no bucket configured
    local = Path(out["evidence"]["annotatedPath"])
    assert local.is_file() and local.name == "2026-10-16T08-00-00Z.annotated.jpg" and local.parent.name == "BE4089629"
    json.dumps(out)  # JSON-serializable
    assert fake_det.calls == 1


def test_prev_photo_gives_a_diff(fake_det, no_bucket):
    out = h.analyze(request(prevPhoto={"url": PREV.as_uri()}))
    assert out["diff"] is not None and set(out["diff"]) >= {"newCount", "goneCount", "cardMoved"}
    assert out["diff"]["meaning"] == "activity"  # lure trap: arrivals / departures
    assert fake_det.calls == 2
    names = [s["tool"] for s in out["trace"]["steps"]]
    assert names[:2] == ["load", "analyze_trap_photo"]


def test_missing_prev_photo_is_a_warning_not_an_error(fake_det, no_bucket):
    out = h.analyze(request(prevPhoto={"url": (FIX / "nope.jpg").as_uri()}))
    assert out["diff"] is None and out["warnings"]


def test_s3_photo_and_evidence_upload(fake_det, monkeypatch, tmp_path):
    s3 = FakeS3({"tenants/t1/cams/BE4089629/p.jpg": CUR.read_bytes()})
    monkeypatch.setattr("sentinel.common.aws.client", lambda name, **kw: s3)
    monkeypatch.setenv("PHOTOS_BUCKET", "flybot-photos-test")
    monkeypatch.delenv("EVIDENCE_DIR", raising=False)
    out = h.analyze(request(photo={"s3Bucket": "flybot-photos-test", "s3Key": "tenants/t1/cams/BE4089629/p.jpg"}))
    assert out["counts"]["total"] == 3
    assert out["evidence"]["annotatedS3Key"] == "evidence/BE4089629/2026-10-16T08-00-00Z.annotated.jpg"
    assert s3.puts and s3.puts[0][:2] == ("flybot-photos-test", out["evidence"]["annotatedS3Key"])


def test_evidence_goes_to_the_request_photo_bucket(fake_det, monkeypatch):
    """FlyBot signs evidence keys against its own bucket: the photo's bucket wins over the Lambda's PHOTOS_BUCKET."""
    s3 = FakeS3({"snapshots/BE4089629/p.jpg": CUR.read_bytes()})
    monkeypatch.setattr("sentinel.common.aws.client", lambda name, **kw: s3)
    monkeypatch.setenv("PHOTOS_BUCKET", "some-other-bucket")
    monkeypatch.delenv("EVIDENCE_DIR", raising=False)
    out = h.analyze(request(photo={"s3Bucket": "flybot-photos-685527430355-dev", "s3Key": "snapshots/BE4089629/p.jpg"}))
    assert s3.puts[0][0] == "flybot-photos-685527430355-dev" and out["evidence"]["bucket"] == "flybot-photos-685527430355-dev"
    assert out["evidence"]["annotatedS3Key"].startswith("evidence/BE4089629/")


def test_optional_request_fields_from_flybot_are_accepted(fake_det, no_bucket):
    """history, camera.battery, camera.isp, photoId (what FlyBot's visionLambda.js may send) never cause a bad_request."""
    cam = {**CAM, "platform": "camos", "battery": 71.5, "isp": {"brightness": 0, "contrast": 50, "wdr": 0, "flicker": 3},
           "alertThreshold": 20}
    hist = [{"capturedAt": "2026-10-15T08:00:00Z", "count": 2, "photoId": "6ac3"}, {"capturedAt": "bad", "count": None}, "junk"]
    out = h.analyze(request(camera=cam, history=hist, photoId="6ac327d44c3227d11cfbd084"))
    assert out["counts"]["total"] == 3 and not out.get("error")


def test_set_isp_only_for_camos_cameras(fake_det, no_bucket):
    analyzer = FakeAnalyzer({CUR.as_uri(): {"issues": ["glare"], "opencv": 2, "glare_ratio": 0.3}})
    for platform in ("ezviz", "kinesis"):
        out = h.analyze(request(camera={**CAM, "platform": platform, "trapKind": "sticky"}, locale="en"), analyzer=analyzer)
        acts = [p["action"] for p in out["proposals"]]
        assert "set_isp" not in acts and "capture_now" in acts, (platform, out["proposals"])


def test_photo_not_found_on_s3(fake_det, monkeypatch, no_bucket):
    s3 = FakeS3({})
    monkeypatch.setattr("sentinel.common.aws.client", lambda name, **kw: s3)
    with pytest.raises(h.PhotoNotFound) as ei:
        h.analyze(request(photo={"s3Bucket": "b", "s3Key": "missing.jpg"}))
    err = json.loads(str(ei.value))
    assert err == {"v": 1, "error": {"code": "photo_not_found", "message": err["error"]["message"]}}
    assert "missing.jpg" in err["error"]["message"]


def test_photo_not_found_on_http_404(fake_det, monkeypatch, no_bucket):
    import httpx

    monkeypatch.setattr(httpx, "get", lambda url, **kw: httpx.Response(404, request=httpx.Request("GET", url)))
    with pytest.raises(h.PhotoNotFound):
        h.analyze(request(photo={"url": "https://example.invalid/x.jpg"}))


def test_bad_requests(fake_det, no_bucket):
    for bad, needle in [({"camera": {}}, "camera.id"), ({"mode": "fast"}, "mode"), ({"photo": {}}, "photo"),
                        ({"camera": {**CAM, "trapKind": "glue"}}, "trapKind"), ({"v": 2}, "version"),
                        ({"mode": "deep"}, "question"), ({"locale": "fr"}, "locale")]:
        with pytest.raises(h.BadRequest) as ei:
            h.analyze(request(**bad))
        assert needle in str(ei.value)
    with pytest.raises(h.BadRequest):
        h.analyze("not json")
    with pytest.raises(h.BadRequest):  # a file that is not an image
        h.analyze(request(photo={"url": Path(__file__).as_uri()}))


def test_model_unavailable(monkeypatch, no_bucket):
    detector._DETECTORS.clear()
    monkeypatch.setenv("DETECTOR", "yolo26")
    monkeypatch.setenv("DETECTOR_ONNX", str(FIX / "does-not-exist.onnx"))
    with pytest.raises(h.ModelUnavailable) as ei:
        h.analyze(request())
    assert json.loads(str(ei.value))["error"]["code"] == "model_unavailable"
    detector._DETECTORS.clear()


def test_handler_wraps_unexpected_errors_as_internal(fake_det, no_bucket, monkeypatch):
    monkeypatch.setattr(h, "analyze", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(h.Internal) as ei:
        h.handler(request())
    assert json.loads(str(ei.value))["error"]["code"] == "internal"


def test_quality_issue_becomes_a_proposal_in_vietnamese(fake_det, no_bucket):
    analyzer = FakeAnalyzer({CUR.as_uri(): {"issues": ["dark"], "opencv": 3, "brightness": 30.0}})
    out = h.analyze(request(), analyzer=analyzer)
    acts = [p["action"] for p in out["proposals"]]
    assert acts == ["capture_now"], out["proposals"]
    p = out["proposals"][0]
    assert p["requiresApproval"] is False and p["confidence"] == 0.9 and p["rule"] == "quality.recapture"
    assert "chụp lại" in p["reason"] and "dark" in p["detail"] and p["params"]["purpose"] == "recapture"
    assert "dark" in out["quality"]["flags"]
    assert any(d["rule"] == "quality.awaiting_photo" for d in out["trace"]["decisions"])


def test_camos_camera_gets_isp_fix_and_verification_capture(fake_det, no_bucket):
    analyzer = FakeAnalyzer({CUR.as_uri(): {"issues": ["glare"], "opencv": 2, "glare_ratio": 0.3}})
    cam = {**CAM, "platform": "camos", "trapKind": "sticky", "isp": {"brightness": 0, "contrast": 50, "wdr": 0, "flicker": 3}}
    out = h.analyze(request(camera=cam, locale="en"), analyzer=analyzer)
    acts = [(p["action"], p["by"]) for p in out["proposals"]]
    assert acts == [("set_isp", "planner"), ("capture_now", "guardrail")], out["proposals"]
    isp = out["proposals"][0]
    assert isp["params"] == {"brightness": -16, "wdr": True} and isp["requiresApproval"] is True
    assert isp["reason"].startswith("Glare or dark photo")
    assert out["proposals"][1]["params"]["purpose"] == "verify_isp"


def test_history_surge_becomes_a_notify_proposal(fake_det, no_bucket):
    analyzer = FakeAnalyzer({CUR.as_uri(): {"opencv": 40}})
    hist = [{"capturedAt": f"2026-10-{d:02d}T08:00:00Z", "count": 4} for d in range(10, 16)]
    out = h.analyze(request(history=hist), analyzer=analyzer)
    acts = [p["action"] for p in out["proposals"]]
    assert "notify" in acts, out["proposals"]
    n = next(p for p in out["proposals"] if p["action"] == "notify")
    assert n["rule"] == "surge.unconfirmed" and n["params"]["kind"] == "recheck_surge" and n["requiresApproval"] is False
    assert "surge" in out["trace"]["outcome"]


def test_deep_mode_uses_the_llm_and_returns_an_answer(fake_det, no_bucket):
    llm = FakeLLM([
        {"text": "looking", "tool_uses": [{"name": "analyze_trap_photo", "input": {"camera": "BE4089629"}}]},
        {"text": "**3 ruồi** trên bẫy; chất lượng ảnh tốt; không có thay đổi lớn."},
    ])
    out = h.analyze(request(mode="deep", question="Why did the count jump today?"), llm=llm)
    assert out["trace"]["planner"] == "llm" and out["trace"]["fallback"] is False
    assert out["answer"].startswith("**3 ruồi**")
    assert "Grower's question" in llm.conversations[0].user and "count jump" in llm.conversations[0].user
    assert out["trace"]["llmTurns"] == 2


def test_deep_mode_falls_back_to_the_rule_planner_and_a_summary(fake_det, no_bucket):
    llm = FakeLLM([RuntimeError("provider down")])
    out = h.analyze(request(mode="deep", question="Tại sao hôm nay tăng?", locale="vi"), llm=llm)
    assert out["trace"]["planner"] == "llm" and out["trace"]["fallback"] is True
    assert "3 ruồi" in out["answer"] and "quy tắc" in out["answer"].lower()


def test_deep_mode_without_an_llm_key_still_answers(fake_det, no_bucket):
    out = h.analyze(request(mode="deep", question="How many?", locale="en"))
    assert out["trace"]["planner"] == "rule" and "3 flies" in out["answer"]


def test_kill_switch_analyzes_but_proposes_nothing(fake_det, no_bucket, monkeypatch):
    monkeypatch.setenv("SENTINEL_KILL_SWITCH", "1")
    analyzer = FakeAnalyzer({CUR.as_uri(): {"issues": ["dark"], "opencv": 1}})
    out = h.analyze(request(), analyzer=analyzer)
    assert out["counts"]["total"] == 1 and out["proposals"] == [] and out["trace"]["killSwitch"] is True


def test_lambda_local_serves_the_same_contract(fake_det, no_bucket):
    from http.server import ThreadingHTTPServer

    from sentinel import lambda_local

    srv = ThreadingHTTPServer(("127.0.0.1", 0), lambda_local.Handler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=10) as r:
            assert json.loads(r.read())["ok"] is True
        body = json.dumps(request()).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/invoke", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            out = json.loads(r.read())
        assert out["counts"]["total"] == 3
        bad = urllib.request.Request(f"http://127.0.0.1:{port}/invoke", data=b'{"v":1}', headers={"Content-Type": "application/json"})
        with pytest.raises(urllib.error.HTTPError) as ei:
            urllib.request.urlopen(bad, timeout=10)
        assert ei.value.code == 400 and json.loads(ei.value.read())["error"]["code"] == "bad_request"
    finally:
        srv.shutdown()


def test_lambda_local_refuses_public_bind(monkeypatch):
    from sentinel import lambda_local

    monkeypatch.delenv("LAMBDA_LOCAL_ALLOW_REMOTE", raising=False)
    assert lambda_local.main(["--host", "0.0.0.0", "--port", "0"]) == 2
