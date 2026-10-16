"""models/fetch_models.py: SHA-256-checked download into the git-ignored model directories (no network here)."""

import hashlib
import importlib.util
import io
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]  # repo root (lambda/vision/tests)
_spec = importlib.util.spec_from_file_location("fetch_models", ROOT / "models" / "fetch_models.py")
fm = importlib.util.module_from_spec(_spec)
sys.modules["fetch_models"] = fm  # dataclasses look the module up while the class is created
_spec.loader.exec_module(fm)

PAYLOAD = b"onnx-bytes" * 1000


def _model(tmp_dest="models/features", payload=PAYLOAD, **kw):
    return fm.Model("features", "x.onnx", tmp_dest, hashlib.sha256(payload).hexdigest(), len(payload),
                    kw.pop("urls", ["https://mirror.invalid/x.onnx"]), "test", **kw)


def _opener(payload, calls):
    def urlopen(req, timeout=None):
        calls.append(req.full_url)
        return io.BytesIO(payload)
    return urlopen


def test_fetch_verifies_and_skips_valid_file(tmp_path):
    m, calls = _model(), []
    status, path, src = fm.fetch_one(m, tmp_path, urlopen=_opener(PAYLOAD, calls))
    assert status == "fetched" and path.read_bytes() == PAYLOAD and src == m.urls[0]
    assert not path.with_name("x.onnx.part").exists()
    status, _, _ = fm.fetch_one(m, tmp_path, urlopen=_opener(PAYLOAD, calls))
    assert status == "ok" and len(calls) == 1  # second run: hash matches, no download


def test_wrong_hash_is_rejected_and_not_left_on_disk(tmp_path):
    m, calls = _model(), []
    status, path, _ = fm.fetch_one(m, tmp_path, urlopen=_opener(b"x" * len(PAYLOAD), calls))
    assert status.startswith("FAILED") and "sha256" in status
    assert not path.exists() and not path.with_name("x.onnx.part").exists()


def test_falls_through_to_next_url_and_https_only(tmp_path):
    m = _model(urls=["http://insecure.invalid/x.onnx", "https://bad.invalid/x.onnx", "https://good.invalid/x.onnx"])
    calls = []

    def urlopen(req, timeout=None):
        calls.append(req.full_url)
        return io.BytesIO(PAYLOAD if "good" in req.full_url else b"short")

    status, _, src = fm.fetch_one(m, tmp_path, urlopen=urlopen)
    assert status == "fetched" and src.startswith("https://good")
    assert all(u.startswith("https://") for u in calls)


def test_vendored_copy_wins_and_check_mode_has_no_network(tmp_path):
    (tmp_path / "vend").mkdir()
    (tmp_path / "vend" / "x.onnx").write_bytes(PAYLOAD)
    m = _model(vendored="vend/x.onnx")
    assert fm.fetch_one(m, tmp_path, urlopen=None)[0] == "vendored"
    m2 = _model()

    def no_net(*a, **k):
        raise AssertionError("network used in --check")

    assert fm.fetch_one(m2, tmp_path, check=True, urlopen=no_net)[0].startswith("FAILED: missing")


@pytest.mark.parametrize("m", fm.catalog(), ids=lambda m: m.file)
def test_catalog_entries_are_pinned(m):
    assert len(m.sha256) == 64 and int(m.sha256, 16) >= 0 and m.size > 0
    assert m.urls and all(u.startswith("https://") for u in m.urls)
    assert m.dest.startswith("models/") or Path(m.dest).is_absolute()
