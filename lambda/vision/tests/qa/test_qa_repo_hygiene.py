"""Repository hygiene checks for the submission (A9, docs/QA_ROUND1.md).

Read-only: they only run `git check-ignore` / `git ls-files`, and skip when git or the repo metadata is missing.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]  # repo root (lambda/vision/tests/qa)
pytestmark = pytest.mark.skipif(shutil.which("git") is None or not (ROOT / ".git").exists(),
                                reason="needs a git checkout")

# Local secrets and git-ignored large artifacts: must stay ignored and untracked.
SECRET_OR_LARGE = [".env", "data/secrets/x.txt", "data/export/x.json", "data/cache/x.json", "data/manifest.csv",
                   "models/yolov7-tiny/fruitfly_deploy.onnx", "models/features/x.onnx"]
MAX_BYTES = 2_000_000  # CLAUDE.md: no file over 2 MB in git


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)


@pytest.mark.parametrize("path", SECRET_OR_LARGE)
def test_secret_and_large_paths_are_ignored_and_untracked(path):
    assert git("check-ignore", "-q", "--no-index", path).returncode == 0, f"{path} is not git-ignored"
    assert git("ls-files", "--", path).stdout.strip() == "", f"{path} is tracked"


def test_no_tracked_file_over_2mb():
    big = []
    for rel in git("ls-files", "-z").stdout.split("\0"):
        p = ROOT / rel
        if rel and p.is_file() and p.stat().st_size > MAX_BYTES:
            big.append(f"{rel} ({p.stat().st_size} B)")
    assert not big, big


def test_sources_cited_by_results_md_are_tracked():  # QA_ROUND1 B1 (fixed in a46f4df): keep it fixed
    text = (ROOT / "eval" / "RESULTS.md").read_text(encoding="utf-8")
    cited = sorted(set(re.findall(r"eval/results/[A-Za-z0-9_./-]+\.(?:json|csv)", text)))
    assert cited, "RESULTS.md cites no eval/results file"
    tracked = set(git("ls-files", "eval/results").stdout.split())
    missing = [c for c in cited if c not in tracked]
    assert not missing, f"cited but not in git: {missing}"


def test_sources_cited_by_cool_report_are_tracked():
    text = (ROOT / "bench" / "REPORT_COOL.md").read_text(encoding="utf-8")
    cited = sorted(set(re.findall(r"bench/results/[A-Za-z0-9_./-]+\.json", text)) - {"bench/results/<config>.json"})
    tracked = set(git("ls-files", "bench/results").stdout.split())
    missing = [c for c in cited if c not in tracked]
    assert not missing, f"cited but not in git: {missing}"
