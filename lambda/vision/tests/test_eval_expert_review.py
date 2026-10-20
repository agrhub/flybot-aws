"""Expert review: answers from the FlyBot review page go into the sheet (eval/import_expert_review.py), the sheet is scored
(eval/score_expert_review.py) and rendered (eval/run_eval.py section 6), with the optional English note column."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]  # repo root (lambda/vision/tests)
sys.path.insert(0, str(ROOT))

from eval import import_expert_review as imp
from eval import score_expert_review as ser
from eval.run_eval import render_expert

SHEET = ROOT / "docs" / "evidence" / "eval" / "expert_review.csv"


@pytest.fixture
def sheet(tmp_path: Path) -> Path:
    dst = tmp_path / "expert_review.csv"
    shutil.copy(SHEET, dst)
    return dst


def _fill(sheet: Path, answers: list[dict], translations: dict[str, str]) -> dict:
    rows = imp.read_rows(sheet)
    res = imp.apply(rows, answers, translations, reviewer="Test Reviewer")
    imp.write_rows(sheet, rows)
    return res


def test_a_sheet_without_the_english_column_still_scores(sheet: Path):
    res = ser.score(sheet)
    assert res["verdicts"] == 0 and "not measured" in res["status"]
    assert res["agent_example_rows_excluded"] == 5 and res["rows_scored_pool"] == 50


def test_answers_are_imported_scored_and_rendered(sheet: Path):
    answers = [{"id": f"AWS-{i:02d}", "verdict": "correct", "note": ""} for i in range(1, 9)]
    answers += [{"id": "AWS-09", "verdict": "incorrect", "note": "ảnh rõ, không cần chụp lại"},
                {"id": "RPL-01", "verdict": "unsure", "note": "không thấy rõ ruồi"},
                {"id": "STG-01", "verdict": "correct", "note": "đề xuất chờ duyệt là đúng"}]
    en = {"AWS-09": "the photo is clear, no recapture needed", "RPL-01": "cannot see the flies clearly",
          "STG-01": "a proposal awaiting approval is right"}
    res = _fill(sheet, answers, en)
    assert res["problems"] == [] and len(res["written"]) == 11

    rows = {r["id"]: r for r in imp.read_rows(sheet)}
    assert list(rows)[:2] == ["AWS-01", "AWS-02"] and "expert_note_en" in rows["AWS-01"]
    assert rows["AWS-09"]["expert_note"] == "reviewer=Test Reviewer; ảnh rõ, không cần chụp lại"
    assert rows["AWS-09"]["expert_note_en"] == "the photo is clear, no recapture needed"
    assert rows["AWS-01"]["expert_note"] == "reviewer=Test Reviewer;"
    assert rows["EX-1"]["expert_note"].startswith("reviewer=agent-example")

    s = ser.score(sheet)
    assert s["status"] == "measured" and s["verdicts"] == 11 and s["agent_example_rows_excluded"] == 5
    o = s["overall"]
    assert (o["correct"], o["incorrect"], o["unsure"]) == (9, 1, 1)
    assert o["policy_correct_rate"] == 0.9 and o["worst_case_rate_unsure_as_incorrect"] == round(9 / 11, 4)
    lo, hi = o["wilson95"]
    assert 0.5 < lo < 0.9 < hi <= 1.0
    assert s["incorrect_rows"][0]["note_en"] == "the photo is clear, no recapture needed"
    assert s["unsure_rows"][0]["id"] == "RPL-01"

    md = "\n".join(render_expert(s))
    assert "| all | 9 | 1 | 1 | 0.9 |" in md
    assert "AWS-09 (aws_live" in md and "the photo is clear, no recapture needed" in md
    assert "translated afterwards by the team" in md


def test_bad_answers_write_nothing(sheet: Path):
    rows = imp.read_rows(sheet)
    res = imp.apply(rows, [{"id": "AWS-01", "verdict": "incorrect", "note": ""},
                           {"id": "AWS-02", "verdict": "correct", "note": "có ghi chú"},
                           {"id": "EX-1", "verdict": "correct"},
                           {"id": "AWS-03", "verdict": "maybe"}], {})
    assert len(res["problems"]) == 4 and res["written"] == []


def test_an_existing_verdict_needs_force(sheet: Path):
    _fill(sheet, [{"id": "FIJ-01", "verdict": "correct"}], {})
    rows = imp.read_rows(sheet)
    res = imp.apply(rows, [{"id": "FIJ-01", "verdict": "unsure"}], {})
    assert res["written"] == [] and "already has a verdict" in res["skipped"][0]
    assert imp.apply(rows, [{"id": "FIJ-01", "verdict": "unsure"}], {}, force=True)["written"] == ["FIJ-01"]
