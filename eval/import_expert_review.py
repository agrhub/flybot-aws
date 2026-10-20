"""Write the advisor's answers from the FlyBot expert review page into docs/evidence/eval/expert_review.csv.

    uv run python -m eval.import_expert_review --answers answers.json --translations notes_en.json
    uv run python -m eval.import_expert_review --url https://flybot.agrhub.com/api/public/expert-review/<token> --translations notes_en.json

The page (FlyBot client/public/review/<token>/) shows the same 50 rows in Vietnamese; its answers come from
GET /api/public/expert-review/<token> as {"answers": [{"id", "verdict": correct|incorrect|unsure, "note", "t"}]}.
For each answered row this fills:
    expert_verdict   the verdict code, as given
    expert_note      `reviewer=<name>; ` + the reviewer's note verbatim (Vietnamese)
    expert_note_en   the English translation of that note, from --translations ({id: text}), made afterwards by the team
The first eight columns and the row order are kept; the agent-example rows (EX-*) are never touched. A row that already
has a reviewer's verdict is only replaced with --force.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.request
from pathlib import Path

from eval.score_expert_review import COLUMNS, CSV_PATH, OPTIONAL_COLUMNS, is_example, normalize

REVIEWER = "Trương Quốc Ánh"
FIELDS = COLUMNS + OPTIONAL_COLUMNS


def load_answers(path: Path | None = None, url: str | None = None) -> list[dict]:
    if url:
        with urllib.request.urlopen(url, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    else:
        body = json.loads(Path(path).read_text(encoding="utf-8"))
    return body["answers"] if isinstance(body, dict) else body


def apply(rows: list[dict], answers: list[dict], translations: dict[str, str], reviewer: str = REVIEWER,
          force: bool = False) -> dict:
    """Fill the rows in place; returns what was written and what was skipped."""
    by_id = {r["id"]: r for r in rows}
    written, skipped, problems = [], [], []
    for a in answers:
        rid, verdict, note = str(a.get("id", "")), normalize(a.get("verdict")), (a.get("note") or "").strip()
        row = by_id.get(rid)
        if row is None or is_example(row):
            problems.append(f"{rid}: not a row to review")
            continue
        if verdict is None:
            skipped.append(f"{rid}: no verdict")
            continue
        if verdict.startswith("invalid:"):
            problems.append(f"{rid}: invalid verdict {verdict[8:]!r}")
            continue
        if verdict == "incorrect" and not note:
            problems.append(f"{rid}: incorrect without a note")
            continue
        if note and not (translations.get(rid) or "").strip():
            problems.append(f"{rid}: note without an English translation")
            continue
        if (row.get("expert_verdict") or "").strip() and not force:
            skipped.append(f"{rid}: already has a verdict (use --force)")
            continue
        row["expert_verdict"] = verdict
        row["expert_note"] = f"reviewer={reviewer};" + (f" {note}" if note else "")
        row["expert_note_en"] = (translations.get(rid) or "").strip()
        written.append(rid)
    return {"written": written, "skipped": skipped, "problems": problems}


def read_rows(csv_path: Path) -> list[dict]:
    with open(csv_path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r.setdefault("expert_note_en", "")
        r["expert_note_en"] = r["expert_note_en"] or ""
    return rows


def write_rows(csv_path: Path, rows: list[dict]) -> None:
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--answers", type=Path, help="JSON saved from the review page's GET endpoint")
    src.add_argument("--url", help="the review page's GET endpoint")
    ap.add_argument("--translations", type=Path, help="JSON {row id: English translation of the note}")
    ap.add_argument("--csv", type=Path, default=CSV_PATH)
    ap.add_argument("--reviewer", default=REVIEWER)
    ap.add_argument("--force", action="store_true", help="replace verdicts already in the sheet")
    a = ap.parse_args(argv)
    translations = json.loads(a.translations.read_text(encoding="utf-8")) if a.translations else {}
    rows = read_rows(a.csv)
    res = apply(rows, load_answers(a.answers, a.url), translations, a.reviewer, a.force)
    print(json.dumps(res, indent=1, ensure_ascii=False))
    if res["problems"]:
        print("nothing written: fix the problems above", file=sys.stderr)
        return 1
    write_rows(a.csv, rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
