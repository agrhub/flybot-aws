"""Score the expert review (T4.3): policy-correct rate once an AgriHub agronomist has filled `expert_verdict`.

    uv run python -m eval.score_expert_review                       # docs/evidence/eval/expert_review.csv -> eval/results/expert_review.json
    uv run python -m eval.score_expert_review path/to/filled.csv --out /tmp/x.json

Verdicts (column `expert_verdict`, case-insensitive; synonyms are mapped):
    correct    the agent's decision is what a careful agronomist would also have done with this evidence
    incorrect  it is not (wrong action, missed problem, needless alarm or action): say what in `expert_note`
    unsure     the evidence does not let the reviewer decide (excluded from the rate, counted separately)
Empty = not reviewed yet.

Policy-correct rate = correct / (correct + incorrect), with a 95 % Wilson interval. `unsure` is not counted as correct: the
"worst case" rate counts every `unsure` as incorrect and is reported next to it.
Rows whose `expert_note` starts with `reviewer=agent-example` are worked examples written by the agent that built the sheet:
they are shown to the expert as a format guide and are EXCLUDED from every number here.
Nothing is filled in: with 0 verdicts the result says "0 verdicts, not measured".
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "docs" / "evidence" / "eval" / "expert_review.csv"
OUT = ROOT / "eval" / "results" / "expert_review.json"
COLUMNS = ["id", "source", "camera", "captured_at", "evidence_link", "opencv_summary", "agent_decision", "agent_reason",
           "expert_verdict", "expert_note"]
EXAMPLE_TAG = "reviewer=agent-example"
SYNONYMS = {"correct": "correct", "c": "correct", "yes": "correct", "y": "correct", "ok": "correct", "agree": "correct",
            "incorrect": "incorrect", "i": "incorrect", "wrong": "incorrect", "no": "incorrect", "n": "incorrect", "disagree": "incorrect",
            "unsure": "unsure", "u": "unsure", "?": "unsure", "unclear": "unsure", "cannot_tell": "unsure"}


def normalize(v: str | None) -> str | None:
    """'correct' | 'incorrect' | 'unsure' | None (empty) | 'invalid:<text>'."""
    t = (v or "").strip().lower()
    if not t:
        return None
    return SYNONYMS.get(t, f"invalid:{t}")


def is_example(row: dict) -> bool:
    return str(row.get("expert_note") or "").strip().lower().startswith(EXAMPLE_TAG)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return round((c - h) / d, 4), round((c + h) / d, 4)


def _rate(c: Counter) -> dict:
    k, i, u = c["correct"], c["incorrect"], c["unsure"]
    n = k + i
    return {"correct": k, "incorrect": i, "unsure": u, "judged": n,
            "policy_correct_rate": round(k / n, 4) if n else None,
            "wilson95": wilson(k, n),
            "worst_case_rate_unsure_as_incorrect": round(k / (n + u), 4) if (n + u) else None}


def score(csv_path: Path = CSV_PATH) -> dict:
    if not csv_path.exists():
        return {"status": "not measured: sheet missing", "source": str(csv_path)}
    with open(csv_path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    missing = [c for c in COLUMNS if rows and c not in rows[0]]
    if missing:
        return {"status": f"not measured: columns missing {missing}", "source": str(csv_path)}
    try:
        shown = csv_path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        shown = str(csv_path)
    examples = [r for r in rows if is_example(r)]
    real = [r for r in rows if not is_example(r)]
    allc, by_src, by_dec, by_cam = Counter(), defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    pending, invalid, ids = 0, [], Counter(r["id"] for r in rows)
    incorrect_rows = []
    for r in real:
        v = normalize(r.get("expert_verdict"))
        if v is None:
            pending += 1
            continue
        if v.startswith("invalid:"):
            invalid.append({"id": r["id"], "verdict": v[8:]})
            continue
        allc[v] += 1
        by_src[r["source"]][v] += 1
        by_dec[r["agent_decision"]][v] += 1
        by_cam[r["camera"]][v] += 1
        if v == "incorrect":
            incorrect_rows.append({"id": r["id"], "source": r["source"], "decision": r["agent_decision"], "note": r.get("expert_note", "")})
    verdicts = sum(allc.values())
    out = {"source": shown, "rows": len(rows), "rows_scored_pool": len(real), "agent_example_rows_excluded": len(examples),
           "duplicate_ids": [k for k, v in ids.items() if v > 1], "pending": pending, "invalid_verdicts": invalid,
           "rows_per_source": dict(Counter(r["source"] for r in real))}
    if verdicts == 0:
        out["status"] = f"0 verdicts, not measured ({pending} rows waiting for the expert, {len(examples)} agent-example rows excluded)"
        out["verdicts"] = 0
        return out
    out.update({"status": "measured", "verdicts": verdicts, "overall": _rate(allc),
                "by_source": {k: _rate(v) for k, v in sorted(by_src.items())},
                "by_decision": {k: _rate(v) for k, v in sorted(by_dec.items())},
                "by_camera": {k: _rate(v) for k, v in sorted(by_cam.items())},
                "incorrect_rows": incorrect_rows})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csv", nargs="?", type=Path, default=CSV_PATH)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    res = score(a.csv)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
