"""Before / after the lure-floor glare limit (quality_thresholds.yaml `glare.floor_ratio_max`), same inputs.

    cd lambda/vision && uv run python ../../eval/glare_rule_ab.py [--skip failure synth export]   # -> eval/results/glare_rule_ab.json

`before` judges glare on a lure tray floor against the sticky-card / frame limit `glare.ratio_max` (the rule up to
this change); `after` is the thresholds file as it is. Nothing else differs between the two runs. Parts:

- `expert_sheet`: the photos of docs/evidence/eval/expert_review.csv (rows with a verdict, agent examples excluded)
  through the in-process handler (`sentinel.handler.analyze`, rule planner, EZVIZ lure trap, the row's captured_at, no
  prevPhoto, no history). A row recaptures when the response proposes `capture_now`. The advisor judged the sheet's
  decisions (earlier pipeline, 14-17 Oct) only; nobody reviewed the re-scored ones.
- `sample_replay`: data/samples (24 real photos, two cameras) through the handler as `make replay-sample` sends them.
- `failure_injection`: eval/failure_injection.py cases (3 photo pairs) and its op 2 strength sweep (40 photos), with
  the pairs and the sweep pool picked once under the `before` rule so both runs see the same frames.
- `export`: op 2 alone on the FlyBot export (data/manifest.csv, git-ignored; skipped when absent).
- `synthetic`: op 2 alone on the synthetic sets (data/synth/{demo-30d,pool}, git-ignored; skipped when absent).
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import sys
import tempfile
import time
import types
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "lambda" / "vision"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from sentinel.vision import quality
from sentinel.vision.card import find_card

OUT = ROOT / "eval" / "results" / "glare_rule_ab.json"
SHEET = ROOT / "docs" / "evidence" / "eval" / "expert_review.csv"
SAMPLES = ROOT / "data" / "samples" / "samples_manifest.csv"
EXPORT = ROOT / "data" / "manifest.csv"
SYNTH_SETS = ("demo-30d", "pool")
EXAMPLE_TAG = "reviewer=agent-example"


def rel(p: Path) -> str:
    try:
        return Path(p).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


def rule_thresholds() -> dict[str, dict]:
    after = quality.load_thresholds()
    before = {**after, "glare": {**after["glare"], "floor_ratio_max": after["glare"]["ratio_max"]}}
    return {"before": before, "after": after}


@contextlib.contextmanager
def thresholds_file(t: dict):
    """Point quality.load_thresholds() (handler, agent re-judge, eval helpers) at a copy of `t`."""
    saved = quality.THRESHOLDS_PATH
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "quality_thresholds.yaml"
        path.write_text(yaml.safe_dump(t), encoding="utf-8")
        quality.THRESHOLDS_PATH = path
        try:
            yield
        finally:
            quality.THRESHOLDS_PATH = saved


def summary_of(res: dict) -> dict:
    q = res["quality"]
    return {"issues": q["issues"], "flags": q["flags"], "glare_ratio": q["scores"].get("glare_ratio"),
            "glare_region": q["scores"].get("glare_region"), "proposals": sorted({p["action"] for p in res["proposals"]}),
            "recapture": any(p["action"] == "capture_now" for p in res["proposals"])}


# --------------------------------------------------------------------------- expert sheet

def expert_sheet(rules: dict) -> dict:
    from sentinel import handler as h

    with open(SHEET, encoding="utf-8", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if (r.get("expert_verdict") or "").strip()
                and not (r.get("expert_note") or "").startswith(EXAMPLE_TAG) and (ROOT / r["evidence_link"]).is_file()]

    def req(r):
        return {"v": 1, "mode": "detect", "photo": {"url": (ROOT / r["evidence_link"]).resolve().as_uri()},
                "camera": {"id": r["camera"], "trapKind": "lure", "platform": "ezviz"}, "capturedAt": r["captured_at"],
                "history": [], "locale": "vi", "photoId": f"ab-{r['id']}"}

    runs = {}
    for name, t in rules.items():
        with thresholds_file(t):
            runs[name] = {r["id"]: summary_of(h.analyze(req(r))) for r in rows}
    per_row = []
    for r in rows:
        b, a = runs["before"][r["id"]], runs["after"][r["id"]]
        per_row.append({"id": r["id"], "source": r["source"], "camera": r["camera"], "verdict": r["expert_verdict"].strip().lower(),
                        "sheet_decision": r["agent_decision"],
                        # capture / *.recapture in the sheet's decision (an ISP change alone is not counted)
                        "sheet_recapture": "capture" in (r["agent_decision"] or "").lower(),
                        "before": b, "after": a, "changed": b["proposals"] != a["proposals"] or b["issues"] != a["issues"]})

    def group(sel) -> dict:
        g = [p for p in per_row if sel(p)]
        return {"rows": len(g), "sheet_recapture": sum(p["sheet_recapture"] for p in g),
                "recapture_before": sum(p["before"]["recapture"] for p in g),
                "recapture_after": sum(p["after"]["recapture"] for p in g),
                "recapture_after_ids": [p["id"] for p in g if p["after"]["recapture"]],
                "changed": [p["id"] for p in g if p["changed"]],
                "issues_before": dict(Counter(i for p in g for i in p["before"]["issues"])),
                "issues_after": dict(Counter(i for p in g for i in p["after"]["issues"]))}

    return {"source": rel(SHEET), "harness": ("sentinel.handler.analyze in-process, rule planner, EZVIZ lure trap, captured_at "
                                              "from the sheet, no prevPhoto, no history"),
            "not_a_review": "The advisor judged the sheet's decisions only; nobody reviewed the re-scored decisions",
            "rejected_or_unsure": group(lambda p: p["verdict"] in ("incorrect", "unsure")),
            "judged_correct": group(lambda p: p["verdict"] == "correct"), "rows": per_row}


# --------------------------------------------------------------------------- bundled sample (make replay-sample)

def sample_replay(rules: dict) -> dict:
    from eval import replay_sample as rs

    from sentinel import handler as h

    rows = rs.rows_of(SAMPLES)
    out = {}
    for name, t in rules.items():
        last: dict[str, dict] = {}
        hist: dict[str, list[dict]] = {}
        res = []
        with thresholds_file(t):
            for row in rows:
                cam = row["device_serial"]
                res.append(summary_of(h.analyze(rs.request_for(row, last.get(cam), hist.get(cam, [])[-14 * 48:], SAMPLES.parent))))
                last[cam] = row
                if row.get("ai_count") not in (None, ""):
                    hist.setdefault(cam, []).append({"capturedAt": row["captured_at"], "count": int(float(row["ai_count"])),
                                                     "snapshotId": row["snapshot_id"]})
        out[name] = {"photos": len(res), "issues": dict(Counter(i for r in res for i in r["issues"])),
                     "flags": dict(Counter(f for r in res for f in r["flags"])),
                     "proposals": dict(Counter(p for r in res for p in r["proposals"])),
                     "capture_now": sum(r["recapture"] for r in res),
                     "glare_ratio_of_glare_flags": sorted(r["glare_ratio"] for r in res if "glare" in r["issues"])}
    return {"source": rel(SAMPLES), "harness": "eval/replay_sample.py requests (prevPhoto, history) to sentinel.handler.analyze", **out}


# --------------------------------------------------------------------------- failure injection + op 2 sweep

def _failure_injection_module():
    """eval/failure_injection.py imports IoT test doubles (FakeIotCloud, MemoryTable, sentinel.devices.camos_iot)
    that flybot-aws does not have. Stand-ins that raise on use: the three cases built on them (rpc_timeout,
    shadow_reported_errors, shadow_not_applied) fail with that error in BOTH runs; every other case runs as written."""
    from sentinel.agent import fakes

    class NotInThisRepo:
        def __init__(self, *a, **k):
            raise RuntimeError("IoT test double not in flybot-aws (eval/failure_injection.py needs CamosIotDevice)")

    for name in ("FakeIotCloud", "MemoryTable"):
        if not hasattr(fakes, name):
            setattr(fakes, name, NotInThisRepo)
    try:
        import sentinel.devices.camos_iot  # noqa: F401
    except ImportError:
        mod = types.ModuleType("sentinel.devices.camos_iot")
        mod.CamosIotDevice = NotInThisRepo
        sys.modules["sentinel.devices.camos_iot"] = mod
    from eval import failure_injection as fi

    return fi


def failure_injection(rules: dict, photos: int = 3, sweep_n: int = 40) -> dict:
    fi = _failure_injection_module()
    with thresholds_file(rules["before"]):
        pairs, pool = fi.pick_pairs(photos), fi.pick_pool(sweep_n)
    fi.pick_pairs, fi.pick_pool = (lambda n, **k: pairs), (lambda n, **k: pool)
    out = {}
    for name, t in rules.items():
        with thresholds_file(t):
            r = fi.run_all(photos=photos, sweep_n=sweep_n, evidence=False)
        cases = [{"case": c["case"], "photo": c.get("photo"), "pass": c["pass"], "false_actuations": c.get("false_actuations"),
                  "observed": (c.get("observed") or "")[:240]} for c in r["cases"]]
        sw = r["sweep"]
        out[name] = {"passed": r["passed"], "total": r["total"], "false_actuations_total": r["false_actuations_total"],
                     "failed": [c["case"] for c in cases if not c["pass"]], "cases": cases,
                     "sweep": {"n_photos": sw["n_photos"], "cameras": sw["cameras"], "clean_false_positive": sw["clean_false_positive"],
                               "rates": {k: [{"level": lv["level"], "unit": lv["unit"],
                                              "no_baseline": lv["no_baseline"]["flag_rate"], "with_baseline": lv["with_baseline"]["flag_rate"]}
                                             for lv in v["levels"]] for k, v in sw["sweeps"].items()}}}
    return {"harness": "eval/failure_injection.py run_all (pairs and sweep pool picked once under the before rule)",
            "photos": [{"camera": p["camera"], "cur": p["cur_id"]} for p in pairs], **out}


# --------------------------------------------------------------------------- op 2 alone on photo sets

def op2_ab(items: list[tuple[str, Path, str, dict]], rules: dict) -> list[dict]:
    """items: (id, image path, trap kind, extra). Scores measured once; issues judged under both rules."""
    rows = []
    for pid, path, kind, extra in items:
        img = cv2.imread(str(path))
        if img is None:
            continue
        if img.shape[1] > 768:
            img = cv2.resize(img, (768, round(img.shape[0] * 768 / img.shape[1])), interpolation=cv2.INTER_AREA)
        scores = quality.measure(img, find_card(img, trap_kind=kind), rules["after"])[0]
        rows.append({"id": pid, "kind": kind, **extra, "glare_ratio": scores["glare_ratio"], "glare_region": scores["glare_region"],
                     **{n: quality.assess(scores, t) for n, t in rules.items()}})
    return rows


def export_ab(rules: dict) -> dict | None:
    if not EXPORT.exists():
        return None
    with open(EXPORT, encoding="utf-8", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["image_path"]]
    # FlyBot's export still says "sticky" for the two orchard cameras; they are lure traps (policy.yaml trap_kinds)
    res = op2_ab([(r["snapshot_id"], EXPORT.parent / r["image_path"], "lure", {"camera": r["device_serial"]}) for r in rows], rules)
    out = {"source": "data/manifest.csv (FlyBot export, git-ignored), op 2 alone, trap kind lure", "photos": len(res)}
    for cam in sorted({r["camera"] for r in res}) + [None]:
        g = [r for r in res if cam is None or r["camera"] == cam]
        out[cam or "all"] = {"photos": len(g), **{f"glare_{n}": sum("glare" in r[n] for r in g) for n in rules},
                             **{f"any_issue_{n}": sum(bool(r[n]) for r in g) for n in rules}}
    return out


def synthetic_ab(rules: dict) -> dict | None:
    items = []
    for s in SYNTH_SETS:
        base = ROOT / "data" / "synth" / s
        if not (base / "manifest.json").exists():
            return None
        m = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
        for p in m["photos"]:
            items.append((p["photoId"], base / p["file"], p["trapKind"], {"set": s, "defect": (p.get("defect") or {}).get("type")}))
    res = op2_ab(items, rules)
    from sentinel.handler import QUALITY_FLAGS

    def flags(issues):  # the contract's quality flags (op 2 glare -> "overexposed")
        return sorted({QUALITY_FLAGS[i] for i in issues if i in QUALITY_FLAGS})

    out = {"source": "data/synth/{demo-30d,pool} manifests (git-ignored), op 2 alone (no op 8, no planner)", "photos": len(res)}
    clean = [r for r in res if not r["defect"]]
    out["clean"] = {"photos": len(clean), **{f"with_quality_flag_{n}": sum(bool(flags(r[n])) for r in clean) for n in rules},
                    **{f"flags_{n}": dict(Counter(f for r in clean for f in flags(r[n]))) for n in rules},
                    "by_kind": {k: {"photos": sum(r["kind"] == k for r in clean),
                                    **{n: sum(bool(flags(r[n])) for r in clean if r["kind"] == k) for n in rules}}
                                for k in sorted({r["kind"] for r in clean})}}
    out["defects"] = {}
    for d in sorted({r["defect"] for r in res if r["defect"]}):
        for k in sorted({r["kind"] for r in res if r["defect"] == d}):
            g = [r for r in res if r["defect"] == d and r["kind"] == k]
            out["defects"][f"{d}/{k}"] = {"photos": len(g), **{f"flags_{n}": dict(Counter(f for r in g for f in flags(r[n])))
                                                              for n in rules}}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--skip", nargs="*", default=[], choices=["expert", "sample", "failure", "export", "synth"])
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    from sentinel import require_opencv5

    require_opencv5()
    rules = rule_thresholds()
    t0 = time.perf_counter()
    kept = json.loads(a.out.read_text(encoding="utf-8")) if a.skip and a.out.exists() else {}  # parts skipped this run
    res = {**kept, "what": "lure-floor glare limit, before vs after on the same inputs (eval/glare_rule_ab.py)",
           "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
           "rule": {n: {"glare.ratio_max": t["glare"]["ratio_max"], "glare.floor_ratio_max": t["glare"]["floor_ratio_max"]}
                    for n, t in rules.items()}}
    parts = {"expert": ("expert_sheet", expert_sheet), "sample": ("sample_replay", sample_replay),
             "export": ("export", export_ab), "synth": ("synthetic", synthetic_ab), "failure": ("failure_injection", failure_injection)}
    for key, (name, fn) in parts.items():
        if key in a.skip:
            continue
        print(f"[{name}]", flush=True)
        res[name] = fn(rules)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    res["seconds"] = round(time.perf_counter() - t0, 1)
    a.out.write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(f"wrote {rel(a.out)} in {res['seconds']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
