"""Write-ups of the checks on SYNTHETIC photos (eval/synth), from the check.jsonl files eval.synth.check wrote.

    uv run --project lambda/vision python -m eval.synth.report [--sets data/synth/demo-30d data/synth/pool]

Two separate reports, both measurements on synthetic composites, kept apart from the real-photo results
(eval/RESULTS.md) and from the human expert review (eval/EXPERT_REVIEW.md):

    eval/SYNTHETIC_RESULTS.md     + eval/results/synth_check.json       detected vs true count, boxes (clean photos)
    eval/SYNTHETIC_AGENT_EVAL.md  + eval/results/synth_agent_eval.json  the agent's decisions vs the known truth:
                                  injected defects vs the expected action, false alarms on clean photos, surges
The second one is an automatic evaluation against generated ground truth. It is not an expert review: no person gave
any of these verdicts, and none is reported as one.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from eval.synth import ROOT, defects

RESULTS = ROOT / "eval" / "results" / "synth_check.json"
REPORT = ROOT / "eval" / "SYNTHETIC_RESULTS.md"
AGENT_RESULTS = ROOT / "eval" / "results" / "synth_agent_eval.json"
AGENT_REPORT = ROOT / "eval" / "SYNTHETIC_AGENT_EVAL.md"
TILES_RESULTS = ROOT / "eval" / "results" / "synth_tiles.json"
DEFAULT_SETS = [ROOT / "data" / "synth" / "demo-30d", ROOT / "data" / "synth" / "pool"]
SIZE_BINS = [(0, 12), (12, 16), (16, 20), (20, 28), (28, 10_000)]  # fly long side in the detector's 640 input
COUNT_BINS = [(0, 0), (1, 5), (6, 15), (16, 40), (41, 10_000)]
QUALITY_ACTION_RULES = ("quality.", "occlusion.", "trap_missing.", "motion.")


def load(sets: list[Path]) -> tuple[list[dict], list[dict]]:
    recs, metas = [], []
    for d in sets:
        d = Path(d)
        metas.append(json.loads((d / "check.meta.json").read_text(encoding="utf-8")))
        with open(d / "check.jsonl", encoding="utf-8") as fh:
            recs += [json.loads(line) for line in fh if line.strip()]
    return recs, metas


def r4(x: float | None) -> float | None:
    return None if x is None else round(x, 4)


def pearson(a: list[float], b: list[float]) -> float | None:
    if len(a) < 3 or statistics.pstdev(a) == 0 or statistics.pstdev(b) == 0:
        return None
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b, strict=True)) / len(a)
    return round(cov / (statistics.pstdev(a) * statistics.pstdev(b)), 4)


def pct(values: list[float], p: float) -> float | None:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    return round(vals[max(0, min(len(vals) - 1, round(p / 100 * (len(vals) - 1))))], 1)


# ---------------------------------------------------------------- counts


def summarize(rows: list[dict]) -> dict:
    """rows: {true, det, npred, tp30, tp50} -> count and box metrics (true = flies visible in the photo)."""
    if not rows:
        return {"photos": 0}
    err = [r["det"] - r["true"] for r in rows]
    n_true, n_pred = sum(r["true"] for r in rows), sum(r["npred"] for r in rows)
    out = {"photos": len(rows), "true_flies": n_true, "detected": n_pred,
           "mae": round(statistics.fmean(abs(e) for e in err), 3), "bias": round(statistics.fmean(err), 3),
           "rmse": round(math.sqrt(statistics.fmean(e * e for e in err)), 3),
           "within_2": sum(abs(e) <= 2 for e in err),
           "within_10pct_or_2": sum(abs(e) <= max(2, 0.1 * r["true"]) for e, r in zip(err, rows, strict=True)),
           "count_r": pearson([r["true"] for r in rows], [r["det"] for r in rows])}
    for thr in ("30", "50"):
        tp = sum(r[f"tp{thr}"] for r in rows)
        p = tp / n_pred if n_pred else None
        rc = tp / n_true if n_true else None
        out[f"iou{thr}"] = {"tp": tp, "precision": r4(p), "recall": r4(rc),
                            "f1": round(2 * p * rc / (p + rc), 4) if p and rc else None}
    empty = [r for r in rows if r["true"] == 0]
    out["empty_photos"] = {"photos": len(empty), "with_detections": sum(r["det"] > 0 for r in empty),
                           "false_flies": sum(r["det"] for r in empty)}
    return out


def detector_only(rows: list[dict], key: str) -> dict:
    rows = [r for r in rows if key in r]
    if not rows:
        return {"photos": 0}
    n_true, n_pred = sum(r["true"] for r in rows), sum(r[key] for r in rows)
    tp = sum(r[f"{key}_tp30"] for r in rows)
    return {"photos": len(rows), "true_flies": n_true, "detected": n_pred,
            "mae": round(statistics.fmean(abs(r[key] - r["true"]) for r in rows), 3),
            "bias": round(statistics.fmean(r[key] - r["true"] for r in rows), 3),
            "precision_iou30": r4(tp / n_pred if n_pred else None), "recall_iou30": r4(tp / n_true if n_true else None)}


def count_report(recs: list[dict], metas: list[dict]) -> dict:
    clean = [{**r, "true": r["visible"]} for r in recs if not r.get("defect")]
    sets = sorted({r["set"] for r in clean})
    size_acc: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in clean:
        for side, hit in r["gtSizes640"]:
            key = next(f"{lo}-{hi if hi < 10_000 else ''}" for lo, hi in SIZE_BINS if lo <= side < hi)
            size_acc[key][0] += 1
            size_acc[key][1] += bool(hit)
    by_count = {}
    for lo, hi in COUNT_BINS:
        sel = [r for r in clean if lo <= r["true"] <= hi]
        key = f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10_000 else f"{lo}+")
        by_count[key] = {"photos": len(sel), "mae": round(statistics.fmean(abs(r["det"] - r["true"]) for r in sel), 3)
                         if sel else None, "bias": round(statistics.fmean(r["det"] - r["true"] for r in sel), 3) if sel else None}
    daily = {}
    for t in sorted({r["trap"] for r in clean if r.get("date")}):
        per_day: dict[str, list] = defaultdict(lambda: [0, 0])
        for r in clean:
            if r["trap"] == t:
                per_day[r["date"]][0] += r["true"]
                per_day[r["date"]][1] += r["det"]
        days = sorted(per_day)
        daily[t] = {"days": len(days), "r_daily_sum": pearson([per_day[d][0] for d in days], [per_day[d][1] for d in days]),
                    "true_sum": sum(per_day[d][0] for d in days), "det_sum": sum(per_day[d][1] for d in days)}
    tiles = any("tiles2x2" in r for r in clean)
    return {
        "kind": "measurement on SYNTHETIC composites (eval/synth), not field photos",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"), "sets": metas,
        "photos": {"all": len(recs), "clean": len(clean), "damaged_left_out": len(recs) - len(clean)},
        "match": "greedy one-to-one, IoU >= 0.3 (primary) and 0.5; true boxes = pasted fly mask at alpha > 0.25",
        "by_set": {s: summarize([r for r in clean if r["set"] == s]) for s in sets},
        "by_set_kind": {f"{s} / {k}": summarize([r for r in clean if r["set"] == s and r["trapKind"] == k])
                        for s in sets for k in ("lure", "sticky") if any(r["set"] == s and r["trapKind"] == k for r in clean)},
        "by_trap": {t: summarize([r for r in clean if r["trap"] == t]) for t in sorted({r["trap"] for r in clean})
                    if not t.startswith("pool-")},
        "by_true_count": by_count,
        "recall_by_fly_size_640": {k: {"flies": v[0], "recall_iou30": r4(v[1] / v[0]) if v[0] else None}
                                   for k, v in sorted(size_acc.items(), key=lambda kv: int(kv[0].split("-")[0]))},
        "daily": daily,
        "detector_only": None if not tiles else {
            "note": "YOLO26n alone on the whole frame vs on 2x2 overlapping tiles (12%), no trap ROI, no box rules: "
                    "not the handler's count, a check of what tiling would change",
            **{f"{v}_{k}": detector_only([r for r in clean if r["trapKind"] == k], v)
               for k in ("lure", "sticky") for v in ("full", "tiles2x2")}},
        "timings_ms": {"total_p50": pct([r["totalMs"] for r in recs], 50), "total_p95": pct([r["totalMs"] for r in recs], 95),
                       "dnn_p50": pct([r["dnnMs"] for r in recs], 50),
                       "note": "dev laptop shared with other jobs, not the Lambda"},
    }


# ---------------------------------------------------------------- agent decisions


def judge(r: dict) -> dict:
    """One damaged photo: did the response do what the policy prescribes for its defect?"""
    e = r["expected"]
    flag = e["flag"] in r["flags"] or e["flag"] in r["issues"]
    rule = any(x and x.startswith(e["rule"]) for x in r["rules"])
    actions = set(e["actions"]) <= set(r["actions"])
    counted_ok = not r["counted"] if e["counted"] is False else True
    return {"flag": flag, "rule": rule, "actions": actions, "counted_ok": counted_ok,
            "outcome": actions and counted_ok, "pass": flag and rule and actions and counted_ok}


def agent_report(recs: list[dict], metas: list[dict]) -> dict:
    dmg = [r for r in recs if r.get("defect")]
    clean = [r for r in recs if not r.get("defect")]
    by_type = {}
    for kind in defects.TYPES:
        for tk in ("lure", "sticky", "all"):
            sel = [r for r in dmg if r["defect"] == kind and (tk == "all" or r["trapKind"] == tk)]
            if not sel:
                continue
            j = [judge(r) for r in sel]
            by_type[f"{kind} / {tk}"] = {
                "photos": len(sel), "expected": sel[0]["expected"] if tk != "all" else defects.EXPECTED[kind],
                "flag_raised": sum(x["flag"] for x in j), "rule_taken": sum(x["rule"] for x in j),
                "actions_proposed": sum(x["actions"] for x in j), "not_counted_ok": sum(x["counted_ok"] for x in j),
                "outcome_correct": sum(x["outcome"] for x in j), "all_correct": sum(x["pass"] for x in j),
                "observed_flags": dict(sorted(_count(f for r in sel for f in r["flags"]).items())),
                "observed_actions": dict(sorted(_count(a for r in sel for a in r["actions"]).items())),
                "with_prev": sum(1 for r in sel if r.get("prevSent"))}
    alarms = [r for r in clean if r["flags"]]
    false_actions = [r for r in clean if any(x and x.startswith(QUALITY_ACTION_RULES) for x in r["rules"])
                     and set(r["actions"]) & {"capture_now", "set_isp", "review_request"}]
    per_bg: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in clean:
        if r["trapKind"] == "lure":
            per_bg[r.get("background", "?")][0] += 1
            per_bg[r.get("background", "?")][1] += bool(r["flags"])
    replay = ROOT / "eval" / "results" / "replay_sample_yolo26.json"
    untouched = None
    if replay.exists():
        rs = json.loads(replay.read_text(encoding="utf-8"))
        untouched = {"source": "eval/results/replay_sample_yolo26.json", "photos": rs.get("photos"),
                     "quality_flags": rs.get("quality_flags")}
    clean_block = {
        "untouched_backgrounds": untouched,
        "lure_backgrounds_flagged": sum(1 for v in per_bg.values() if v[1] > 0), "lure_backgrounds": len(per_bg),
        "photos": len(clean), "with_quality_flag": len(alarms),
        "flags": dict(sorted(_count(f for r in clean for f in r["flags"]).items())),
        "with_quality_action": len(false_actions),
        "actions": dict(sorted(_count(a for r in false_actions for a in r["actions"]).items())),
        "not_counted": sum(1 for r in clean if not r["counted"]),
        "by_kind": {k: {"photos": sum(1 for r in clean if r["trapKind"] == k),
                        "with_quality_flag": sum(1 for r in alarms if r["trapKind"] == k),
                        "with_quality_action": sum(1 for r in false_actions if r["trapKind"] == k)}
                    for k in ("lure", "sticky")}}
    surge = {}
    series = [r for r in recs if r.get("surgeTruth") is not None]
    for t in sorted({r["trap"] for r in series}):
        sel = [r for r in series if r["trap"] == t]
        tp = sum(1 for r in sel if r["surgeTruth"] and r["surgeAgent"])
        fp = sum(1 for r in sel if not r["surgeTruth"] and r["surgeAgent"])
        fn = sum(1 for r in sel if r["surgeTruth"] and not r["surgeAgent"])
        days_t = {r["date"] for r in sel if r["surgeTruth"]}
        days_a = {r["date"] for r in sel if r["surgeAgent"]}
        surge[t] = {"trapKind": sel[0]["trapKind"], "photos": len(sel), "truth_photos": tp + fn, "agent_photos": tp + fp,
                    "tp": tp, "fp": fp, "fn": fn, "precision": r4(tp / (tp + fp)) if tp + fp else None,
                    "recall": r4(tp / (tp + fn)) if tp + fn else None,
                    "truth_days": sorted(days_t), "agent_days": sorted(days_a),
                    "days_both": len(days_t & days_a), "days_agent_only": len(days_a - days_t),
                    "days_truth_only": len(days_t - days_a)}
    return {
        "kind": "automatic evaluation of the agent's decisions against the generated truth of SYNTHETIC photos; "
                "not an expert review (no person gave these verdicts)",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"), "sets": metas,
        "photos": {"all": len(recs), "damaged": len(dmg), "clean": len(clean)},
        "expected_source": "eval/synth/defects.py EXPECTED, written from sentinel/agent/policy.yaml and the RulePlanner",
        "defects": by_type, "clean": clean_block, "surge": surge,
        "surge_rule": "sentinel.agent.domain.detect_surge (FlyBot surgeService: 3 days, ratio 1.5, rise >= 3, count >= 5) "
                      "on the true counts, card changes known; the agent sees the handler's counts as history, "
                      "without card-change dates (the contract's history has none)",
    }


def _count(items) -> dict:
    out: dict = defaultdict(int)
    for x in items:
        out[x] += 1
    return dict(out)


# ---------------------------------------------------------------- markdown


def _f(v) -> str:
    return "-" if v is None else str(v)


def _sets_lines(metas: list[dict]) -> list[str]:
    out = []
    for m in metas:
        mod = m.get("model") or {}
        out.append(f"- `{m['set']}` (seed {m['seed']}, {m['ok']}/{m['photos']} photos, generator git {m['generator_git']}); "
                   f"model {_f(mod.get('name'))} {_f(mod.get('version'))} on {_f(mod.get('engine'))}; prevPhoto "
                   f"{'sent' if m['prev_photo'] else 'not sent'}; {m['transport']}; run {m['run_s']} s")
    return out


def tiles_lines(path: Path = TILES_RESULTS) -> list[str]:
    """The DETECTOR_TILES table from eval.synth.tiles (eval/results/synth_tiles.json), when it was run."""
    if not path.is_file():
        return []
    t = json.loads(path.read_text(encoding="utf-8"))
    fm = t["forward_ms"]
    lines = ["", "## Tiled detection (DETECTOR_TILES)", "",
             (f"`uv run --project lambda/vision python -m eval.synth.tiles` on {t['generated_at']}, source "
              f"`eval/results/synth_tiles.json`: {t['scope']}. Tiling: {t['tiling']}. Sets: "
              + ", ".join(f"`{s['set']}` ({s['clean_photos']} clean photos)" for s in t["sets"]) + f"; {t['model']}."),
             "", ("Variants: `off` whole frame (the default); `on` 2x2 windows on every photo; `sticky` windows on sticky "
             "traps only; `autoN` a whole-frame pass, then windows when the median box long side at the 640 input is "
             "under N px or no box was found on a frame >= 2 x 640 (whole-frame boxes join the merge; `-full`: they "
             "do not)."), "",
             ("| variant | trap | photos | tiled | forwards / photo | true flies | detected | MAE | bias | precision @0.3 | "
             "recall @0.3 | F1 @0.3 |"), "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, by_kind in t["results"].items():
        for kind, v in by_kind.items():
            if v.get("photos"):
                lines.append(f"| {name} | {kind} | {v['photos']} | {v['tiled_photos']} | {v['forwards_per_photo']} | "
                             f"{v['true_flies']} | {v['detected']} | {_f(v['mae'])} | {_f(v['bias'])} | "
                             f"{_f(v['precision_iou30'])} | {_f(v['recall_iou30'])} | {_f(v['f1_iou30'])} |")
    lines += ["", (f"DNN forward p50 during this run: whole frame {fm['whole_frame_p50']} ms, the 4 windows together "
                   f"{fm['four_windows_p50']} ms ({fm['note']}).")]
    if t.get("latency"):
        lt = t["latency"]
        lines += ["", (f"Latency cost (`--latency`, {lt['photos']} sticky photos of `{lt['set']}`, {lt['size']}): "
                       f"`detect()` p50 {lt['off']['detect_ms_p50']} ms whole frame vs {lt['sticky']['detect_ms_p50']} ms "
                       f"tiled (forwards {lt['off']['forward_ms_p50']} vs {lt['sticky']['forward_ms_p50']} ms); "
                       f"{lt['note']}.")]
    if t.get("handler"):
        lines += ["", ("Whole handler (trap ROI, box rules, quality ops, agent; prevPhoto sent, so a tiled photo costs 4 "
                       "forwards for the photo and 4 for the previous one) on the clean photos, one run per "
                       "DETECTOR_TILES value (its times come from runs that shared the laptop with other jobs; the latency "
                       "line above is the cost):"), "",
                  ("| DETECTOR_TILES | set | trap | photos | true flies | detected | MAE | bias | precision @0.3 | "
                   "recall @0.3 | F1 @0.3 | DNN ms p50 (photo) | handler ms p50 |"),
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for name, run in t["handler"].items():
            for kind in ("lure", "sticky"):
                v = run.get(kind) or {}
                if v.get("photos"):
                    lines.append(f"| {name} | {run['set']} | {kind} | {v['photos']} | {v['true_flies']} | {v['detected']} | "
                                 f"{_f(v['mae'])} | {_f(v['bias'])} | {_f(v['precision_iou30'])} | "
                                 f"{_f(v['recall_iou30'])} | {_f(v['f1_iou30'])} | {_f(v['dnn_ms_p50'])} | "
                                 f"{_f(v['total_ms_p50'])} |")
    return lines


def count_markdown(res: dict) -> str:
    lines = [
        "# Synthetic composites: detected vs true count", "",
        "> **SIMULATED DATA.** Every number on this page is measured on synthetic composites made by `eval/synth`: fruit",
        "> flies cut from the camos/fruit-fly v3 test and valid splits (Roboflow Universe, CC BY 4.0) pasted onto real",
        "> photos of *empty* lure traps (data/samples) or a synthetic sticky card. It is not a field result and is kept",
        "> apart from [RESULTS.md](RESULTS.md), which reports the real-photo evaluation. The true count is known exactly",
        "> because the generator placed every fly. The agent's decisions on these photos are evaluated separately in",
        "> [SYNTHETIC_AGENT_EVAL.md](SYNTHETIC_AGENT_EVAL.md).", "",
        (f"Generated by `uv run --project lambda/vision python -m eval.synth.report` on {res['generated_at']} from the "
        "sets' check.jsonl. Do not edit by hand: source `eval/results/synth_check.json`."), "",
        *_sets_lines(res["sets"]),
        (f"- Photos: {res['photos']['clean']} clean photos measured; the {res['photos']['damaged_left_out']} photos damaged "
        "on purpose are left out of the counts (they are for the agent evaluation)."),
        "- True count = flies visible in the photo. " + res["match"], "",
        "## Counts", "",
        "| scope | photos | true flies | detected | MAE | bias (det - true) | RMSE | within 2 | within max(2, 10%) | r |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    def crow(name, s):
        return (f"| {name} | {s['photos']} | {s.get('true_flies', '-')} | {s.get('detected', '-')} | {_f(s.get('mae'))} | "
                f"{_f(s.get('bias'))} | {_f(s.get('rmse'))} | {_f(s.get('within_2'))} | {_f(s.get('within_10pct_or_2'))} | "
                f"{_f(s.get('count_r'))} |")

    lines += [crow(k, s) for k, s in res["by_set_kind"].items()]
    lines += [crow(t, s) for t, s in res["by_trap"].items()]
    lines += ["", "## Boxes", "", "| scope | precision @0.3 | recall @0.3 | F1 @0.3 | precision @0.5 | recall @0.5 | F1 @0.5 |",
              "|---|---|---|---|---|---|---|"]
    for name, s in res["by_set_kind"].items():
        a, b = s.get("iou30") or {}, s.get("iou50") or {}
        lines.append(f"| {name} | {_f(a.get('precision'))} | {_f(a.get('recall'))} | {_f(a.get('f1'))} | "
                     f"{_f(b.get('precision'))} | {_f(b.get('recall'))} | {_f(b.get('f1'))} |")
    lines += ["", "Photos with no visible fly: " + "; ".join(
        f"{k} {v['empty_photos']['photos']} ({v['empty_photos']['with_detections']} with a detection, "
        f"{v['empty_photos']['false_flies']} false flies)" for k, v in res["by_set_kind"].items() if v.get("empty_photos")) + ".",
              "", "## Error by crowding (true count per photo)", "", "| true count | photos | MAE | bias |", "|---|---|---|---|"]
    lines += [f"| {k} | {v['photos']} | {_f(v['mae'])} | {_f(v['bias'])} |" for k, v in res["by_true_count"].items()]
    lines += ["", "## Recall by fly size (long side in the detector's 640 px input, IoU 0.3)", "",
              "| size px | flies | recall |", "|---|---|---|"]
    lines += [f"| {k} | {v['flies']} | {_f(v['recall_iou30'])} |" for k, v in res["recall_by_fly_size_640"].items()]
    if res["daily"]:
        lines += ["", "## Daily curve per trap (sum of the day's clean photos)", "",
                  "| trap | days | true sum | detected sum | r daily |", "|---|---|---|---|---|"]
        lines += [f"| {t} | {d['days']} | {d['true_sum']} | {d['det_sum']} | {_f(d['r_daily_sum'])} |" for t, d in res["daily"].items()]
    if res.get("detector_only"):
        lines += ["", "## Detector only: whole frame vs 2x2 tiles", "", res["detector_only"]["note"] + ".", "",
                  "| variant | photos | true flies | detected | MAE | bias | precision @0.3 | recall @0.3 |",
                  "|---|---|---|---|---|---|---|---|"]
        for k, v in res["detector_only"].items():
            if k != "note":
                lines.append(f"| {k} | {v['photos']} | {v.get('true_flies', '-')} | {v.get('detected', '-')} | "
                             f"{_f(v.get('mae'))} | {_f(v.get('bias'))} | {_f(v.get('precision_iou30'))} | "
                             f"{_f(v.get('recall_iou30'))} |")
    tm = res["timings_ms"]
    lines += ["", (f"Time per photo (whole handler) p50 {tm['total_p50']} ms, p95 {tm['total_p95']} ms; DNN p50 "
              f"{tm['dnn_p50']} ms ({tm['note']}).")]
    lines += [*tiles_lines(), "", "## Limits", "",
              "- The flies come from the same dataset family the detector was trained on (other images: test/valid splits),",
              "  photographed on white discs; field flies on these traps may look different (pose, dust, decay, lighting).",
              "- Sticky traps: the yellow card is synthetic as well (no real sticky-card photo exists), so the sticky rows",
              "  measure the pipeline on a synthetic card, not on a real one.",
              "- Fly size is a generator setting (`flyScale`). Lure: 0.11 x floor width, from FlyBot's production boxes on",
              "  BE4089629 photos of 2026-09-19..28 (long side median 63 px on 768 px photos; RF-DETR boxes, not human",
              "  labels). Sticky: 0.04 x the card's long side (a ~1 cm fly on the 25 cm card of the ESP32 trap guide).",
              "  Recall depends strongly on size (table above).", ""]
    return "\n".join(lines)


def agent_markdown(res: dict) -> str:
    lines = [
        "# Automatic evaluation of the agent on synthetic photos", "",
        "> **SIMULATED DATA, NOT AN EXPERT REVIEW.** This page compares the agent's decisions with the truth the generator",
        "> knows for each synthetic photo (`eval/synth`): the flies it placed, the defect it injected and the action the",
        "> policy prescribes for that defect. No person reviewed these photos or gave any verdict here. The human expert",
        "> review is [EXPERT_REVIEW.md](EXPERT_REVIEW.md) (separate, with its own status); real-photo results are in",
        "> [RESULTS.md](RESULTS.md); counting accuracy on the same synthetic photos is in",
        "> [SYNTHETIC_RESULTS.md](SYNTHETIC_RESULTS.md).", "",
        (f"Generated by `uv run --project lambda/vision python -m eval.synth.report` on {res['generated_at']}. Do not edit "
        "by hand: source `eval/results/synth_agent_eval.json`."), "",
        *_sets_lines(res["sets"]),
        (f"- Photos: {res['photos']['all']} ({res['photos']['damaged']} damaged on purpose, {res['photos']['clean']} clean). "
        "Each request is what FlyBot sends: platform and trap kind, capture time, the trap's latest clean photo as "
        "`prevPhoto`, its earlier counts (as the handler gave them) as `history`; rule planner."),
        (f"- Expected actions: {res['expected_source']}. A photo is correct when the response raises the expected flag, "
        "takes the expected decision rule, proposes the expected actions and, where the policy says so, does not count it."),
        "", "## Damaged photos: defect vs the expected action", "",
        ("| defect / trap kind | photos | expected flag | expected actions | flag raised | rule taken | actions proposed | "
        "not counted (when required) | outcome correct | all correct |"),
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for k, v in res["defects"].items():
        if k.endswith("/ all"):
            continue
        e = v["expected"]
        nc = "-" if e["counted"] else f"{v['not_counted_ok']}/{v['photos']}"
        lines.append(f"| {k} | {v['photos']} | {e['flag']} | {', '.join(e['actions'])} | {v['flag_raised']}/{v['photos']} | "
                     f"{v['rule_taken']}/{v['photos']} | {v['actions_proposed']}/{v['photos']} | {nc} | "
                     f"{v['outcome_correct']}/{v['photos']} | {v['all_correct']}/{v['photos']} |")
    lines += ["", ("Outcome correct = the expected actions are proposed and, where required, the photo is not counted (what "
              "the farmer sees); all correct = also the expected flag and decision rule (the right reason).")]
    lines += ["", "What the response raised and proposed instead, per defect (all trap kinds):", "",
              "| defect | photos | flags seen | actions seen |", "|---|---|---|---|"]
    for k, v in res["defects"].items():
        if k.endswith("/ all"):
            lines.append(f"| {k.removesuffix(' / all')} | {v['photos']} | {v['observed_flags'] or '-'} | "
                         f"{v['observed_actions'] or '-'} |")
    c = res["clean"]
    lines += ["", "## Clean photos: false alarms", "",
              (f"{c['photos']} photos without any injected defect: {c['with_quality_flag']} raised a quality flag "
              f"({c['flags'] or 'none'}), {c['with_quality_action']} got a quality / motion / occlusion / trap-missing "
              f"action ({c['actions'] or 'none'}), {c['not_counted']} were not counted."), "",
              "| trap kind | photos | with a quality flag | with such an action |", "|---|---|---|---|"]
    lines += [f"| {k} | {v['photos']} | {v['with_quality_flag']} | {v['with_quality_action']} |" for k, v in c["by_kind"].items()]
    u = c.get("untouched_backgrounds")
    lines += ["", f"Lure backgrounds with at least one flagged clean composite: {c['lure_backgrounds_flagged']} of "
              f"{c['lure_backgrounds']}."
              + (f" The untouched real photos (no fly pasted in) already get {u['quality_flags']} from the handler on "
                 f"{u['photos']} photos ({u['source']}): most of these flags are the backgrounds' own."
                 if u else "")]
    if res["surge"]:
        lines += ["", "## Surges: the agent vs the surge rule on the true counts", "", res["surge_rule"] + ".", "",
                  ("| trap | kind | photos | true surge photos | agent surge photos | both | agent only | truth only | "
                  "precision | recall | days both / agent only / truth only |"),
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for t, v in res["surge"].items():
            lines.append(f"| {t} | {v['trapKind']} | {v['photos']} | {v['truth_photos']} | {v['agent_photos']} | {v['tp']} | "
                         f"{v['fp']} | {v['fn']} | {_f(v['precision'])} | {_f(v['recall'])} | "
                         f"{v['days_both']} / {v['days_agent_only']} / {v['days_truth_only']} |")
    lines += ["", "## Limits", "",
              "- The defects are synthetic (eval/synth/defects.py); their strength was chosen inside the ranges where the",
              "  failure-injection sweep (eval/results/failure_injection.json) shows op 2 reacting. Real damage looks different.",
              "- Each photo is one stateless Lambda call: what the agent does on the *next* photo (a recapture, a second",
              "  failure, a person's answer) is not part of this evaluation.",
              "- Surge truth uses the true counts; the agent works from the handler's counts, so a counting error shows up as",
              "  a surge error too.", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sets", type=Path, nargs="+", default=DEFAULT_SETS, help="folders with check.jsonl")
    a = ap.parse_args(argv)
    recs, metas = load(a.sets)
    cr, ar = count_report(recs, metas), agent_report(recs, metas)
    for path, res, md, text in ((RESULTS, cr, REPORT, count_markdown(cr)), (AGENT_RESULTS, ar, AGENT_REPORT, agent_markdown(ar))):
        path.write_text(json.dumps(res, indent=2), encoding="utf-8", newline="\n")
        md.write_text(text, encoding="utf-8", newline="\n")
        print(f"written: {path}, {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
