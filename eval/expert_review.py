"""Build the expert-review sheet: 50 agent decisions + 5 worked examples, for an AgriHub agronomist.

    uv run python -m eval.expert_review                    # writes docs/evidence/eval/expert_review.csv (+ images, index.html)
    uv run --with awscrt python -m eval.expert_review --fetch-s3     # also downloads the live frames (read-only S3 GET)

Stratified, seeded (deterministic) sample of 50 decisions:
    aws_live             15  data/eval/aws_traces.jsonl: live per-photo traces on AWS (recapture and no_op, both cameras)
    replay_30d_review    12  data/eval/replay/30d_approve-after-2h/trace.jsonl: the real 30-day replay (human reviews first, no_op to fill)
    staged_proposal       8  data/eval/replay/stress_7d_synthetic_surge: the staged surge (counts INJECTED, photo real): proposal, neighbour checks, follow-ups
    failure_injection    15  eval/results/failure_injection.json: perturbed real photos and their outcome
plus 5 worked examples (`EX-*`, expert_note starts with `reviewer=agent-example`): shown as a format guide, never scored.

Columns: id, source, camera, captured_at, evidence_link, opencv_summary, agent_decision, agent_reason, expert_verdict, expert_note.
`expert_verdict` and `expert_note` stay empty (except the examples). Every row's image is copied (<= 800 px, JPEG) to
docs/evidence/eval/expert_review/<id>.jpg so the reviewer needs no AWS access; AWS rows keep the S3 key in `evidence_link`.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import random
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.aws_traces import read_rows
from eval.score_expert_review import COLUMNS

OUT_DIR = ROOT / "docs" / "evidence" / "eval"
CSV_PATH = OUT_DIR / "expert_review.csv"
IMG_DIR = OUT_DIR / "expert_review"
REPLAY = ROOT / "data" / "eval" / "replay"
SEED = 20261015
QUOTA = {"aws_live": 15, "replay_30d_review": 12, "staged_proposal": 8, "failure_injection": 15}
S3_BUCKET = "sentinel-685527430355-dev"


# ------------------------------------------------------------------------------------------------ describing a trace

def _obs(tr: dict) -> dict:
    o = list((tr.get("opencv_outputs") or {}).values())
    return o[0] if o else {}


def opencv_summary(tr: dict) -> str:
    o = _obs(tr)
    if not o:
        return "no observation in the trace"
    c, q = o.get("counts") or {}, o.get("quality") or {}
    bits = [f"count opencv={c.get('opencv')} npu={c.get('npu')} production={c.get('production')}",
            f"quality ok={q.get('ok')} issues={q.get('issues') or []} score={q.get('score')}"]
    sc = q.get("scores") or {}
    if sc.get("sharpness") is not None:
        bits.append(f"sharpness={sc['sharpness']} brightness={sc.get('brightness')} glare_ratio={sc.get('glare_ratio')}")
    card = o.get("card") or {}
    if card:
        bits.append(f"trap found={card.get('found')} kind={card.get('trap_kind')} coverage={card.get('coverage')}")
    nc = o.get("new_catch") or {}
    if nc and nc.get("new_count") is not None:
        bits.append(f"new_catch={nc.get('new_count')} card_change_suspected={nc.get('card_change_suspected')}")
    return "; ".join(bits)


def agent_decision(tr: dict) -> str:
    decs = tr.get("decisions") or []
    if not decs:
        return tr.get("outcome") or "no decision"
    return " + ".join(dict.fromkeys(f"{d.get('action')} ({d.get('rule')})" for d in decs))


def agent_reason(tr: dict) -> str:
    out = []
    for d in tr.get("decisions") or []:
        out += [str(r) for r in d.get("reasons") or []]
    txt = " | ".join(dict.fromkeys(out)) or tr.get("reasoning_summary") or ""
    extra = tr.get("outcome")
    blocked = [c for c in tr.get("tool_calls") or [] if c.get("status") == "blocked"]
    if blocked:
        extra = f"{extra} (blocked: {blocked[0].get('blocked_reason')})"
    return (txt[:420] + (" ... " if len(txt) > 420 else "") + f" [outcome: {extra}]").strip()


def local_image(uri: str | None) -> Path | None:
    if not uri:
        return None
    if uri.startswith("file://"):
        return Path(unquote(urlparse(uri).path.lstrip("/")))
    if uri.startswith("s3://"):
        return None
    p = Path(uri)
    return p if p.is_absolute() else ROOT / p


def s3_key(uri: str | None) -> str | None:
    if uri and uri.startswith("s3://"):
        return uri
    return None


# ------------------------------------------------------------------------------------------------ the pools

def pool_aws() -> list[dict]:
    rows = []
    for r in read_rows():
        tr = r["trace"]
        o = _obs(tr)
        rows.append({"camera": tr.get("camera"), "captured_at": o.get("captured_at"), "tr": tr, "image": o.get("image"),
                     "stratum": (tr.get("camera"), tr.get("decisions", [{}])[0].get("rule") if tr.get("decisions") else "none",
                                 ",".join((o.get("quality") or {}).get("issues") or []))})
    return rows


def pool_replay(name: str, want) -> list[dict]:
    p = REPLAY / name / "trace.jsonl"
    rows = []
    if not p.exists():
        return rows
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("kind") != "photo" or not r.get("agent"):
                continue
            if not want(r):
                continue
            tr = r["agent"]
            o = _obs(tr)
            rows.append({"camera": r["camera"], "captured_at": r.get("captured_at"), "tr": tr, "image": o.get("image"),
                         "stratum": (r["camera"], "|".join(r.get("decision") or []), (r.get("captured_at") or "")[:10])})
    return rows


def pick(pool: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Round-robin over strata (largest strata last), deterministic."""
    by: dict = {}
    for r in sorted(pool, key=lambda x: (str(x["stratum"]), str(x["captured_at"]))):
        by.setdefault(r["stratum"], []).append(r)
    for v in by.values():
        rng.shuffle(v)
    keys = sorted(by, key=lambda k: (len(by[k]), str(k)))
    out: list[dict] = []
    while len(out) < n and any(by.values()):
        for k in keys:
            if by[k] and len(out) < n:
                out.append(by[k].pop())
    return out


def sample() -> tuple[list[dict], list[dict]]:
    rng = random.Random(SEED)
    chosen: dict[str, list[dict]] = {}
    # AWS: recapture decisions and no_op, both cameras
    aws = pool_aws()
    rec = [r for r in aws if r["stratum"][1].endswith(".recapture")]
    noop = [r for r in aws if not r["stratum"][1].endswith(".recapture")]
    chosen["aws_live"] = pick(rec, 8, rng) + pick(noop, QUOTA["aws_live"] - min(8, len(rec)), rng)
    # replay 30 d: human reviews first, no_op / other decisions to fill
    rv = pool_replay("30d_approve-after-2h", lambda r: any(d != "no_op" for d in (r.get("decision") or [])))
    fill = pool_replay("30d_approve-after-2h", lambda r: (r.get("decision") or []) == ["no_op"] and r.get("quality_issues"))
    chosen["replay_30d_review"] = pick(rv, QUOTA["replay_30d_review"], rng)
    if len(chosen["replay_30d_review"]) < QUOTA["replay_30d_review"]:
        chosen["replay_30d_review"] += pick(fill, QUOTA["replay_30d_review"] - len(chosen["replay_30d_review"]), rng)
    # staged: the proposal, then the other non-no_op decision shapes, then two no_op around the surge
    st_all = pool_replay("stress_7d_synthetic_surge", lambda r: True)
    prop = [r for r in st_all if "propose_treatment" in r["stratum"][1]]
    others = [r for r in st_all if r["stratum"][1] not in ("no_op",) and r not in prop]
    quiet = [r for r in st_all if r["stratum"][1] == "no_op"]
    chosen["staged_proposal"] = prop[:1] + pick(others, 5, rng) + pick(quiet, QUOTA["staged_proposal"] - 1 - min(5, len(others)), rng)
    # extra picks for the examples: not in the 50
    used = {id(x) for v in chosen.values() for x in v}
    ex_pool = [r for r in aws if id(r) not in used]
    ex_aws = pick([r for r in ex_pool if r["stratum"][1].endswith(".recapture")], 2, rng) + pick([r for r in ex_pool if not r["stratum"][1].endswith(".recapture")], 2, rng)
    ex_rv = pick([r for r in rv if r not in chosen["replay_30d_review"]], 1, rng)
    return chosen, ex_aws + ex_rv


def pool_failure() -> list[dict]:
    p = ROOT / "eval" / "results" / "failure_injection.json"
    if not p.exists():
        return []
    d = json.loads(p.read_text(encoding="utf-8"))
    first = d["photos"][0]["cur"] if d.get("photos") else None
    return [c for c in d["cases"] if c.get("photo") == first]


FAIL_PRIORITY = ["blur_moderate", "blur_strong", "glare_ai_camera", "glare_ezviz_camera", "dark_gamma5", "ir_night_frame", "occluder_hand",
                 "trap_cropped_away", "injected_text_pixels", "injected_text_metadata_llm_obeys", "duplicate_webhook_id", "llm_timeout",
                 "camera_offline", "quiet_hours", "battery_15_percent"]


def failure_rows() -> list[dict]:
    cases = {c["case"]: c for c in pool_failure()}
    out = []
    for name in FAIL_PRIORITY:
        c = cases.get(name)
        if c:
            out.append(c)
    return out[:QUOTA["failure_injection"]]


# ------------------------------------------------------------------------------------------------ writing

def save_image(src: Path | None, dst: Path) -> bool:
    if src is None or not src.exists():
        return False
    img = cv2.imread(str(src))
    if img is None:
        return False
    h, w = img.shape[:2]
    if max(h, w) > 800:
        k = 800 / max(h, w)
        img = cv2.resize(img, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return True


def fetch_s3(uri: str, dst: Path) -> bool:
    import boto3

    u = urlparse(uri)
    s3 = boto3.Session(profile_name="sentinel", region_name="ap-southeast-2").client("s3")
    tmp = dst.with_suffix(".src")
    dst.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(u.netloc, u.path.lstrip("/"), str(tmp))
    ok = save_image(tmp, dst)
    tmp.unlink(missing_ok=True)
    return ok


def build(fetch: bool) -> list[dict]:
    chosen, examples = sample()
    for old in IMG_DIR.glob("*.jpg"):  # a rebuilt sheet must not keep pictures of a previous sample
        old.unlink()
    rows: list[dict] = []
    prefix = {"aws_live": "AWS", "replay_30d_review": "RPL", "staged_proposal": "STG", "failure_injection": "FIJ"}

    def add(rid, source, item, *, ex=False):
        tr = item["tr"]
        img = item.get("image")
        link = s3_key(img)
        dst = IMG_DIR / f"{rid}.jpg"
        if link:
            if fetch and not dst.exists():
                try:
                    fetch_s3(link, dst)
                except Exception as e:  # noqa: BLE001 (report, keep the row)
                    print(f"  {rid}: S3 fetch failed: {e}")
        else:
            save_image(local_image(img), dst)
        have = dst.exists()
        summary = opencv_summary(tr)
        if source == "staged_proposal":
            summary = "STAGED: the count fields were rewritten to simulate a surge (the photo is real); " + summary
        rows.append({"id": rid, "source": source, "camera": item["camera"], "captured_at": item.get("captured_at") or "",
                     "evidence_link": link or (dst.relative_to(ROOT).as_posix() if have else ""), "opencv_summary": summary,
                     "agent_decision": agent_decision(tr), "agent_reason": agent_reason(tr), "expert_verdict": "", "expert_note": "",
                     "_image": dst.relative_to(ROOT).as_posix() if have else ""})

    for src, items in chosen.items():
        for i, it in enumerate(sorted(items, key=lambda x: (str(x["captured_at"]), str(x["camera"]))), 1):
            add(f"{prefix[src]}-{i:02d}", src, it)
    for i, c in enumerate(failure_rows(), 1):
        rid = f"FIJ-{i:02d}"
        ev = c.get("evidence_image")
        note = ""
        if not ev and c.get("photo"):  # event-level case: no perturbed picture; show the clean photo it ran on
            ev, note = f"docs/evidence/eval/failure_injection/{c['photo']}__clean.jpg", " (the photo shown is the unperturbed one; the fault is in the event, not in the picture)"
        src_img = ROOT / ev if ev else None
        dst = IMG_DIR / f"{rid}.jpg"
        ok = save_image(src_img, dst) if src_img else False
        rows.append({"id": rid, "source": "failure_injection", "camera": c.get("camera") or "", "captured_at": "",
                     "evidence_link": dst.relative_to(ROOT).as_posix() if ok else "(no image: event-level case)",
                     "opencv_summary": f"INJECTED FAULT: {c['perturbation']}{note}", "agent_decision": c["observed"].split("; outcome=")[-1][:200],
                     "agent_reason": f"observed: {c['observed'][:420]}",
                     "expert_verdict": "", "expert_note": "", "_image": dst.relative_to(ROOT).as_posix() if ok else ""})
    for i, it in enumerate(examples, 1):
        add(f"EX-{i}", "example", it, ex=True)
    return rows


def write(rows: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    # a contact sheet: image + decision per row (static, relative links)
    cards = []
    for r in rows:
        img = f'<img src="expert_review/{html.escape(r["id"])}.jpg" width="480">' if r.get("_image") else "<i>no image</i>"
        cards.append(f'<tr><td><b>{html.escape(r["id"])}</b><br>{html.escape(r["source"])}<br>{html.escape(r["camera"])}<br>{html.escape(r["captured_at"])}</td>'
                     f'<td>{img}</td><td><b>{html.escape(r["agent_decision"])}</b><br>{html.escape(r["agent_reason"])}<br><small>{html.escape(r["opencv_summary"])}</small></td></tr>')
    (OUT_DIR / "expert_review_sheet.html").write_text(
        "<!doctype html><meta charset=utf-8><title>Expert review sheet</title><style>body{font:14px sans-serif;margin:1rem}td{vertical-align:top;border-bottom:1px solid #ccc;padding:6px}</style>"
        "<h2>Expert review sheet (fill the verdicts in expert_review.csv)</h2><table>" + "".join(cards) + "</table>", encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fetch-s3", action="store_true", help="download the live frames (read-only S3 GET; needs awscrt)")
    ap.add_argument("--force", action="store_true", help="overwrite a sheet that already holds expert verdicts")
    a = ap.parse_args(argv)
    if CSV_PATH.exists() and not a.force:
        from eval.score_expert_review import is_example

        with open(CSV_PATH, encoding="utf-8", newline="") as fh:
            filled = [r["id"] for r in csv.DictReader(fh) if (r.get("expert_verdict") or "").strip() and not is_example(r)]
        if filled:
            print(f"refusing to overwrite {CSV_PATH}: {len(filled)} expert verdicts present (use --force)")
            return 1
    rows = build(a.fetch_s3)
    write(rows)
    from collections import Counter

    print(json.dumps(Counter(r["source"] for r in rows)), "->", CSV_PATH)
    print("no image:", [r["id"] for r in rows if not r["_image"]])
    return 0


if __name__ == "__main__":
    sys.exit(main())
