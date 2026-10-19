"""Copy a few photos of a generated set, with their manifest rows, into eval/synth/sample/ (committed).

    uv run --project lambda/vision python -m eval.synth.sample [--src data/synth/demo-30d] [--dst eval/synth/sample]

Picks per lure trap the busiest photo, for each sticky trap the fullest card before a change plus the clean card
after it, and one photo damaged on purpose per trap kind, so the sample shows the count behaviours and the defects.
The sample manifest keeps the full set's header (config, seed, credits) and lists only the copied photos.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from eval.synth import ROOT
from eval.synth.generate import readme, write_csv


def pick(photos: list[dict]) -> list[dict]:
    chosen = []
    for trap in sorted({p["trap"] for p in photos}):
        rows = [p for p in photos if p["trap"] == trap]
        if rows[0]["trapKind"] == "lure":
            chosen.append(max(rows, key=lambda p: (p["trueCount"], p["capturedAt"])))
            continue
        for i, p in enumerate(rows):
            if p["cardChanged"] and i > 0:
                chosen += [rows[i - 1], p]
                break
        else:
            chosen.append(max(rows, key=lambda p: p["trueCount"]))
    for kind, want in (("lure", "occluded_hand"), ("sticky", "camera_moved")):  # one damaged photo per trap kind
        hit = next((p for p in photos if p["trapKind"] == kind and (p.get("defect") or {}).get("type") == want), None)
        if hit and hit not in chosen:
            chosen.append(hit)
    return sorted(chosen, key=lambda p: (p["trap"], p["capturedAt"] or ""))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", type=Path, default=ROOT / "data" / "synth" / "demo-30d")
    ap.add_argument("--dst", type=Path, default=Path(__file__).resolve().parent / "sample")
    a = ap.parse_args(argv)
    m = json.loads((a.src / "manifest.json").read_text(encoding="utf-8"))
    photos = pick(m["photos"])
    if a.dst.exists():
        shutil.rmtree(a.dst)  # only ever eval/synth/sample (or the --dst given): regenerated from the set
    for p in photos:
        (a.dst / p["file"]).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(a.src / p["file"], a.dst / p["file"])
    keep = {p["trap"] for p in photos}
    m.update(photos=photos, traps=[t for t in m["traps"] if t["id"] in keep],
             description=(m.get("description") or "") + f" Sample: {len(photos)} photos copied from the full set.")
    (a.dst / "manifest.json").write_text(json.dumps(m, indent=1, ensure_ascii=False), encoding="utf-8", newline="\n")
    write_csv(a.dst / "manifest.csv", photos)
    (a.dst / "README.md").write_text(readme(m), encoding="utf-8", newline="\n")
    size = sum(f.stat().st_size for f in a.dst.rglob("*") if f.is_file())
    print(f"sample: {len(photos)} photos, {size / 1e6:.2f} MB -> {a.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
