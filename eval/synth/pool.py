"""Photo pool for virtual traps (SIMULATED DATA): photos that name no trap and no time, to be assigned by the demo.

    uv run --project lambda/vision python -m eval.synth.pool [--config eval/synth/pool.json] [--out data/synth/pool]

The FlyBot demo's daily service gives ~35 virtual traps across Vietnam one or two photos a day, picking from this pool
by the count its own series wants. So the visible stamp says only "SIMULATED", `capturedAt` is null, and the rows are:

    lure     trueCount spread over 0..`max` in buckets of `bucket` (about `perBucket` photos each), on both real
             empty-trap cameras (the 768 px one only up to `smallMaxCount` flies); flies may touch when the floor
             is crowded (`maxOverlapIoUSteps`), as on a real trap at 40+ flies
    sticky   one 7-day card per `dailyCatch` value, every slot of the schedule, flies accumulating on the card
    defects  per defect type (eval/synth/defects.py) `defects.lure` lure photos and `defects.sticky` sticky photos,
             with the expected agent action; `previousPhotoId` names the clean photo to compare with (op 8)

Same manifest schema as the series (flybot-synth-manifest/1); photo ids synth-pool-...
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from eval.synth import FILE_PREFIX, ROOT, defects
from eval.synth.flies import DEFAULT_DATASET
from eval.synth.generate import PHOTO, Context, place_flies, render_photo, series_photo, write_manifest

DEFAULT_CONFIG = Path(__file__).resolve().parent / "pool.json"
LURE_STREAM, DEFECT_STREAM = 11, 12


def lure_photo(ctx: Context, out: Path, trap: dict, n: int, index: int, photo_id: str, defect: str | None = None) -> dict:
    """One stand-alone lure photo with `n` flies (no series, no time)."""
    prng = ctx.rng(trap["id"], LURE_STREAM, index)
    pool = ctx.pool(trap["backgroundCamera"])
    bg = pool[int(prng.integers(len(pool)))]
    base, region, floor_w = ctx.bgs.get(bg["file"])
    long_side = ctx.fly_size(trap, floor_w)
    specs = place_flies(base, ctx.where(base, region, long_side), ctx.bank, n, long_side, [], prng,
                        ctx.cfg.get("maxOverlapIoUSteps") or float(ctx.cfg.get("maxOverlapIoU", 0.05)))
    row = render_photo(ctx, out, trap=trap, base=base, bg=bg, specs=specs, new_at=None, prng=prng,
                       defect_rng=ctx.rng(trap["id"], DEFECT_STREAM, index), defect=defect, stamp_text="SIMULATED",
                       photo_id=photo_id, file_rel=f"images/lure/{FILE_PREFIX}{photo_id.removeprefix('synth-')}.jpg",
                       captured_at=None, extra={"localDate": None, "slot": None, "cardId": None, "cardChanged": False})
    row["curveNew"], row["notPlaced"] = n, max(0, n - row["trueCount"])
    return row


def sticky_trap(cfg: dict, j: int, catch: float) -> dict:
    s = cfg["sticky"]
    return {"id": f"pool-sticky-c{j + 1:02d}", "name": f"Pool sticky card {j + 1}", "trapKind": "sticky",
            "backgroundCamera": s["backgroundCamera"], "cardDays": int(s.get("cardDays", 7)),
            "flyScale": float(s.get("flyScale", 0.04)), "cap": int(s.get("cap", 300)),
            "cardCoverage": float(s.get("cardCoverage", 0.84)),
            "wave": {"baseline": float(catch), "amplitude": 0.0, "noise": float(s.get("noise", 0.2))}}


def sticky_naming(suffix: str = ""):
    def name(trap, day, slot, _local):
        pid = f"synth-{trap['id']}-d{day + 1}-s{slot}{suffix}"
        return pid, f"images/sticky/{FILE_PREFIX}{pid.removeprefix('synth-')}.jpg", None, "SIMULATED"
    return name


def build(cfg: dict, out: Path, dataset: Path = DEFAULT_DATASET, log=print) -> dict:
    t0 = time.perf_counter()
    ctx = Context(cfg, dataset, log)
    out.mkdir(parents=True, exist_ok=True)
    lc = cfg["lure"]
    lure_traps = [{"trapKind": "lure", "name": f"Pool lure ({t['backgroundCamera']})", **t} for t in lc["traps"]]
    small, big = lure_traps[0], lure_traps[-1]
    photos, last_clean = [], {}
    rng = ctx.rng("pool-lure", PHOTO)
    k = 0
    for lo in range(0, int(lc["max"]) + 1, int(lc["bucket"])):
        for i in range(int(lc["perBucket"])):
            n = min(int(lc["max"]), lo + int(rng.integers(int(lc["bucket"]))))
            trap = small if (n <= int(small.get("smallMaxCount", 40)) and i % 2 == 0) else big
            k += 1
            row = lure_photo(ctx, out, trap, n, k, f"synth-pool-lure-{k:04d}")
            row["previousPhotoId"] = last_clean.get(trap["id"])
            last_clean[trap["id"]] = row["photoId"]
            photos.append(row)
    log(f"pool lure: {len(photos)} photos ({time.perf_counter() - t0:.0f} s)")
    sc = cfg["sticky"]
    s_traps = [sticky_trap(cfg, j, c) for j, c in enumerate(sc["dailyCatch"])]
    n_slots = len(ctx.schedule)
    for trap in s_traps:
        prev = None
        for d in range(int(sc.get("cardDays", 7))):
            for s in range(n_slots):
                row = series_photo(ctx, out, trap, d, s, naming=sticky_naming(), defect=None)
                row.update(localDate=None, dayInCard=d + 1, previousPhotoId=prev)
                prev = row["photoId"]
                photos.append(row)
    log(f"pool sticky: {sum(1 for p in photos if p['trapKind'] == 'sticky')} photos ({time.perf_counter() - t0:.0f} s)")
    dc = cfg.get("defects", {})
    drng = ctx.rng("pool-defects", PHOTO)
    for kind in defects.TYPES:
        for i in range(int(dc.get("lure", 0))):
            trap = lure_traps[i % len(lure_traps)]
            n = int(drng.integers(3, 30))
            k += 1
            row = lure_photo(ctx, out, trap, n, k, f"synth-pool-lure-{kind.replace('_', '-')}-{i + 1}", defect=kind)
            row["previousPhotoId"] = last_clean.get(trap["id"])
            photos.append(row)
        for i in range(int(dc.get("sticky", 0))):
            trap = s_traps[(i * 2 + 1) % len(s_traps)]
            d, s = int(drng.integers(1, int(sc.get("cardDays", 7)))), int(drng.integers(1, n_slots))
            row = series_photo(ctx, out, trap, d, s, naming=sticky_naming(f"-{kind.replace('_', '-')}"), defect=kind)
            row.update(localDate=None, dayInCard=d + 1, previousPhotoId=f"synth-{trap['id']}-d{d + 1}-s{s - 1}")
            photos.append(row)
    log(f"pool defects: {sum(1 for p in photos if p.get('defect'))} photos ({time.perf_counter() - t0:.0f} s)")
    traps = [{"id": t["id"], "name": t["name"], "trapKind": "lure", "platform": "ezviz", "lat": None, "lon": None,
              "backgroundCamera": t["backgroundCamera"], "wave": {}, "cap": None, "cardDays": None, "cards": [],
              "dailyLevels": []} for t in lure_traps]
    traps += [{"id": t["id"], "name": t["name"], "trapKind": "sticky", "platform": "camos", "lat": None, "lon": None,
               "backgroundCamera": t["backgroundCamera"], "wave": t["wave"], "cap": t["cap"], "cardDays": t["cardDays"],
               "cards": [{"id": f"{t['id']}-card-1", "from": None,
                          "quad": [[round(x, 1), round(y, 1)] for x, y in ctx.card(t, 0)["quad"]],
                          "background": ctx.card(t, 0)["background"]["file"].relative_to(ROOT).as_posix()}],
               "dailyLevels": []} for t in s_traps]
    m = write_manifest(ctx, out, photos, traps, extra={"pool": True})
    log(f"wrote {len(photos)} pool photos to {out} in {time.perf_counter() - t0:.0f} s")
    return m


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--out", type=Path, default=None, help="default data/synth/<config name>")
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    a = ap.parse_args(argv)
    cfg = json.loads(a.config.read_text(encoding="utf-8"))
    build(cfg, (a.out or ROOT / "data" / "synth" / cfg["name"]).resolve(), a.dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
