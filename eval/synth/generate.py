"""Generate a labelled set of synthetic trap photos (SIMULATED DATA) from a config such as eval/synth/demo_30d.json.

    uv run --project lambda/vision python -m eval.synth.generate [--config eval/synth/demo_30d.json]
        [--out data/synth/demo-30d] [--from 2026-09-19] [--to 2026-10-18] [--day 2026-10-25] [--days N]
        [--traps demo-lure-a,demo-sticky-d] [--seed S] [--dataset DIR]

Deterministic per trap and day: every random draw comes from a stream keyed by (seed, trap id, day, slot), so one
trap-day generated alone (`--day`, the daily service) gives the same photo ids and bytes as the bulk run. Sticky
cards are replayed from the card's first photo, so their flies are the same either way.

Writes into --out (git-ignored under data/synth/):
    images/<trap>/SYNTH_<trap>_<YYYYmmddTHHMM>Z.jpg   composites, label in the filename, EXIF, XMP, comment, stamp
    manifest.json      schema eval/synth/manifest.schema.json: config, credits, traps, cards, one row per photo
    manifest.csv       the same rows flattened (boxes as JSON)
    README.md          what the set is, how it was made, credits (CC BY 4.0 dataset)
A configured share of photos is damaged on purpose (eval/synth/defects.py) and says so in `defect`.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import time
import zlib
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
from eval.synth import BACKGROUND_CREDIT, DATASET_CREDIT, FILE_PREFIX, LABEL, ROOT, SCHEMA, curve, defects
from eval.synth import composite as C
from eval.synth.flies import DEFAULT_DATASET, FlyCrop, load_bank

DEFAULT_CONFIG = Path(__file__).resolve().parent / "demo_30d.json"
EXCLUDED = ROOT / "data" / "excluded_cameras.csv"
PLATFORM = {"lure": "ezviz", "sticky": "camos"}  # default camera platform per trap kind (policy: ISP fixes on camos)
# random streams: (seed, trap key, STREAM, ...)
LEVELS, COUNTS, PHOTO, CARD, DEFECT = 1, 2, 3, 4, 5


def parse_offset(s: str) -> timezone:
    sign = -1 if s.startswith("-") else 1
    hh, mm = s.lstrip("+-").split(":")
    return timezone(sign * timedelta(hours=int(hh), minutes=int(mm)))


def trap_key(trap_id: str) -> int:
    return zlib.crc32(trap_id.encode("utf-8"))


def backgrounds_by_camera(manifest: Path) -> dict[str, list[dict]]:
    """Real empty-trap photos per camera from a samples manifest; cameras in data/excluded_cameras.csv never used."""
    excluded = set()
    if EXCLUDED.exists():
        with open(EXCLUDED, encoding="utf-8", newline="") as fh:
            excluded = {r["device_serial"] for r in csv.DictReader(fh)}
    out: dict[str, list[dict]] = {}
    with open(manifest, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["device_serial"] in excluded:
                continue
            out.setdefault(r["device_serial"], []).append(
                {"file": (manifest.parent / r["image_path"]).resolve(), "camera": r["device_serial"],
                 "capturedAt": r["captured_at"]})
    return out


class Backgrounds:
    """Decoded background photos and their trap floor polygon (sentinel.vision.card.find_lure_floor), cached."""

    def __init__(self):
        self._cache: dict[Path, tuple[np.ndarray, list | None, float]] = {}

    def get(self, path: Path) -> tuple[np.ndarray, list | None, float]:
        """(float32 BGR with the date overlay erased, floor polygon px or None, floor width px)."""
        if path not in self._cache:
            from sentinel.vision.card import find_lure_floor, roi_polygon_px

            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(path)
            poly = roi_polygon_px(find_lure_floor(img))
            width = float(np.ptp(np.asarray(poly)[:, 0])) if poly else float(img.shape[1])
            self._cache[path] = (C.erase_osd(img.astype(np.float32)), poly, width)
        return self._cache[path]


def card_in_frame(base: np.ndarray, rng: np.random.Generator, coverage: float = 0.84,
                  aspect: float = 25 / 15) -> tuple[np.ndarray, list, float]:
    """A synthetic yellow sticky card (sentinel.vision.synth.make_card, no flies) filling most of the frame, as the
    ESP32 trap's assembly guide asks (15 x 25 cm card, 15-25 cm from the lens). Returns (image, quad, card long side).

    No real sticky-card photo exists (eval/RESULTS.md, op 1), so the card itself is synthetic too; the manifest says so.
    """
    from sentinel.vision.card import order_corners
    from sentinel.vision.synth import make_card, paste_card

    H, W = base.shape[:2]
    cw = min(coverage * W, 0.9 * H * aspect) * rng.uniform(0.97, 1.0)
    ch = cw / aspect
    rect = cv2.boxPoints(((W / 2 + rng.uniform(-0.02, 0.02) * W, H / 2 + rng.uniform(-0.02, 0.02) * H), (cw, ch),
                          rng.uniform(-3, 3)))
    quad = order_corners(rect.astype(np.float32))
    card, _ = make_card(int(cw), int(ch), n_flies=0, rng=rng, grid=True)
    out = paste_card(np.clip(base, 0, 255).astype(np.uint8), card, quad).astype(np.float32)
    # the card takes the frame's light: the floor's smoothed brightness, relative to its median under the card
    m = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(m, np.round(quad).astype(np.int32), 255)
    small = cv2.resize(cv2.cvtColor(np.clip(base, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY), (W // 8, H // 8),
                       interpolation=cv2.INTER_AREA).astype(np.float32)
    lum = cv2.resize(cv2.GaussianBlur(small, (0, 0), W / 48.0), (W, H), interpolation=cv2.INTER_LINEAR)
    # light falloff only (sigma W/6 at full size), not the floor's details (bait port, stains)
    shade = np.clip(lum / max(1.0, float(np.median(lum[m > 0]))), 0.8, 1.08)
    out[m > 0] *= shade[m > 0][:, None]
    return out, [[float(x), float(y)] for x, y in quad], float(cw)


def place_flies(img: np.ndarray, where: tuple[np.ndarray, np.ndarray], bank: list[FlyCrop], n: int,
                long_side: float, existing: list[list[float]], rng: np.random.Generator,
                max_iou: float | list[float]) -> list[dict]:
    """Choose `n` fly placements (no paste yet): crop, centre, size, angle, blur; skip ones that would overlap.
    `where` = (ys, xs) of the allowed centres (np.nonzero of composite.placement_mask). `max_iou` may be a list of
    growing limits: flies that find no room at the first one are placed at the next (a crowded trap: flies touch)."""
    ys, xs = where
    placed: list[dict] = []
    boxes = list(existing)
    if len(xs) == 0:
        return placed
    W = img.shape[1]
    steps = list(max_iou) if isinstance(max_iou, list | tuple) else [max_iou]
    for limit in steps:
        for _ in range(n - len(placed)):
            for _try in range(60):
                k = int(rng.integers(len(xs)))
                fly = bank[int(rng.integers(len(bank)))]
                spec = {"fly": fly, "center": (float(xs[k]), float(ys[k])),
                        "long_side": float(long_side * rng.uniform(0.8, 1.2)), "angle": float(rng.uniform(0, 360)),
                        "blur": float(max(0.3, rng.uniform(0.4, 0.9) * W / 1200.0))}
                half = spec["long_side"] * 0.5
                approx = [spec["center"][0] - half, spec["center"][1] - half, spec["center"][0] + half,
                          spec["center"][1] + half]
                if all(C.iou(approx, b) <= limit for b in boxes):
                    spec["approx"] = approx
                    placed.append(spec)
                    boxes.append(approx)
                    break
        if len(placed) >= n:
            break
    return placed


def render(base: np.ndarray, specs: list[dict], blend: str) -> tuple[np.ndarray, list[dict]]:
    """Paste every spec on a copy of `base`; returns the image and the specs that fit, with their boxes."""
    img = base.copy()
    kept = []
    for s in specs:
        box = C.paste_fly(img, s["fly"], s["center"], s["long_side"], s["angle"], s["blur"], blend)
        if box is not None:
            kept.append({**s, "box": box})
    return img, kept


def box_row(s: dict, W: int, H: int, added_at: str | None = None, M: np.ndarray | None = None,
            hidden: np.ndarray | None = None) -> dict:
    x0, y0, x1, y1 = s["box"]
    px = [round(x0, 1), round(y0, 1), round(x1 - x0, 1), round(y1 - y0, 1)]
    if M is not None:  # the camera moved: the fly moved with the frame
        px = defects.move_box(px, M)
    row = {"x": round(px[0] / W, 5), "y": round(px[1] / H, 5), "w": round(px[2] / W, 5), "h": round(px[3] / H, 5),
           "px": px, "flyCrop": s["fly"].id, "angle": round(s["angle"], 1), "longSidePx": round(s["long_side"], 1)}
    if added_at:
        row["addedAt"] = added_at
    if hidden is not None:
        cx, cy = int(px[0] + px[2] / 2), int(px[1] + px[3] / 2)
        inside = 0 <= cx < W and 0 <= cy < H
        row["hidden"] = bool(not inside or hidden[cy, cx])
    return row


def git_head() -> str | None:
    """Short HEAD hash, with "+local" when eval/synth has uncommitted changes (the hash alone would not be the code)."""
    try:
        head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              check=False, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no", "--", "eval/synth"],
                               capture_output=True, text=True, check=False, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return (head + ("+local" if dirty else "")) or None


def rel(p: Path) -> str:
    try:
        return Path(p).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


# ---------------------------------------------------------------- context and one photo


class Context:
    """Everything shared by the photos of one run: config, fly bank, backgrounds, caches."""

    def __init__(self, cfg: dict, dataset: Path = DEFAULT_DATASET, log=print):
        self.cfg, self.log = cfg, log
        self.seed = int(cfg["seed"])
        self.tz = parse_offset(cfg.get("utcOffset", "+07:00"))
        self.schedule = [tuple(int(x) for x in s.split(":")) for s in cfg.get("schedule", ["09:00"])]
        self.weights = curve.slot_weights([h + m / 60 for h, m in self.schedule])
        self.start = date.fromisoformat(cfg.get("startDate", "2026-01-01"))
        self.blend = cfg.get("blend", "multiply")
        fb = cfg.get("flyBank", {})
        self.bank_info = {"splits": list(fb.get("splits", ["test", "valid"]))}
        self.bank = load_bank(dataset, tuple(self.bank_info["splits"]), int(fb.get("maxCrops", 1200)),
                              int(fb.get("seed", 7)))
        self.bank_info["crops"] = len(self.bank)
        log(f"fly bank: {len(self.bank)} crops from {Path(dataset).name} {self.bank_info['splits']}")
        self.pools = backgrounds_by_camera((ROOT / cfg.get("backgrounds", "data/samples/samples_manifest.csv")).resolve())
        self.bgs = Backgrounds()
        # no commit hash in the JPEG: a day regenerated after a later commit keeps its bytes (the manifest has the hash)
        self.software = f"FlyBot synthetic generator eval/synth ({SCHEMA})"
        self._where: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
        self._cards: dict[tuple[str, int], dict] = {}
        self._levels: dict[str, list[float]] = {}
        self._sticky: dict[str, dict] = {}

    def rng(self, trap_id: str, *parts: int) -> np.random.Generator:
        return np.random.default_rng([self.seed, trap_key(trap_id), *parts])

    def pool(self, camera: str) -> list[dict]:
        pool = self.pools.get(camera)
        if not pool:
            raise ValueError(f"no background photos for camera {camera}")
        return pool

    def where(self, base: np.ndarray, region, long_side: float) -> tuple[np.ndarray, np.ndarray]:
        key = (id(base), round(long_side, 2))
        if key not in self._where:  # whole fly on the floor / card; the rotated crop (~1.3 x) inside the frame
            self._where[key] = np.nonzero(C.placement_mask(base, region, margin=0.6 * long_side,
                                                           frame_margin=1.3 * long_side))
        return self._where[key]

    def levels(self, trap: dict, day: int) -> list[float]:
        """Daily curve levels from day 0 through `day` (prefix-stable: computing more days never changes earlier ones)."""
        have = self._levels.get(trap["id"], [])
        if len(have) <= day:
            have = curve.daily_levels(curve.Wave(**trap.get("wave", {})), day + 1, self.rng(trap["id"], LEVELS))
            self._levels[trap["id"]] = have
        return have[: day + 1]

    def slot_counts(self, trap: dict, day: int) -> list[int]:
        lv = self.levels(trap, day)[day]
        rng = self.rng(trap["id"], COUNTS, day)
        if trap["trapKind"] == "lure":
            return curve.lure_counts([lv], self.weights, rng, int(trap.get("cap", 60)))[0]
        return curve.sticky_new([lv], self.weights, rng)[0]

    def card(self, trap: dict, index: int) -> dict:
        """Sticky card `index` of a trap: synthetic card on one of the camera's frames (cached)."""
        key = (trap["id"], index)
        if key not in self._cards:
            rng = self.rng(trap["id"], CARD, index)
            pool = self.pool(trap["backgroundCamera"])
            bg = pool[int(rng.integers(len(pool)))]
            base0, _, _ = self.bgs.get(bg["file"])
            base, quad, long_side = card_in_frame(base0, rng, float(trap.get("cardCoverage", 0.84)))
            self._cards[key] = {"id": f"{trap['id']}-card-{index + 1}", "index": index, "background": bg, "base": base,
                                "quad": quad, "floor_w": long_side}
        return self._cards[key]

    def fly_size(self, trap: dict, floor_w: float) -> float:
        # a share of the trap floor width (lure) or of the card's long side (sticky)
        return float(trap.get("flyScale", self.cfg.get("flyScale", 0.11))) * floor_w

    def local_time(self, day: int, slot: int, seconds: int = 0) -> datetime:
        d = self.start + timedelta(days=day)
        hh, mm = self.schedule[slot]
        return datetime(d.year, d.month, d.day, hh, mm, seconds, tzinfo=self.tz)


def _iso(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def sticky_state(ctx: Context, trap: dict, day: int, slot: int) -> tuple[dict, list[dict], list[dict], bool,
                                                                         np.random.Generator, int, int]:
    """Replay a sticky card from its first photo up to (day, slot).

    Returns (card, flies already stuck, new flies of this photo, card changed on this photo, the photo's rng positioned
    after placement, capture seconds, flies the curve asked for). Consumption of each photo's rng: seconds, then placement.
    """
    days_per_card = int(trap.get("cardDays", 7))
    c = day // days_per_card
    card = ctx.card(trap, c)
    first = (c * days_per_card, 0)
    st = ctx._sticky.get(trap["id"])
    n_slots = len(ctx.schedule)
    prev_pos = (day, slot - 1) if slot > 0 else (day - 1, n_slots - 1)
    if st and st["card"] == c and st["pos"] == prev_pos:
        stuck, pos = list(st["stuck"]), (day, slot)
    else:
        stuck, pos = [], first
    long_side = ctx.fly_size(trap, card["floor_w"])
    where = ctx.where(card["base"], card["quad"], long_side)
    cap = int(trap.get("cap", 300))
    while True:
        d, s = pos
        prng = ctx.rng(trap["id"], PHOTO, d, s)
        seconds = int(prng.integers(3, 16))
        changed = pos == first and c > 0
        n_new = 0 if changed else ctx.slot_counts(trap, d)[s]
        new = place_flies(card["base"], where, ctx.bank, min(n_new, max(0, cap - len(stuck))), long_side,
                          [f["approx"] for f in stuck], prng, float(ctx.cfg.get("maxOverlapIoU", 0.05)))
        at = _iso(ctx.local_time(d, s, seconds))
        for f in new:
            f["addedAt"] = at
        if pos == (day, slot):
            ctx._sticky[trap["id"]] = {"card": c, "pos": pos, "stuck": stuck + new}
            return card, stuck, new, changed, prng, seconds, n_new
        stuck = stuck + new
        pos = (d, s + 1) if s + 1 < n_slots else (d + 1, 0)


def render_photo(ctx: Context, out: Path, *, trap: dict, base: np.ndarray, bg: dict, specs: list[dict], new_at: str | None,
                 prng: np.random.Generator, defect_rng: np.random.Generator, defect: str | None, stamp_text: str | None,
                 photo_id: str, file_rel: str, captured_at: str | None, extra: dict) -> dict:
    """Render, damage (optional), stamp, label and write one photo; returns its manifest row."""
    kind = trap["trapKind"]
    platform = trap.get("platform") or PLATFORM[kind]
    img, kept = render(base, specs, ctx.blend)
    H, W = img.shape[:2]
    gain, contrast = float(prng.uniform(0.94, 1.06)), float(prng.uniform(0.95, 1.05))
    noise, quality = float(prng.uniform(0.8, 2.0)), int(prng.integers(80, 93))
    u8 = C.finish(img, prng, None, gain, contrast, noise)
    hidden = M = None
    defect_row = None
    if defect:
        u8, params, hidden, M = defects.apply(u8, defect, defect_rng, kind)
        defect_row = {"type": defect, **params, "expected": defects.expected_for(defect, platform)}
    if stamp_text:
        u8 = C.stamp(u8, stamp_text)
    fields = {"synthetic": "true", "photoId": photo_id, "trap": trap["id"], "trapKind": kind,
              "capturedAt": captured_at or "none", "trueCount": len(kept), "defect": defect or "none"}
    data = C.encode_jpeg(u8, quality, fields, ctx.software)
    path = out / file_rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    boxes = [box_row(k, W, H, k.get("addedAt") if kind == "sticky" else None, M, hidden) for k in kept]
    return {
        "photoId": photo_id, "synthetic": True, "file": file_rel, "trap": trap["id"], "trapKind": kind,
        "platform": platform, "capturedAt": captured_at, **extra, "trueCount": len(kept),
        "visibleCount": sum(1 for b in boxes if not b.get("hidden")),
        "newSincePrevious": sum(1 for k in kept if k.get("addedAt") == new_at) if new_at else len(kept),
        "width": W, "height": H, "boxes": boxes, "defect": defect_row,
        "background": {"file": rel(bg["file"]), "camera": bg["camera"], "capturedAt": bg["capturedAt"],
                       "stickyCard": "synthetic yellow card (sentinel.vision.synth.make_card)" if kind == "sticky" else None},
        "render": {"blend": ctx.blend, "jpegQuality": quality, "gain": round(gain, 4), "contrast": round(contrast, 4),
                   "noiseSigma": round(noise, 3)},
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def pick_defect(ctx: Context, trap: dict, day: int, slot: int) -> tuple[str | None, np.random.Generator]:
    """Whether this photo is damaged on purpose, and how: its own stream, so it never shifts the flies."""
    rng = ctx.rng(trap["id"], DEFECT, day, slot)
    spec = trap.get("defects", ctx.cfg.get("defects")) or {}
    types = spec.get("types") or list(defects.TYPES)
    if rng.random() < float(spec.get("rate", 0.0)):
        return types[int(rng.integers(len(types)))], rng
    return None, rng


def series_photo(ctx: Context, out: Path, trap: dict, day: int, slot: int, stamp: str | None = "full",
                 naming=None, defect: str | None = "auto") -> dict:
    """Photo (day, slot) of a trap's series (lure: independent photos; sticky: the card's accumulated flies)."""
    kind = trap["trapKind"]
    if kind == "sticky":
        card, stuck, new, changed, prng, seconds, n_new = sticky_state(ctx, trap, day, slot)
        base, bg, specs = card["base"], card["background"], stuck + new
        extra_card = {"cardId": card["id"], "cardChanged": changed}
    else:
        prng = ctx.rng(trap["id"], PHOTO, day, slot)
        seconds = int(prng.integers(3, 16))
        pool = ctx.pool(trap["backgroundCamera"])
        bg = pool[int(prng.integers(len(pool)))]
        base, region, floor_w = ctx.bgs.get(bg["file"])
        long_side = ctx.fly_size(trap, floor_w)
        n_new = ctx.slot_counts(trap, day)[slot]
        specs = place_flies(base, ctx.where(base, region, long_side), ctx.bank, n_new, long_side, [], prng,
                            float(ctx.cfg.get("maxOverlapIoU", 0.05)))
        extra_card = {"cardId": None, "cardChanged": False}
    local = ctx.local_time(day, slot, seconds)
    at = _iso(local)
    slot_utc = ctx.local_time(day, slot).astimezone(UTC).strftime("%Y%m%dT%H%M")
    if naming:
        photo_id, file_rel, captured, stamp_text = naming(trap, day, slot, local)
    else:
        photo_id = f"synth-{trap['id']}-{slot_utc}Z"
        file_rel = f"images/{trap['id']}/{FILE_PREFIX}{trap['id']}_{slot_utc}Z.jpg"
        captured = at
        stamp_text = (f"SIMULATED {local.strftime('%Y-%m-%d %H:%M')} {trap['id']}" if stamp == "full"
                      else "SIMULATED" if stamp == "plain" else None)
    if defect == "auto":
        defect, drng = pick_defect(ctx, trap, day, slot)
    else:
        drng = ctx.rng(trap["id"], DEFECT, day, slot, 1)
    for f in specs:
        f.setdefault("addedAt", at)
    row = render_photo(ctx, out, trap=trap, base=base, bg=bg, specs=specs, new_at=at, prng=prng, defect_rng=drng,
                       defect=defect, stamp_text=stamp_text, photo_id=photo_id, file_rel=file_rel, captured_at=captured,
                       extra={"localDate": (ctx.start + timedelta(days=day)).isoformat(), "slot": slot,
                              **extra_card, "dailyLevel": ctx.levels(trap, day)[day]})
    row["curveNew"] = n_new
    row["notPlaced"] = max(0, n_new - row["newSincePrevious"]) if not extra_card["cardChanged"] else 0
    return row


# ---------------------------------------------------------------- whole runs


def stamp_mode(cfg: dict) -> str | None:
    s = cfg.get("stamp", "full")
    return "full" if s is True else (None if s is False else s)


def generate(cfg: dict, out: Path, dataset: Path = DEFAULT_DATASET, log=print, first: date | None = None,
             last: date | None = None, ctx: Context | None = None) -> dict:
    """Every trap of `cfg`, local dates `first`..`last` (default: startDate and `days` of the config)."""
    t0 = time.perf_counter()
    ctx = ctx or Context(cfg, dataset, log)
    first = first or ctx.start
    last = last or ctx.start + timedelta(days=int(cfg.get("days", 1)) - 1)
    d0, d1 = (first - ctx.start).days, (last - ctx.start).days
    if d0 < 0 or d1 < d0:
        raise ValueError(f"date range {first}..{last} must start on or after startDate {ctx.start}")
    out.mkdir(parents=True, exist_ok=True)
    photos, traps_out = [], []
    for trap in cfg["traps"]:
        rows = []
        for d in range(d0, d1 + 1):
            for s in range(len(ctx.schedule)):
                rows.append(series_photo(ctx, out, trap, d, s, stamp_mode(cfg)))
            if (d - d0 + 1) % 10 == 0 or d == d1:
                log(f"{trap['id']}: day {d - d0 + 1}/{d1 - d0 + 1} ({time.perf_counter() - t0:.0f} s)")
        for prev, row in zip([None, *rows[:-1]], rows, strict=True):
            row["previousPhotoId"] = prev["photoId"] if prev else None
        photos += rows
        traps_out.append(trap_entry(ctx, trap, rows, d0, d1))
    manifest = write_manifest(ctx, out, photos, traps_out, extra={"dates": [first.isoformat(), last.isoformat()]})
    log(f"wrote {len(photos)} photos to {out} in {time.perf_counter() - t0:.0f} s")
    return manifest


def trap_entry(ctx: Context, trap: dict, rows: list[dict], d0: int, d1: int) -> dict:
    kind = trap["trapKind"]
    cards = []
    for r in rows:
        if r.get("cardId") and (not cards or cards[-1]["id"] != r["cardId"]):
            c = next(v for (tid, _), v in ctx._cards.items() if tid == trap["id"] and v["id"] == r["cardId"])
            cards.append({"id": c["id"], "from": r["capturedAt"], "quad": [[round(x, 1), round(y, 1)] for x, y in c["quad"]],
                          "background": rel(c["background"]["file"])})
    return {"id": trap["id"], "name": trap.get("name", trap["id"]), "trapKind": kind,
            "platform": trap.get("platform") or PLATFORM[kind], "lat": trap.get("lat"), "lon": trap.get("lon"),
            "backgroundCamera": trap["backgroundCamera"], "wave": trap.get("wave", {}), "cap": trap.get("cap"),
            "cardDays": int(trap.get("cardDays", 7)) if kind == "sticky" else None, "cards": cards,
            "dailyLevels": [{"date": (ctx.start + timedelta(days=d)).isoformat(), "level": ctx.levels(trap, d)[d]}
                            for d in range(d0, d1 + 1)]}


def write_manifest(ctx: Context, out: Path, photos: list[dict], traps: list[dict], extra: dict | None = None) -> dict:
    cfg = ctx.cfg
    manifest = {
        "schema": SCHEMA, "synthetic": True, "label": LABEL, "name": cfg.get("name"),
        "description": cfg.get("description"), "createdAt": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator": {"module": "eval.synth.generate", "git": git_head(), "opencv": cv2.__version__},
        "seed": ctx.seed, "config": cfg, **(extra or {}),
        "credits": {"flies": DATASET_CREDIT, "backgrounds": BACKGROUND_CREDIT, "flyBank": ctx.bank_info},
        "traps": traps, "photos": photos,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8",
                                       newline="\n")
    write_csv(out / "manifest.csv", photos)
    (out / "README.md").write_text(readme(manifest), encoding="utf-8", newline="\n")
    return manifest


CSV_COLUMNS = ["photoId", "synthetic", "file", "trap", "trapKind", "platform", "capturedAt", "localDate", "slot",
               "trueCount", "visibleCount", "newSincePrevious", "cardId", "cardChanged", "previousPhotoId", "width",
               "height", "dailyLevel", "defectType", "expectedFlag", "expectedActions", "backgroundFile",
               "backgroundCamera", "flyCrops", "boxes", "sha256"]


def write_csv(path: Path, photos: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for p in photos:
            d = p.get("defect") or {}
            w.writerow({**p, "synthetic": "true", "cardChanged": str(bool(p.get("cardChanged"))).lower(),
                        "defectType": d.get("type"), "expectedFlag": (d.get("expected") or {}).get("flag"),
                        "expectedActions": ";".join((d.get("expected") or {}).get("actions", [])),
                        "backgroundFile": p["background"]["file"], "backgroundCamera": p["background"]["camera"],
                        "flyCrops": ";".join(b["flyCrop"] for b in p["boxes"]),
                        "boxes": json.dumps([{k: b[k] for k in ("x", "y", "w", "h")} for b in p["boxes"]])})


def readme(m: dict) -> str:
    photos = m["photos"]
    per_trap: dict[str, int] = {}
    for p in photos:
        per_trap[p["trap"]] = per_trap.get(p["trap"], 0) + 1
    dates = [p["localDate"] for p in photos if p.get("localDate")]
    span = f"local dates {min(dates)} .. {max(dates)}" if dates else "no capture times (pool photos)"
    n_def = sum(1 for p in photos if p.get("defect"))
    lines = [
        f"# SIMULATED DATA: {m['name']}", "",
        f"**{m['label']}.**", "",
        "Every photo here is a composite made by `eval/synth` (flybot-aws): a real photo of an *empty* trap with",
        "fruit flies cut from a public dataset pasted in by OpenCV. None of it was captured in the field. The label is in",
        "the file name (`SYNTH_`), the JPEG metadata (EXIF ImageDescription, XMP `flybot:synthetic=\"true\"`, comment),",
        "a visible `SIMULATED` stamp, and `synthetic: true` on every manifest row.", "",
        f"- Photos: {len(photos)} ({', '.join(f'{k} {v}' for k, v in per_trap.items())}), {span}",
        f"- Damaged on purpose (`defect` set, with the action the policy expects): {n_def}",
        f"- Seed {m['seed']}, generator git {m['generator']['git']}, OpenCV {m['generator']['opencv']}",
        "- Manifest: `manifest.json` (schema `eval/synth/manifest.schema.json`), `manifest.csv`", "",
        "## Credits", "",
        f"- Flies: {m['credits']['flies']}. Crops from the {'/'.join(m['credits']['flyBank']['splits'])} splits.",
        f"- Backgrounds: {m['credits']['backgrounds']}. No photo with people is used (data/samples/README.md).",
        "- Sticky traps: the yellow card is synthetic too (no real sticky-card photo exists).", "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--out", type=Path, default=None, help="default data/synth/<config name>")
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--from", dest="first", type=date.fromisoformat, default=None, help="first local date")
    ap.add_argument("--to", dest="last", type=date.fromisoformat, default=None, help="last local date")
    ap.add_argument("--day", type=date.fromisoformat, default=None, help="one local date (= --from D --to D)")
    ap.add_argument("--days", type=int, default=None, help="number of days from startDate")
    ap.add_argument("--traps", default=None, help="comma-separated trap ids (default: all in the config)")
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args(argv)
    cfg = json.loads(a.config.read_text(encoding="utf-8"))
    if a.days:
        cfg["days"] = a.days
    if a.seed is not None:
        cfg["seed"] = a.seed
    if a.traps:
        keep = set(a.traps.split(","))
        cfg["traps"] = [t for t in cfg["traps"] if t["id"] in keep]
    first, last = (a.day, a.day) if a.day else (a.first, a.last)
    if first and not last:
        last = date.fromisoformat(cfg["startDate"]) + timedelta(days=int(cfg.get("days", 1)) - 1)
    if a.out:  # the set is named after its folder (demo-ahead, a daily folder...)
        cfg["name"] = a.out.resolve().name
    out = (a.out or ROOT / "data" / "synth" / cfg["name"]).resolve()
    generate(cfg, out, a.dataset, first=first, last=last)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
