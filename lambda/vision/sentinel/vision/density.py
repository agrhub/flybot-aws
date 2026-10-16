"""Op 7: orchard pest-pressure map from trap points (IDW + inpaint + blur, hot-zone contours).

    uv run python -m sentinel.vision.density TRAPS.json [--out DIR] [--hot 0.6]

TRAPS.json: a list of {"id", "value", and either "lat"+"lon" or "x"+"y" (metres)}; `value` is the trap's
pressure (e.g. flies per trap per day). `build_density_map(traps)` returns a JSON-serializable dict:

    grid {width, height, cell_m, origin, crs}, value_range [min, max], hot_threshold, hot_zones (GeoJSON
    FeatureCollection: one Polygon per hot zone, properties {zone, area_m2, peak, mean, traps}), traps
    (each with its grid cell and interpolated value), time_ms, evidence {png, geojson}

Method: the trap layout is projected to local metres (equirectangular around the centroid for lat/lon;
fine for an orchard of a few km). Inverse-distance weighting (power 2, numpy) fills every grid cell
within `max_dist_m` of a trap; cells farther away are unknown and filled with `cv2.inpaint` (Telea), so
the map does not invent hot spots far from any trap but stays continuous; `cv2.GaussianBlur` smooths the
IDW cusps. Hot zones = `cv2.findContours` on the map thresholded at `hot` (absolute pressure, or the
`hot_quantile` of the trap values when not given), polygons simplified with `approxPolyDP` and turned
back into lon/lat (or x/y). The PNG uses `cv2.applyColorMap` (TURBO) with a drawn colorbar and the traps.

`neighbors(traps, trap_id, k)` (or `TrapLayout(traps).neighbors(trap_id, k)`): the k nearest traps by
distance (haversine metres for lat/lon), nearest first.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

EARTH_R = 6_371_008.8  # mean Earth radius, metres
CELL_M = 5.0
MARGIN_M = 30.0
MAX_GRID = 600  # cells on the long side; cell size grows when the orchard is larger
IDW_POWER = 2.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_R * math.asin(math.sqrt(a))


def _is_geo(traps: list[dict]) -> bool:
    return all("lat" in t and "lon" in t and t["lat"] is not None for t in traps)


class TrapLayout:
    """Trap positions in local metres (x east, y north) plus the projection back to lon/lat."""

    def __init__(self, traps: list[dict]):
        if not traps:
            raise ValueError("no traps")
        self.traps = traps
        self.ids = [str(t["id"]) for t in traps]
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("duplicate trap ids")
        self.geo = _is_geo(traps)
        if self.geo:
            self.lat0 = float(np.mean([t["lat"] for t in traps]))
            self.lon0 = float(np.mean([t["lon"] for t in traps]))
            self._k = math.cos(math.radians(self.lat0))
            self.xy = np.array([self.to_xy(t["lat"], t["lon"]) for t in traps], np.float64)
        else:
            self.xy = np.array([[float(t["x"]), float(t["y"])] for t in traps], np.float64)

    def to_xy(self, lat: float, lon: float) -> tuple[float, float]:
        return (math.radians(lon - self.lon0) * EARTH_R * self._k, math.radians(lat - self.lat0) * EARTH_R)

    def to_lonlat(self, x: float, y: float) -> tuple[float, float]:
        return (self.lon0 + math.degrees(x / (EARTH_R * self._k)), self.lat0 + math.degrees(y / EARTH_R))

    def distance_m(self, i: int, j: int) -> float:
        if self.geo:
            a, b = self.traps[i], self.traps[j]
            return haversine_m(a["lat"], a["lon"], b["lat"], b["lon"])
        return float(np.linalg.norm(self.xy[i] - self.xy[j]))

    def neighbors(self, trap_id, k: int = 3) -> list[dict]:
        """The k nearest other traps, nearest first: [{id, distance_m}]."""
        tid = str(trap_id)
        if tid not in self.ids:
            raise KeyError(f"unknown trap {trap_id!r}")
        i = self.ids.index(tid)
        d = [(self.distance_m(i, j), self.ids[j]) for j in range(len(self.ids)) if j != i]
        d.sort()
        return [{"id": t, "distance_m": round(m, 2)} for m, t in d[: max(0, k)]]


def neighbors(traps: list[dict], trap_id, k: int = 3) -> list[dict]:
    """The k nearest traps to `trap_id` by distance (see TrapLayout.neighbors)."""
    return TrapLayout(traps).neighbors(trap_id, k)


def idw(points: np.ndarray, values: np.ndarray, gx: np.ndarray, gy: np.ndarray, power: float = IDW_POWER,
        max_dist: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Inverse-distance weighting on grid coordinates (gx, gy 2-D). Returns (map, known mask)."""
    dx = gx[..., None] - points[:, 0]
    dy = gy[..., None] - points[:, 1]
    d = np.sqrt(dx * dx + dy * dy)
    exact = d < 1e-9
    w = 1.0 / np.maximum(d, 1e-9) ** power
    z = (w * values).sum(axis=-1) / w.sum(axis=-1)
    hit = exact.any(axis=-1)
    if hit.any():
        z[hit] = values[np.argmax(exact[hit], axis=-1)]
    known = np.ones(z.shape, bool) if max_dist is None else d.min(axis=-1) <= max_dist
    return z, known


def _colorbar(height: int, vmin: float, vmax: float, label: str, width: int = 90) -> np.ndarray:
    bar = np.zeros((height, width, 3), np.uint8)
    bar[:] = 255
    top, bot = 20, height - 20
    ramp = np.linspace(255, 0, bot - top).astype(np.uint8).reshape(-1, 1)
    colors = cv2.applyColorMap(np.repeat(ramp, 20, axis=1), cv2.COLORMAP_TURBO)
    bar[top:bot, 8:28] = colors
    cv2.rectangle(bar, (8, top), (27, bot - 1), (0, 0, 0), 1)
    for f in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = round(bot - 1 - f * (bot - top - 1))
        cv2.line(bar, (28, y), (33, y), (0, 0, 0), 1)
        cv2.putText(bar, f"{vmin + f * (vmax - vmin):.1f}", (35, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1)
    cv2.putText(bar, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1)
    return bar


def build_density_map(traps: list[dict], **kw) -> dict:
    """Pressure map from trap values (JSON-serializable dict). See module docstring for the fields."""
    return density_map(traps, **kw)[0]


def density_map(
    traps: list[dict],
    *,
    cell_m: float = CELL_M,
    margin_m: float = MARGIN_M,
    power: float = IDW_POWER,
    max_dist_m: float | None = None,
    hot: float | None = None,
    hot_quantile: float = 0.75,
    blur_m: float | None = None,
    label: str = "flies/trap/day",
    out_dir: str | Path | None = None,
    name: str = "density",
) -> tuple[dict, np.ndarray]:
    """(result dict, float32 pressure map, row 0 = north) from trap values."""
    t0 = time.perf_counter()
    lay = TrapLayout(traps)
    vals = np.array([float(t["value"]) for t in traps], np.float64)
    xy = lay.xy
    lo = xy.min(axis=0) - margin_m
    hi = xy.max(axis=0) + margin_m
    span = hi - lo
    cell = max(cell_m, float(span.max()) / MAX_GRID)
    W, H = math.ceil(span[0] / cell) + 1, math.ceil(span[1] / cell) + 1
    # row 0 = north (image convention): y decreases with the row index
    gx, gy = np.meshgrid(lo[0] + (np.arange(W) + 0.5) * cell, hi[1] - (np.arange(H) + 0.5) * cell)
    if max_dist_m is None:
        if len(xy) > 1:  # twice the median nearest-neighbour spacing, at least 3 cells
            dd = np.linalg.norm(xy[:, None] - xy[None], axis=2)
            np.fill_diagonal(dd, np.inf)
            max_dist_m = max(3 * cell, 2.0 * float(np.median(dd.min(axis=1))))
        else:
            max_dist_m = float(span.max())
    z, known = idw(xy, vals, gx, gy, power, max_dist_m)

    vmin, vmax = float(vals.min()), float(vals.max())
    rng = max(1e-9, vmax - vmin)
    z8 = np.clip((z - vmin) / rng * 255.0, 0, 255).astype(np.uint8)
    unknown = (~known).astype(np.uint8) * 255
    if unknown.any() and known.any():
        z8 = cv2.inpaint(z8, unknown, max(3, round(max_dist_m / cell / 4)), cv2.INPAINT_TELEA)
    zf = vmin + z8.astype(np.float32) / 255.0 * rng
    sigma = (blur_m if blur_m is not None else max(cell, 0.15 * max_dist_m)) / cell
    zf = cv2.GaussianBlur(zf, (0, 0), sigmaX=max(0.5, sigma))

    thr = float(hot) if hot is not None else float(np.quantile(vals, hot_quantile))
    hot_mask = (zf >= thr).astype(np.uint8) * 255
    contours, _ = cv2.findContours(hot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    trap_px = [((p[0] - lo[0]) / cell, (hi[1] - p[1]) / cell) for p in xy]

    def to_out(px: float, py: float) -> list[float]:
        x, y = lo[0] + px * cell, hi[1] - py * cell
        if lay.geo:
            lon, lat = lay.to_lonlat(x, y)
            return [round(lon, 7), round(lat, 7)]
        return [round(x, 2), round(y, 2)]

    features = []
    for zi, c in enumerate(sorted(contours, key=cv2.contourArea, reverse=True)):
        if cv2.contourArea(c) < 1.0:
            continue
        ap = cv2.approxPolyDP(c, 0.75, True).reshape(-1, 2)
        if len(ap) < 3:
            continue
        ring = [to_out(float(px), float(py)) for px, py in ap]
        ring.append(ring[0])
        zm = np.zeros_like(hot_mask)
        cv2.drawContours(zm, [c], -1, 255, -1)
        inside = [lay.ids[i] for i, (px, py) in enumerate(trap_px)
                  if zm[min(H - 1, int(py)), min(W - 1, int(px))] > 0]  # the trap's own grid cell
        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [ring]},
            "properties": {"zone": zi + 1, "area_m2": round(float(np.count_nonzero(zm)) * cell * cell, 1),
                           "peak": round(float(zf[zm > 0].max()), 3), "mean": round(float(zf[zm > 0].mean()), 3),
                           "traps": inside},
        })
    geo = {"type": "FeatureCollection", "features": features,
           "properties": {"crs": "EPSG:4326 (lon, lat)" if lay.geo else "local metres (x east, y north)",
                          "hot_threshold": round(thr, 3), "label": label}}
    res = {
        "grid": {"width": W, "height": H, "cell_m": round(cell, 3),
                 "origin": to_out(0, 0), "crs": geo["properties"]["crs"], "max_dist_m": round(max_dist_m, 2),
                 "known_share": round(float(known.mean()), 4)},
        "value_range": [round(vmin, 3), round(vmax, 3)],
        "hot_threshold": round(thr, 3),
        "hot_zones": geo,
        "n_hot_zones": len(features),
        "traps": [{"id": lay.ids[i], "value": float(vals[i]), "cell": [int(trap_px[i][0]), int(trap_px[i][1])],
                   "map_value": round(float(zf[min(H - 1, int(trap_px[i][1])), min(W - 1, int(trap_px[i][0]))]), 3)}
                  for i in range(len(vals))],
        "time_ms": None,
        "evidence": {},
    }
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        res["evidence"]["png"] = write_png(res, zf, trap_px, lay.ids, vals, vmin, vmax, label, out / f"{name}.png")
        gpath = out / f"{name}.geojson"
        gpath.write_text(json.dumps(geo, indent=1), encoding="utf-8")
        res["evidence"]["geojson"] = gpath.as_posix()
    res["time_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
    return res, zf


def write_png(res: dict, zf: np.ndarray, trap_px, ids, vals, vmin, vmax, label: str, path: Path, long_side: int = 640) -> str:
    H, W = zf.shape
    s = long_side / float(max(H, W))
    z8 = np.clip((zf - vmin) / max(1e-9, vmax - vmin) * 255.0, 0, 255).astype(np.uint8)
    img = cv2.applyColorMap(cv2.resize(z8, (max(1, round(W * s)), max(1, round(H * s))), interpolation=cv2.INTER_CUBIC),
                            cv2.COLORMAP_TURBO)
    hot = (cv2.resize(zf, (img.shape[1], img.shape[0])) >= res["hot_threshold"]).astype(np.uint8) * 255
    cs, _ = cv2.findContours(hot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cs, -1, (255, 255, 255), 2)
    for (px, py), tid, v in zip(trap_px, ids, vals):
        c = (round(px * s), round(py * s))
        cv2.circle(img, c, 5, (0, 0, 0), -1)
        cv2.circle(img, c, 5, (255, 255, 255), 1)
        cv2.putText(img, f"{tid}:{v:g}", (c[0] + 7, c[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 2)
        cv2.putText(img, f"{tid}:{v:g}", (c[0] + 7, c[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
    bar = _colorbar(img.shape[0], vmin, vmax, label)
    canvas = np.hstack([img, bar])
    cv2.putText(canvas, f"hot >= {res['hot_threshold']:g} (white outline), cell {res['grid']['cell_m']:g} m",
                (6, canvas.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), canvas):
        raise OSError(f"could not write {path}")
    return path.as_posix()


def synthetic_layout(rows: int = 4, cols: int = 5, spacing_m: float = 40.0, seed: int = 0,
                     hot_xy: tuple[float, float] = (60.0, 80.0), lat0: float | None = None,
                     lon0: float | None = None) -> list[dict]:
    """A grid of traps with a Gaussian pressure bump at `hot_xy` (metres) plus noise, for tests/demos."""
    rng = np.random.default_rng(seed)
    traps = []
    for r in range(rows):
        for c in range(cols):
            x = c * spacing_m + rng.uniform(-5, 5)
            y = r * spacing_m + rng.uniform(-5, 5)
            d2 = (x - hot_xy[0]) ** 2 + (y - hot_xy[1]) ** 2
            v = round(2.0 + 30.0 * math.exp(-d2 / (2 * 35.0 ** 2)) + rng.uniform(0, 1.5), 1)
            t = {"id": f"T{r}{c}", "value": v}
            if lat0 is not None and lon0 is not None:
                t["lat"] = lat0 + math.degrees(y / EARTH_R)
                t["lon"] = lon0 + math.degrees(x / (EARTH_R * math.cos(math.radians(lat0))))
            else:
                t["x"], t["y"] = x, y
            traps.append(t)
    return traps


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("traps", help="JSON list of traps, or 'synthetic'")
    ap.add_argument("--out", default=None)
    ap.add_argument("--hot", type=float, default=None)
    ap.add_argument("--name", default="density")
    a = ap.parse_args(argv)
    if a.traps == "synthetic":
        traps = synthetic_layout(lat0=11.1284, lon0=106.4824)
    else:
        p = Path(a.traps)
        if not p.is_file():
            print(f"cannot read {a.traps}", file=sys.stderr)
            return 2
        traps = json.loads(p.read_text(encoding="utf-8"))
    print(json.dumps(build_density_map(traps, hot=a.hot, out_dir=a.out, name=a.name), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
