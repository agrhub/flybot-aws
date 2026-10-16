"""Op 7: orchard density map (IDW + inpaint + blur, hot-zone contours) and trap neighbours."""

import json
from pathlib import Path

import numpy as np
import pytest

from sentinel.vision.density import (
    TrapLayout,
    build_density_map,
    density_map,
    haversine_m,
    idw,
    neighbors,
    synthetic_layout,
)


def test_idw_exact_at_traps_and_between():
    pts = np.array([[0.0, 0.0], [10.0, 0.0]])
    vals = np.array([0.0, 10.0])
    gx, gy = np.meshgrid(np.array([0.0, 5.0, 10.0]), np.array([0.0]))
    z, known = idw(pts, vals, gx, gy)
    assert z[0, 0] == pytest.approx(0.0) and z[0, 2] == pytest.approx(10.0) and z[0, 1] == pytest.approx(5.0)
    assert known.all()


def test_hot_zone_around_the_hot_traps(tmp_path):
    traps = synthetic_layout(hot_xy=(60.0, 80.0))
    res, zf = density_map(traps, out_dir=tmp_path, name="d")
    json.dumps(res)
    assert res["n_hot_zones"] >= 1
    zone = res["hot_zones"]["features"][0]
    in_zones = {i for f in res["hot_zones"]["features"] for i in f["properties"]["traps"]}
    hot_ids = {t["id"] for t in traps if t["value"] > res["hot_threshold"] * 1.2}
    assert hot_ids and hot_ids <= in_zones  # clearly hot traps lie inside a hot zone (smoothing may trim borderline ones)
    # the hottest trap is inside the first (largest) zone, the coldest corner trap is not
    hottest = max(traps, key=lambda t: t["value"])["id"]
    coldest = min(traps, key=lambda t: t["value"])["id"]
    assert hottest in zone["properties"]["traps"] and coldest not in zone["properties"]["traps"]
    ring = zone["geometry"]["coordinates"][0]
    assert ring[0] == ring[-1] and len(ring) >= 4
    assert Path(res["evidence"]["png"]).is_file() and Path(res["evidence"]["geojson"]).is_file()
    assert zf.shape == (res["grid"]["height"], res["grid"]["width"])


def test_geojson_in_lon_lat_near_the_traps():
    traps = synthetic_layout(lat0=11.1284, lon0=106.4824)
    res = build_density_map(traps)
    assert res["grid"]["crs"].startswith("EPSG:4326")
    lons = [p[0] for f in res["hot_zones"]["features"] for p in f["geometry"]["coordinates"][0]]
    lats = [p[1] for f in res["hot_zones"]["features"] for p in f["geometry"]["coordinates"][0]]
    assert lons and min(lons) > 106.4820 and max(lons) < 106.4850
    assert min(lats) > 11.1280 and max(lats) < 11.1300


def test_uniform_pressure_has_no_partial_hot_zone():
    traps = [{"id": f"t{i}", "x": float(x), "y": float(y), "value": 5.0} for i, (x, y) in
             enumerate([(0, 0), (50, 0), (0, 50), (50, 50)])]
    res = build_density_map(traps, hot=6.0)
    assert res["n_hot_zones"] == 0


def test_inpaint_fills_far_cells():
    traps = [{"id": "a", "x": 0.0, "y": 0.0, "value": 10.0}, {"id": "b", "x": 200.0, "y": 0.0, "value": 2.0}]
    res, zf = density_map(traps, max_dist_m=30.0)
    assert res["grid"]["known_share"] < 0.6  # most cells are far from both traps
    assert np.isfinite(zf).all() and zf.min() >= 2.0 - 1e-3 and zf.max() <= 10.0 + 1e-3


def test_neighbors_by_distance():
    traps = [{"id": "A", "x": 0, "y": 0, "value": 1}, {"id": "B", "x": 10, "y": 0, "value": 1},
             {"id": "C", "x": 0, "y": 30, "value": 1}, {"id": "D", "x": 100, "y": 100, "value": 1}]
    n = neighbors(traps, "A", 2)
    assert [x["id"] for x in n] == ["B", "C"] and n[0]["distance_m"] == pytest.approx(10.0)
    assert len(TrapLayout(traps).neighbors("D", 10)) == 3
    with pytest.raises(KeyError):
        neighbors(traps, "Z", 1)


def test_neighbors_geo_haversine():
    traps = [{"id": "BE4089629", "lat": 11.12836, "lon": 106.4824, "value": 1},
             {"id": "BE4111159", "lat": 11.17034, "lon": 106.3434, "value": 1},
             {"id": "X", "lat": 11.12836, "lon": 106.4834, "value": 1}]
    n = neighbors(traps, "BE4089629", 2)
    assert n[0]["id"] == "X" and n[0]["distance_m"] == pytest.approx(haversine_m(11.12836, 106.4824, 11.12836, 106.4834), abs=0.01)
    assert 108 < n[0]["distance_m"] < 110  # 0.001 deg of longitude at 11.1 deg N
    assert n[1]["id"] == "BE4111159" and 15_000 < n[1]["distance_m"] < 16_500


def test_duplicate_ids_rejected():
    with pytest.raises(ValueError):
        TrapLayout([{"id": 1, "x": 0, "y": 0, "value": 1}, {"id": 1, "x": 1, "y": 1, "value": 1}])
