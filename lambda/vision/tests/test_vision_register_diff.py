import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from sentinel.vision import synth
from sentinel.vision.blobs import detect_dark_blobs
from sentinel.vision.diff import compare_photos, diff_detections, match_boxes, transform_boxes
from sentinel.vision.register import extract, models_available, register, register_features

FIX = Path(__file__).parent / "fixtures"


def _card_pair(seed=0, n_new=5):
    rng = np.random.default_rng(seed)
    c1, boxes = synth.make_card(rng=rng, n_flies=20)
    c2 = c1.copy()
    new = []
    r2 = np.random.default_rng(seed + 100)
    for _ in range(n_new):
        cx, cy = r2.uniform(20, 380), r2.uniform(20, 230)
        cv2.ellipse(c2, (int(cx), int(cy)), (7, 4), 30, 0, 360, (25, 40, 60), -1)
        new.append([cx - 9, cy - 9, cx + 9, cy + 9])
    return c1, c2, boxes, new


def test_sift_recovers_known_shift(tmp_path):
    c1, _, _, _ = _card_pair()
    big = cv2.resize(c1, (800, 500))
    M = np.float32([[1, 0, 12], [0, 1, -7]])
    moved = cv2.warpAffine(big, M, (800, 500), borderMode=cv2.BORDER_REPLICATE)
    r = register(big, moved, method="sift_flann", out_dir=tmp_path, name="s")
    assert r["ok"] and r["method"] == "sift_flann" and r["inliers"] >= 80
    H = np.asarray(r["H"])
    assert abs(H[0, 2] - 12) < 1.5 and abs(H[1, 2] + 7) < 1.5
    assert Path(r["evidence"]["aligned_pair"]).is_file() and Path(r["evidence"]["matches"]).is_file()
    json.dumps(r)


def test_registration_fails_on_unrelated_images():
    a = synth.noise_background(640, 400, np.random.default_rng(1))
    b = np.full((400, 640, 3), 128, np.uint8)
    r = register(a, b, method="sift_flann")
    assert not r["ok"]


def test_real_consecutive_frames_register():
    a = cv2.imread(str(FIX / "vision_be4089629_prev.jpg"))
    b = cv2.imread(str(FIX / "vision_be4089629_cur.jpg"))
    r = register(a, b, method="sift_flann")
    assert r["ok"] and r["inliers"] >= 15


@pytest.mark.skipif(not (models_available()["aliked"] and models_available()["lightglue"]),
                    reason="ALIKED/LightGlue ONNX not in models/features/")
def test_aliked_lightglue_registers():
    c1, _, _, _ = _card_pair(2)
    big = cv2.resize(c1, (640, 400))
    moved = cv2.warpAffine(big, np.float32([[1, 0, 8], [0, 1, 5]]), (640, 400), borderMode=cv2.BORDER_REPLICATE)
    fa, fb = extract(big, "aliked_lightglue", 128), extract(moved, "aliked_lightglue", 128)
    r = register_features(fa, fb)
    assert r["ok"], r
    assert abs(r["H"][0, 2] - 8) < 2 and abs(r["H"][1, 2] - 5) < 2


def test_match_boxes_hungarian():
    prev = [[0, 0, 10, 10], [50, 50, 60, 60]]
    cur = [[1, 1, 11, 11], [100, 100, 110, 110], [51, 49, 61, 59]]
    pairs, gone, fresh = match_boxes(prev, cur)
    assert {(i, j) for i, j, _ in pairs} == {(0, 0), (1, 2)}
    assert gone == [] and fresh == [1]


def test_transform_boxes_translation():
    H = np.array([[1, 0, 5], [0, 1, -3], [0, 0, 1]], float)
    assert transform_boxes([[0, 0, 10, 10]], H) == [[5.0, -3.0, 15.0, 7.0]]


def test_diff_counts_new_and_card_change(tmp_path):
    prev = [[i * 20, 0, i * 20 + 10, 10] for i in range(10)]
    cur = prev[:8] + [[300, 300, 310, 310], [320, 300, 330, 310]]
    d = diff_detections(prev, cur, np.eye(3), out_dir=tmp_path)
    assert (d["new_count"], d["persisted_count"], d["disappeared_count"]) == (2, 8, 2)
    assert not d["card_change_suspected"]
    swapped = diff_detections(prev, [[500, 500, 510, 510]], np.eye(3))
    assert swapped["card_change_suspected"] and "disappeared>>persisted" in swapped["card_change_reasons"]


def test_compare_photos_synthetic_new_flies(tmp_path):
    c1, c2, _, _new = _card_pair(3, n_new=5)
    rng = np.random.default_rng(3)
    bg = synth.noise_background(1280, 720, rng)
    q1 = synth.random_quad(1280, 720, rng, coverage=(0.25, 0.35), max_tilt=0.08)
    q2 = q1 + np.float32([[10, -5], [12, -4], [9, 4], [8, 3]])
    r = compare_photos(synth.paste_card(bg, c1, q1), synth.paste_card(bg, c2, q2), method="sift_flann",
                       out_dir=tmp_path, name="p")
    assert r["frame_source"] == ["card", "card"] and r["register"]["ok"]
    assert r["diff"]["new_count"] == 5
    assert r["diff"]["disappeared_count"] <= 1
    assert not r["diff"]["card_change_suspected"]
    assert Path(r["evidence"]["heatmap"]).is_file()
    json.dumps(r)


def test_blobs_find_dark_spots():
    c1, _, boxes, _ = _card_pair(4)
    r = detect_dark_blobs(c1)
    assert abs(r["n"] - len(boxes)) <= 3
