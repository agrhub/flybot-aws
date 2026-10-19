"""Synthetic trap photos for the clearly labelled FlyBot demo: real empty trap photos with fruit flies composited in.

Every image this package writes is a composite, never a field photo: the filename starts with `SYNTH_`, the JPEG
carries the label in EXIF, XMP and a comment, a visible "SIMULATED" stamp replaces the camera's date overlay, and
every manifest row has `synthetic: true`. See eval/synth/README.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repository root (eval/synth/__init__.py)
VISION = ROOT / "lambda" / "vision"
if str(VISION) not in sys.path:  # the `sentinel` package (OpenCV 5 vision ops) lives in lambda/vision
    sys.path.insert(0, str(VISION))

LABEL = "SIMULATED DATA: synthetic composite (real empty trap photo + pasted fruit flies), not a field photo"
FILE_PREFIX = "SYNTH_"
SCHEMA = "flybot-synth-manifest/1"
DATASET_CREDIT = ('"fruit-fly" dataset by CamOS on Roboflow Universe, '
                  "https://universe.roboflow.com/camos/fruit-fly-wmcqi/dataset/3, licensed CC BY 4.0")
BACKGROUND_CREDIT = "Empty lure-trap photos from AGRHUB's orchard cameras (FlyBot), data/samples in this repository"
