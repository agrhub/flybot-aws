"""Fetch the model files that are not in git, verify their SHA-256, and put them where the code looks.

    make models                                   # = uv run python models/fetch_models.py
    uv run python models/fetch_models.py --check  # verify what is on disk, no network
    uv run python models/fetch_models.py --only detector --force
    uv run python models/fetch_models.py --with-disk   # also DISK + DISK-LightGlue (not used by the code)

Standard library only (also runs as `python models/fetch_models.py`). Each file is streamed to `<dest>.part`,
hashed while it downloads, and moved into place only when size and SHA-256 match; a file already on disk with the
right hash is not downloaded again. Exit status 1 if any file could not be fetched or verified.

| group     | file                                   | git-ignored destination                     | used by                     |
|-----------|----------------------------------------|---------------------------------------------|-----------------------------|
| detector  | fruitfly_deploy.onnx (YOLOv7-tiny)     | models/yolov7-tiny/                         | op 5, detect_yolov7, eval   |
| features  | aliked-n16rot-top1k-640.onnx           | models/features/ ($SENTINEL_FEATURES_DIR)   | op 3 / op 8 `aliked`        |
| features  | aliked_lightglue.onnx                  | models/features/                            | op 3 / op 8 `aliked`        |
| face      | face_detection_yunet_2023mar.onnx      | vendored in sentinel/vision/models/; fetched to models/features/ only if missing | privacy blur |
| disk      | disk.onnx, disk_lightglue.onnx         | models/features/ (opt-in `--with-disk`)     | nothing yet                 |

Detector sources, tried in order: $SENTINEL_MODELS_BASE_URL (any mirror with the same `<file>` names, for example a
GitHub release `https://github.com/<owner>/<repo>/releases/download/models-v1`), then the public S3 bucket
`sentinel-685527430355-models-public` (anonymous HTTPS GET of the one object; no listing). The other models come
from the upstream URLs that OpenCV's own test data uses (opencv_extra 5.x testdata/dnn/download_models.py) and
opencv_zoo; see docs/DECISIONS.md.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_BUCKET_URL = "https://sentinel-685527430355-models-public.s3.ap-southeast-2.amazonaws.com/models"
LIGHTGLUE_OPENCV = "https://raw.githubusercontent.com/YangGuanyuhan/lightglue_opencv_project/main/model"
LIGHTGLUE_ONNX_REL = "https://github.com/fabio-sim/LightGlue-ONNX/releases/download/v0.1.0"
OPENCV_ZOO_YUNET = ("https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
                    "face_detection_yunet_2023mar.onnx")
CHUNK = 1 << 20


@dataclass
class Model:
    group: str
    file: str
    dest: str  # repo-relative directory
    sha256: str
    size: int
    urls: list[str]
    license: str
    sha1: str | None = None  # the hash opencv_extra publishes, checked too when given
    vendored: str | None = None  # repo-relative path of a copy kept in git; fetched only when it is missing
    optional: bool = False
    extra: dict = field(default_factory=dict)

    def path(self, root: Path) -> Path:
        return root / self.dest / self.file


def _detector_urls(file: str, sub: str) -> list[str]:
    urls = []
    if base := os.environ.get("SENTINEL_MODELS_BASE_URL"):
        urls.append(f"{base.rstrip('/')}/{file}")
    urls.append(f"{PUBLIC_BUCKET_URL}/{sub}/{file}")
    return urls


def catalog() -> list[Model]:
    feats = os.environ.get("SENTINEL_FEATURES_DIR")
    feat_dir = Path(feats).as_posix() if feats else "models/features"
    return [
        Model("detector", "fruitfly_deploy.onnx", "models/yolov7-tiny",
              "0cf45768bafd58180c91007270815a8634598cedb280f60f274bf0002b31a16f", 24_146_362,
              _detector_urls("fruitfly_deploy.onnx", "yolov7-tiny"),
              "GPL-3.0 (fine-tuned from WongKinYiu/yolov7 yolov7-tiny.pt; data camos/fruit-fly-wmcqi CC BY 4.0)"),
        Model("features", "aliked-n16rot-top1k-640.onnx", feat_dir,
              "b46868de0794ea0f2d878985d99dc17d3639dfab85d852994254a4cdf0cdc0ea", 6_438_045,
              [f"{LIGHTGLUE_OPENCV}/aliked-n16rot-top1k-640.onnx"],
              "ALIKED (upstream BSD-3-Clause), ONNX export used by OpenCV's tests",
              sha1="41faa7bf5d7eb68a2851471ba03aa20c9db30e4c"),
        Model("features", "aliked_lightglue.onnx", feat_dir,
              "a1bd99d294c4c6c08ca978af3128c85f8c0e603b04a03e4391428455d9624439", 45_837_864,
              [f"{LIGHTGLUE_OPENCV}/aliked_lightglue.onnx"],
              "LightGlue (Apache-2.0) ONNX export used by OpenCV's tests",
              sha1="02723aa521990e57fe33d90b67977590c460e351"),
        Model("face", "face_detection_yunet_2023mar.onnx", feat_dir,
              "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4", 232_589,
              [OPENCV_ZOO_YUNET],
              "MIT (opencv_zoo YuNet)", vendored="sentinel/vision/models/face_detection_yunet_2023mar.onnx"),
        Model("disk", "disk.onnx", feat_dir,
              "b204e36d77538dfd938645e88bad03fb2790e3baef43fb3cfbc34da0c98150ad", 4_404_734,
              [f"{LIGHTGLUE_ONNX_REL}/disk.onnx"],
              "see fabio-sim/LightGlue-ONNX v0.1.0 (DISK: cvlab-epfl/disk)", sha1="5f6a9069aed0af7302b67dcfb6d24b0d46707aec",
              optional=True),
        Model("disk", "disk_lightglue.onnx", feat_dir,
              "025c36dddb245d70cf80674b75eebbbac7e20479a82b3ef3897c36b36d095406", 46_596_282,
              [f"{LIGHTGLUE_ONNX_REL}/disk_lightglue.onnx"],
              "see fabio-sim/LightGlue-ONNX v0.1.0 (LightGlue: Apache-2.0)",
              sha1="9c885c13df94a0cadd5fb577c278471b78cc7058", optional=True),
    ]


def hashes(path: Path) -> tuple[str, str, int]:
    h256, h1, n = hashlib.sha256(), hashlib.sha1(), 0  # sha1 only to match opencv_extra's list
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h256.update(chunk)
            h1.update(chunk)
            n += len(chunk)
    return h256.hexdigest(), h1.hexdigest(), n


def verified(m: Model, path: Path) -> tuple[bool, str]:
    if not path.is_file():
        return False, "missing"
    s256, s1, n = hashes(path)
    if n != m.size:
        return False, f"size {n} != {m.size}"
    if s256 != m.sha256:
        return False, f"sha256 {s256[:16]}... != {m.sha256[:16]}..."
    if m.sha1 and s1 != m.sha1:
        return False, f"sha1 {s1[:16]}... != {m.sha1[:16]}..."
    return True, "ok"


def download(m: Model, dest: Path, urlopen=urllib.request.urlopen, retries: int = 2) -> str:
    """Download `m` to `dest` through `<dest>.part`; returns the URL used. Raises RuntimeError on failure."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    errors = []
    for url in m.urls:
        if not url.startswith("https://"):
            errors.append(f"{url}: not https")
            continue
        for attempt in range(retries + 1):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "flybot-sentinel-fetch-models/1"})
                h256, h1, n = hashlib.sha256(), hashlib.sha1(), 0
                with urlopen(req, timeout=60) as resp, open(part, "wb") as out:
                    while chunk := resp.read(CHUNK):
                        out.write(chunk)
                        h256.update(chunk)
                        h1.update(chunk)
                        n += len(chunk)
                        if n > m.size:
                            raise ValueError(f"larger than the expected {m.size} bytes")
                if n != m.size:
                    raise ValueError(f"got {n} bytes, expected {m.size}")
                if h256.hexdigest() != m.sha256:
                    raise ValueError(f"sha256 {h256.hexdigest()} != expected {m.sha256}")
                if m.sha1 and h1.hexdigest() != m.sha1:
                    raise ValueError(f"sha1 {h1.hexdigest()} != expected {m.sha1}")
                os.replace(part, dest)
                return url
            except (urllib.error.URLError, OSError, ValueError) as e:
                part.unlink(missing_ok=True)
                errors.append(f"{url}: {type(e).__name__}: {e}")
                if isinstance(e, ValueError):  # wrong content: retrying the same URL will not help
                    break
                if attempt < retries:
                    time.sleep(1.5 * (attempt + 1))
    raise RuntimeError("; ".join(errors))


def fetch_one(m: Model, root: Path, *, force: bool = False, check: bool = False, urlopen=urllib.request.urlopen):
    """(status, path, source) for one model. status: ok | vendored | fetched | FAILED: ..."""
    if m.vendored and not force:
        v = root / m.vendored
        ok, why = verified(m, v)
        if ok:
            return "vendored", v, "git"
    dest = m.path(root)
    ok, why = verified(m, dest)
    if ok and not force:
        return "ok", dest, "on disk"
    if check:
        return f"FAILED: {why}", dest, "-"
    try:
        url = download(m, dest, urlopen=urlopen)
    except RuntimeError as e:
        return f"FAILED: {e}", dest, "-"
    return "fetched", dest, url


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", default=None, help="comma list of groups: detector,features,face,disk")
    ap.add_argument("--with-disk", action="store_true", help="also fetch DISK + DISK-LightGlue (unused by the code)")
    ap.add_argument("--force", action="store_true", help="download again even when the file on disk is valid")
    ap.add_argument("--check", action="store_true", help="verify files on disk only, no network")
    ap.add_argument("--root", default=str(ROOT), help="repository root (default: this checkout)")
    a = ap.parse_args(argv)
    root = Path(a.root).resolve()
    groups = set(a.only.split(",")) if a.only else None
    models = [m for m in catalog()
              if (groups is None and (not m.optional or a.with_disk)) or (groups is not None and m.group in groups)]
    failed = 0
    for m in models:
        t0 = time.perf_counter()
        status, path, src = fetch_one(m, root, force=a.force, check=a.check)
        failed += status.startswith("FAILED")
        try:
            shown = path.relative_to(root).as_posix()
        except ValueError:
            shown = path.as_posix()
        print(f"{status.split(':')[0]:9} {m.group:9} {shown}  {m.size:,} B  sha256 {m.sha256}  "
              f"from {src}  ({time.perf_counter() - t0:.1f} s)")
        if status.startswith("FAILED"):
            print(f"          {status}", file=sys.stderr)
    print(f"{len(models) - failed}/{len(models)} model files verified under {root.as_posix()}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
