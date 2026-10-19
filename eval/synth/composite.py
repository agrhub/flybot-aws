"""OpenCV 5 compositing: paste flies onto a trap photo, make it look like a capture, label it as synthetic.

    paste_fly       one fly (FlyCrop) at a centre, size and angle; multiply blend under a soft mask (default)
                    or cv2.seamlessClone; returns the fly's box in the photo
    placement_mask  where flies may land: the trap floor found by sentinel.vision.card.find_lure_floor (the same
                    ROI the detector counts in), eroded by a margin, minus the stamp corner
    finish          brightness / contrast jitter, sensor noise, then the visible "SIMULATED" stamp over the
                    camera's own date overlay (inpainted first, so no real date stays on a composite)
    encode_jpeg     JPEG bytes with the synthetic label in EXIF (ImageDescription, Software), XMP and a comment
    read_label      the label back from JPEG bytes (tests, and the check before import)
"""

from __future__ import annotations

import struct

import cv2
import numpy as np
from eval.synth import LABEL
from eval.synth.flies import FlyCrop

OSD_REGION = (0.0, 0.90, 0.30, 1.0)  # x0, y0, x1, y1 fractions: the EZVIZ date overlay sits bottom-left
XMP_NS = "https://flybot.agrhub.com/ns/synthetic/1.0/"


def paste_fly(img: np.ndarray, fly: FlyCrop, center: tuple[float, float], long_side: float, angle: float,
              blur_sigma: float = 0.0, blend: str = "multiply") -> list[float] | None:
    """Paste `fly` into float32 BGR `img` in place. The fly's labelled box gets `long_side` px on its long side.

    Returns the pasted fly's box (xyxy px, axis-aligned bound of the mask at alpha > 0.25) or None when it does not fit.
    """
    s = long_side / max(1e-6, fly.long_side)
    h, w = fly.alpha.shape
    side = int(np.ceil(np.hypot(w, h) * s)) + 6
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, s)
    M[:, 2] += (side / 2.0 - w / 2.0, side / 2.0 - h / 2.0)
    ratio = cv2.warpAffine(fly.ratio, M, (side, side), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=(1.0, 1.0, 1.0))
    alpha = cv2.warpAffine(fly.alpha, M, (side, side), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                           borderValue=0)
    if blur_sigma > 0:  # the fly was shot sharper than a trap camera sees it
        ratio = cv2.GaussianBlur(ratio, (0, 0), blur_sigma)
        alpha = cv2.GaussianBlur(alpha, (0, 0), blur_sigma)
    H, W = img.shape[:2]
    x0, y0 = round(center[0] - side / 2.0), round(center[1] - side / 2.0)
    if x0 < 0 or y0 < 0 or x0 + side > W or y0 + side > H:
        return None
    ys, xs = np.nonzero(alpha > 0.25)
    if len(xs) == 0:
        return None
    roi = img[y0:y0 + side, x0:x0 + side]
    a = alpha[..., None]
    if blend == "seamless":
        local_bg = np.median(roi[alpha < 0.05], axis=0) if (alpha < 0.05).any() else roi.reshape(-1, 3).mean(0)
        src = np.clip(ratio * local_bg, 0, 255).astype(np.uint8)
        # Poisson blend over the whole patch (2 px border = boundary): a mask hugging the fly loses its edges
        mask = np.zeros((side, side), np.uint8)
        mask[2:-2, 2:-2] = 255
        x_lo, y_lo = max(0, x0 - 2), max(0, y0 - 2)
        x_hi, y_hi = min(W, x0 + side + 2), min(H, y0 + side + 2)
        win = np.clip(img[y_lo:y_hi, x_lo:x_hi], 0, 255).astype(np.uint8)  # local window keeps it fast
        out = cv2.seamlessClone(src, win, mask, (x0 - x_lo + side // 2, y0 - y_lo + side // 2), cv2.NORMAL_CLONE)
        img[y_lo:y_hi, x_lo:x_hi] = out.astype(np.float32)
    elif blend == "multiply":
        roi *= (1.0 - a) + a * ratio
    else:
        raise ValueError(f"blend must be multiply or seamless, not {blend!r}")
    return [float(x0 + xs.min()), float(y0 + ys.min()), float(x0 + xs.max() + 1), float(y0 + ys.max() + 1)]


def placement_mask(img: np.ndarray, polygon: list | None, margin: float, frame_margin: float | None = None) -> np.ndarray:
    """uint8 mask of allowed fly centres: inside `polygon` (px) or the whole frame, `margin` px from its edge, and
    `frame_margin` px (default `margin`) from the frame edge, the stamp corner and the logo corner."""
    H, W = img.shape[:2]
    fm = int(margin if frame_margin is None else frame_margin)
    m = np.zeros((H, W), np.uint8)
    if polygon:
        cv2.fillPoly(m, [np.round(np.asarray(polygon, np.float32)).astype(np.int32)], 255)
    else:
        m[:] = 255
    m[0, :], m[-1, :], m[:, 0], m[:, -1] = 0, 0, 0, 0  # the frame edge counts as outside
    m = ((cv2.distanceTransform(m, cv2.DIST_L2, 3) > margin) * 255).astype(np.uint8)
    _x0, y0, x1, _y1 = OSD_REGION
    m[int(y0 * H) - fm:, : int(x1 * W) + fm] = 0
    m[: int(0.08 * H) + fm, : int(0.15 * W) + fm] = 0  # camera logo corner
    m[:fm], m[H - fm:], m[:, :fm], m[:, W - fm:] = 0, 0, 0, 0
    return m


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def erase_osd(img: np.ndarray) -> np.ndarray:
    """Inpaint the camera's bright date overlay (bottom-left) so a composite never shows the background's real date."""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = (int(OSD_REGION[0] * W), int(OSD_REGION[1] * H), int(OSD_REGION[2] * W), int(OSD_REGION[3] * H))
    u8 = np.clip(img, 0, 255).astype(np.uint8)
    region = u8[y0:y1, x0:x1]
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2].astype(np.float32)
    text = ((v > np.percentile(v, 70) + 25) | (hsv[..., 1] < 40) & (v > 200)).astype(np.uint8) * 255
    text = cv2.dilate(text, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    mask = np.zeros(u8.shape[:2], np.uint8)
    mask[y0:y1, x0:x1] = text
    return cv2.inpaint(u8, mask, max(3, H // 150), cv2.INPAINT_TELEA).astype(np.float32)


def stamp(img_u8: np.ndarray, text: str) -> np.ndarray:
    """Draw the visible label (white on a dark box) in the bottom-left corner, where the camera date was."""
    H, W = img_u8.shape[:2]
    scale = max(0.4, W / 1600.0)
    th = max(1, round(scale * 2))
    (tw, tht), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
    x, y = int(0.015 * W), H - int(0.02 * H)
    cv2.rectangle(img_u8, (x - 4, y - tht - 6), (x + tw + 4, y + base + 2), (20, 20, 20), -1)
    cv2.putText(img_u8, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), th, cv2.LINE_AA)
    return img_u8


def finish(img: np.ndarray, rng: np.random.Generator, stamp_text: str | None, gain: float, contrast: float,
           noise_sigma: float) -> np.ndarray:
    """Photo-level jitter (gain, contrast around mid-grey, Gaussian sensor noise) -> uint8, stamped."""
    out = cv2.convertScaleAbs(img, alpha=contrast * gain, beta=(128.0 - 128.0 * contrast) * gain)
    noise = np.empty(out.shape, np.int16)
    cv2.setRNGSeed(int(rng.integers(2**31 - 1)))
    cv2.randn(noise, 0, noise_sigma)  # OpenCV's RNG: much faster than numpy on a 3 Mpx photo
    u8 = cv2.add(out, noise, dtype=cv2.CV_8U)
    return stamp(u8, stamp_text) if stamp_text else u8


# ---------------------------------------------------------------- JPEG label (EXIF, XMP, comment)


def _exif(description: str, software: str) -> bytes:
    """Minimal little-endian EXIF APP1 payload: IFD0 with ImageDescription (0x010E) and Software (0x0131)."""
    entries = [(0x010E, description.encode("ascii", "replace") + b"\0"),
               (0x0131, software.encode("ascii", "replace") + b"\0")]
    n = len(entries)
    data_off = 8 + 2 + n * 12 + 4
    ifd, blob = struct.pack("<H", n), b""
    for tag, val in entries:
        ifd += struct.pack("<HHII", tag, 2, len(val), data_off + len(blob))
        blob += val + (b"\0" if len(val) % 2 else b"")
    ifd += struct.pack("<I", 0)
    return b"Exif\0\0" + b"II*\0" + struct.pack("<I", 8) + ifd + blob


def _xmp(fields: dict) -> bytes:
    attrs = " ".join(f'flybot:{k}="{_xml(str(v))}"' for k, v in fields.items())
    xml = ('<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>'
           '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
           f'<rdf:Description rdf:about="" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:flybot="{XMP_NS}" {attrs}>'
           f'<dc:description><rdf:Alt><rdf:li xml:lang="x-default">{_xml(LABEL)}</rdf:li></rdf:Alt></dc:description>'
           '</rdf:Description></rdf:RDF></x:xmpmeta><?xpacket end="w"?>')
    return b"http://ns.adobe.com/xap/1.0/\0" + xml.encode("utf-8")


def _xml(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _segment(marker: int, payload: bytes) -> bytes:
    if len(payload) + 2 > 0xFFFF:
        raise ValueError("JPEG segment too large")
    return struct.pack(">BBH", 0xFF, marker, len(payload) + 2) + payload


def encode_jpeg(img_u8: np.ndarray, quality: int, fields: dict, software: str) -> bytes:
    """cv2.imencode JPEG with the label inserted right after SOI: APP1 EXIF, APP1 XMP, COM."""
    ok, buf = cv2.imencode(".jpg", img_u8, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    raw = buf.tobytes()
    assert raw[:2] == b"\xff\xd8"
    rest = raw[2:]
    if rest[:2] == b"\xff\xe0":  # drop OpenCV's JFIF APP0 so EXIF is the first segment (EXIF spec order)
        n = struct.unpack(">H", rest[2:4])[0]
        rest = rest[2 + n:]
    meta = (_segment(0xE1, _exif(LABEL, software)) + _segment(0xE1, _xmp(fields))
            + _segment(0xFE, (LABEL + " | " + " ".join(f"{k}={v}" for k, v in fields.items())).encode("utf-8")))
    return b"\xff\xd8" + meta + rest


def read_label(data: bytes) -> dict:
    """Synthetic markers found in JPEG bytes: {"exif": str|None, "xmp_synthetic": bool, "comment": str|None}."""
    out = {"exif": None, "xmp_synthetic": False, "comment": None}
    i = 2
    while i + 4 <= len(data) and data[i] == 0xFF:
        marker = data[i + 1]
        if marker in (0xDA, 0xD9):  # start of scan: metadata is over
            break
        n = struct.unpack(">H", data[i + 2:i + 4])[0]
        payload = data[i + 4:i + 2 + n]
        if marker == 0xE1 and payload.startswith(b"Exif\0\0"):
            tiff = payload[6:]
            off = struct.unpack("<I", tiff[4:8])[0]
            for k in range(struct.unpack("<H", tiff[off:off + 2])[0]):
                tag, _typ, cnt, voff = struct.unpack("<HHII", tiff[off + 2 + 12 * k: off + 14 + 12 * k])
                if tag == 0x010E:
                    out["exif"] = tiff[voff:voff + cnt].rstrip(b"\0").decode("ascii", "replace")
        elif marker == 0xE1 and payload.startswith(b"http://ns.adobe.com/xap/1.0/\0"):
            out["xmp_synthetic"] = b'flybot:synthetic="true"' in payload
        elif marker == 0xFE:
            out["comment"] = payload.decode("utf-8", "replace")
        i += 2 + n
    return out
