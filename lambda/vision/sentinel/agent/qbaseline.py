"""Per-camera rolling op 2 baseline and the "may this photo be counted" rule.

Op 2's relative rules (blur via edge ratio, haze via contrast, dirt via flat share; sentinel/vision/quality.py
`assess(..., baseline=...)`) need the camera's usual scores. The agent keeps them in its state store
(`qbaseline`: the KEYS of its last `baseline_size` accepted photos) and re-judges each new photo against their
median once `baseline_min` photos exist, whatever analyzer produced the scores. Issues are only ever added.
"""

from __future__ import annotations

import statistics

from sentinel.agent.policy import local
from sentinel.common.schemas import Observation, PolicyConfig

KEYS = ("contrast", "edge_ratio", "flat_share", "sharpness")  # quality.Baseline.KEYS
NEEDED = ("brightness", "clip_high", "sharpness", "contrast", "flat_share", "saturation_mean", "glare_ratio", "edge_ratio")
CARD_VISIBILITY = ("no_card", "partial_card")


def baseline_of(rows: list[dict], policy: PolicyConfig) -> dict | None:
    if len(rows) < policy.quality.baseline_min:
        return None
    return {k: float(statistics.median(r[k] for r in rows)) for k in KEYS} | {"size": len(rows), "source": "store"}


def previous_row(ref: str | None, trap_kind: str) -> dict | None:
    """Op 2 KEYS of the camera's previous photo when op 2 finds nothing wrong with it (bootstrap baseline).
    Only real images (local path, file://, s3://, http(s)://); any failure -> None."""
    if not ref or ref.startswith(("fake://", "replay://")):
        return None
    try:
        from sentinel.agent.vision import load_image
        from sentinel.vision.quality import assess_quality

        q = assess_quality(load_image(ref), find=True, trap_kind=trap_kind)
    except Exception:  # noqa: BLE001 - no bootstrap when the previous photo cannot be read
        return None
    if [i for i in q["issues"] if i not in CARD_VISIBILITY]:
        return None
    return {k: float(q["scores"][k]) for k in KEYS}


def rejudge(obs: Observation, rows: list[dict], policy: PolicyConfig) -> list[str]:
    """Apply the camera baseline to `obs` (in place). Returns the issues it added."""
    scores = obs.quality.scores or {}
    if any(scores.get(k) is None for k in NEEDED):
        return []
    base = baseline_of(rows, policy)
    if base is None and policy.quality.baseline_bootstrap_previous:
        prev = previous_row(obs.previous_image, obs.trap_kind)
        if prev is not None:
            base = {**prev, "size": 1, "source": "previous_photo"}
    if base is None:
        return []
    from sentinel.vision.quality import ISSUES, assess

    hour = local(obs.captured_at, policy).hour if obs.captured_at else None
    found = assess(scores, None, base, hour)
    added = [i for i in found if i not in obs.quality.issues]
    obs.quality.baseline = base
    if added:
        merged = set(obs.quality.issues) | set(added)
        obs.quality.issues = [i for i in ISSUES if i in merged] + [i for i in obs.quality.issues if i not in ISSUES]
        obs.quality.ok = False
    return added


def occluded(obs: Observation) -> bool:
    """Op 8 saw a large object in THIS photo (op 8 diffs against the previous photo: `occluder.in_frame` "prev"
    means the object was in the previous one, e.g. the photo before a recapture, and this one is clear)."""
    mo = obs.motion or {}
    return bool(mo.get("occluded_by_large_object")) and (mo.get("occluder") or {}).get("in_frame") != "prev"


def not_countable(obs: Observation, policy: PolicyConfig) -> list[str]:
    """Why the photo must not enter the daily rollup ([] = count it)."""
    q = policy.quality
    why = sorted(set(obs.quality.issues) & set(q.recapture_on + q.postpone_on + q.trap_missing_on))
    if occluded(obs):
        why.append(f"occluded_by_large_object (area {((obs.motion or {}).get('occluder') or {}).get('area_share')})")
    return why


def accepted_row(obs: Observation) -> dict | None:
    """The photo's KEYS when it may join the baseline (no issue except card visibility, not occluded)."""
    scores = obs.quality.scores or {}
    if [i for i in obs.quality.issues if i not in CARD_VISIBILITY] or occluded(obs):
        return None
    if any(not isinstance(scores.get(k), int | float) for k in KEYS):
        return None
    return {k: float(scores[k]) for k in KEYS}
