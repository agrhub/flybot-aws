"""Count curves: a plausible fruit-fly wave per trap and day, then counts per scheduled photo.

The daily level is `baseline + amplitude * wave(day)`, where `wave` is an asymmetric bell: a rise over `rise_days`
to the peak on `peak_day`, a slower decline over `decline_days`. Day-to-day noise is a log-normal factor with AR(1)
correlation (`noise`, `noise_corr`), so bad and good days come in runs as weather does. The curve is not tied to
temperature or degree-days (no weather series is used).

    lure:   the trap is read photo by photo; each photo's count is Poisson(level * slot weight), independent of the
            previous photo (the count resets every photo).
    sticky: `level` is the day's catch; each slot adds Poisson(level * slot weight / slots) new flies that stay on
            the card until the card is changed every `card_days` days (first photo of that day shows a clean card).

Slot weights model daytime activity (fruit flies are most active mid-morning and late afternoon); they average 1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Wave:
    baseline: float = 1.0
    amplitude: float = 20.0
    peak_day: float = 18.0
    rise_days: float = 4.0
    decline_days: float = 6.0
    noise: float = 0.25
    noise_corr: float = 0.5

    def mean(self, day: float) -> float:
        width = self.rise_days if day <= self.peak_day else self.decline_days
        return self.baseline + self.amplitude * math.exp(-0.5 * ((day - self.peak_day) / max(0.1, width)) ** 2)


def daily_levels(wave: Wave, days: int, rng: np.random.Generator) -> list[float]:
    """Expected flies per day (lure: per photo at slot weight 1; sticky: caught per day), with correlated noise."""
    out, e = [], 0.0
    for d in range(days):
        e = wave.noise_corr * e + math.sqrt(1 - wave.noise_corr ** 2) * rng.normal(0.0, wave.noise)
        out.append(round(wave.mean(d) * math.exp(e - wave.noise ** 2 / 2), 3))
    return out


def slot_weights(local_hours: list[float]) -> list[float]:
    """Daytime activity weight per capture slot (two humps, 09:00 and 16:00 local), normalised to mean 1."""
    raw = [0.25 + math.exp(-0.5 * ((h - 9.0) / 1.8) ** 2) + 0.8 * math.exp(-0.5 * ((h - 16.0) / 1.8) ** 2)
           for h in local_hours]
    m = sum(raw) / len(raw)
    return [r / m for r in raw]


def lure_counts(levels: list[float], weights: list[float], rng: np.random.Generator, cap: int) -> list[list[int]]:
    """Per day, per slot: flies visible in a lure trap photo (resets every photo)."""
    return [[min(cap, int(rng.poisson(lv * w))) for w in weights] for lv in levels]


def sticky_new(levels: list[float], weights: list[float], rng: np.random.Generator) -> list[list[int]]:
    """Per day, per slot: new flies stuck since the previous slot."""
    n = len(weights)
    return [[int(rng.poisson(lv * w / n)) for w in weights] for lv in levels]
