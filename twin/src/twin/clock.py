"""The simulated clock.

Time in the city is a whole number (the tick) plus a start time. SimClock never reads the real clock:
time_of(tick) is a plain calculation, so the same tick always gives the same simulated time.
How fast ticks happen in real life is the simulation's job (sim.py), not the clock's.

The start time comes from `clock.start` in the config. Two forms are allowed:
  * a fixed UTC time, for example 2026-10-05T08:00:00.000Z. This is the default: every run and every test repeats exactly.
  * the word `now`: the start is the real time at launch, so readings carry today's date. A partner that checks
    freshness against its own clock, or signs tokens with the real time, only works with this form.
resolve_start() below is the one place that asks the real clock.
"""
from datetime import datetime, timedelta, timezone

from .util import format_rfc3339

START_NOW = "now"


def utc_now() -> datetime:
    """The real time. A function of its own so tests can replace it."""
    return datetime.now(timezone.utc)


def resolve_start(value: str) -> str:
    """Turn the configured `clock.start` into a fixed RFC 3339 time. A fixed time comes back unchanged, `now` becomes the real time."""
    return format_rfc3339(utc_now()) if value == START_NOW else value


class SimClock:
    def __init__(self, start: str, tick_seconds: float):
        self.start = datetime.fromisoformat(start)          # python 3.12 reads the trailing Z
        self.tick_seconds = tick_seconds

    def time_of(self, tick: int) -> datetime:
        return self.start + timedelta(seconds=tick * self.tick_seconds)

    def iso_of(self, tick: int) -> str:
        return format_rfc3339(self.time_of(tick))
