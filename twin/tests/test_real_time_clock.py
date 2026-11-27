"""clock.start: now. Tick 0 is the real moment of launch, so timestamps carry today's date.

Why it exists: a partner that checks freshness against its own clock, or signs tokens with the real time, refuses a Twin
whose clock sits on a fixed date in the past. The fixed date stays the default, so every seeded run and test is repeatable.
"""
import asyncio
import contextlib
import logging
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

import twin.clock as clock_module
from twin.actions import ActionBook
from twin.auth import Identity
from twin.clock import resolve_start
from twin.generator import generate
from twin.settings import ClockCfg, ConfigError, load_config
from twin.sim import Simulation
from twin.tokens import make_token
from twin.util import format_rfc3339

FIXED_START = "2026-10-05T08:00:00.000Z"
LAUNCH = datetime(2026, 11, 30, 21, 0, 0, tzinfo=timezone.utc)
BRAIN = Identity.CITY_BRAIN


class FakeNow:
    """A real clock we control. Install it with `monkeypatch.setattr(clock_module, "utc_now", fake)`."""

    def __init__(self, at: datetime):
        self.at = at

    def __call__(self) -> datetime:
        return self.at


def make_sim(start: str, **clock_edits) -> Simulation:
    cfg = load_config()
    cfg.clock.start = start
    for name, value in clock_edits.items():
        setattr(cfg.clock, name, value)
    topo, _ = generate(cfg)
    return Simulation(topo, cfg)


# ------------------------------------------------------------------ the default is untouched
def test_the_default_start_is_still_the_fixed_date():
    sim = make_sim(FIXED_START)
    assert load_config().clock.start == FIXED_START
    assert sim.real_time is False
    assert sim.clock.iso_of(0) == FIXED_START and sim.clock.iso_of(90) == "2026-10-05T08:01:30.000Z"


# ------------------------------------------------------------------ what the config accepts
def test_the_word_now_is_a_valid_start():
    assert ClockCfg(start="now").start == "now"


@pytest.mark.parametrize("bad", ["nowish", "NOW", "Now", "", "yesterday", "2026-10-05T08:00:00", "2026-13-45T99:00:00Z"])
def test_anything_else_is_refused_with_a_message_that_names_the_fix(bad):
    with pytest.raises(ValidationError) as e:
        ClockCfg(start=bad)
    assert "clock.start" in str(e.value) and "now" in str(e.value)


def test_resolve_start_leaves_a_fixed_time_alone_and_turns_now_into_the_real_time(monkeypatch):
    monkeypatch.setattr(clock_module, "utc_now", FakeNow(LAUNCH + timedelta(milliseconds=123, microseconds=900)))
    assert resolve_start(FIXED_START) == FIXED_START
    assert resolve_start("now") == "2026-11-30T21:00:00.123Z"          # milliseconds, a literal Z, like every other timestamp


# ------------------------------------------------------------------ the switch from the environment
def test_the_environment_can_switch_it_on_without_editing_the_yaml(monkeypatch):
    monkeypatch.setenv("TWIN_CLOCK_START", "now")
    assert load_config().clock.start == "now"


def test_an_empty_environment_value_changes_nothing(monkeypatch):
    monkeypatch.setenv("TWIN_CLOCK_START", "  ")
    assert load_config().clock.start == FIXED_START


def test_a_bad_environment_value_is_a_clear_config_error(monkeypatch):
    monkeypatch.setenv("TWIN_CLOCK_START", "soon")
    with pytest.raises(ConfigError) as e:
        load_config()
    assert "clock.start" in str(e.value)


# ------------------------------------------------------------------ what the simulation does with it
def test_tick_zero_is_the_launch_moment_and_time_counts_on_from_there(monkeypatch):
    monkeypatch.setattr(clock_module, "utc_now", FakeNow(LAUNCH))
    sim = make_sim("now")
    assert sim.real_time is True
    assert sim.clock.iso_of(0) == "2026-11-30T21:00:00.000Z"
    assert sim.clock.iso_of(90) == "2026-11-30T21:01:30.000Z"
    stamps = [r.timestamp for r in sim.readings_at(5).readings]
    assert all(s.startswith("2026-11-30T21:00:0") and s.endswith("Z") for s in stamps)      # today's date, not 5 Oct


def test_readings_have_the_same_values_whatever_the_start_date(monkeypatch):
    """The replay promise: same seed and tick give the same numbers. Only the timestamps change."""
    monkeypatch.setattr(clock_module, "utc_now", FakeNow(LAUNCH))
    fixed, live = make_sim(FIXED_START), make_sim("now")
    for tick in (0, 7, 500):
        a, b = fixed.readings_at(tick).readings, live.readings_at(tick).readings
        assert [(r.reading_id, r.device_id, r.value) for r in a] == [(r.reading_id, r.device_id, r.value) for r in b]
    assert fixed.readings_at(7).readings[0].timestamp != live.readings_at(7).readings[0].timestamp


async def test_a_reset_starts_the_new_run_at_the_real_time_of_the_reset(monkeypatch):
    """Without this, a scenario reset after ten minutes would date the new run ten minutes in the past."""
    now = FakeNow(LAUNCH)
    monkeypatch.setattr(clock_module, "utc_now", now)
    sim = make_sim("now")
    old_run = sim.run_id
    await sim.advance(600)
    now.at = LAUNCH + timedelta(seconds=600)
    await sim.reset()
    assert sim.run_id != old_run and sim.tick == 0
    assert sim.clock.iso_of(0) == format_rfc3339(LAUNCH + timedelta(seconds=600))


async def test_a_reset_with_a_fixed_start_still_goes_back_to_the_fixed_date(monkeypatch):
    monkeypatch.setattr(clock_module, "utc_now", FakeNow(LAUNCH))
    sim = make_sim(FIXED_START)
    await sim.advance(600)
    await sim.reset()
    assert sim.clock.iso_of(0) == FIXED_START


async def test_the_first_tick_counts_from_when_ticking_starts_not_from_when_the_app_was_built(monkeypatch):
    now = FakeNow(LAUNCH)
    monkeypatch.setattr(clock_module, "utc_now", now)
    sim = make_sim("now", speed=1000.0)                  # fast ticks so the test is quick (the speed warning is tested below)
    now.at = LAUNCH + timedelta(seconds=30)              # the app took 30 s to come up
    task = asyncio.create_task(sim.run_forever())
    await asyncio.sleep(0)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    assert sim.clock.iso_of(0) == format_rfc3339(LAUNCH + timedelta(seconds=30))


async def test_with_real_speed_simulated_time_stays_close_to_real_time():
    """No fake clock here: let it tick for a couple of real seconds and compare with the wall clock."""
    sim = make_sim("now", speed=1.0)
    task = asyncio.create_task(sim.run_forever())
    await asyncio.sleep(2.3)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    assert sim.tick >= 2
    off = abs((datetime.now(timezone.utc) - sim.clock.time_of(sim.tick)).total_seconds())
    assert off < 1.0


# ------------------------------------------------------------------ warnings
def messages(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == "twin.sim"]


def test_real_speed_gives_no_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="twin.sim"):
        make_sim("now", speed=1.0, autorun=True)
    assert messages(caplog) == []


def test_a_fast_clock_warns_that_time_will_drift(caplog):
    with caplog.at_level(logging.WARNING, logger="twin.sim"):
        make_sim("now", speed=20.0)
    assert len(messages(caplog)) == 1 and "speed" in messages(caplog)[0] and "drift" in messages(caplog)[0]


def test_a_stopped_clock_warns_that_time_will_fall_behind(caplog):
    with caplog.at_level(logging.WARNING, logger="twin.sim"):
        make_sim("now", autorun=False)
    assert len(messages(caplog)) == 1 and "autorun" in messages(caplog)[0]


def test_a_fixed_start_never_warns(caplog):
    with caplog.at_level(logging.WARNING, logger="twin.sim"):
        make_sim(FIXED_START, speed=20.0, autorun=False)
    assert messages(caplog) == []


# ------------------------------------------------------------------ the reason it matters: tokens
def _guardian_token(sim: Simulation, issued_at: datetime) -> str:
    """A token the way a Guardian with its own real-time clock would stamp it."""
    return make_token({"iss": "guardian", "aud": "twin", "jti": f"tk-{uuid.uuid4().hex[:8]}", "run_id": sim.run_id,
                       "iat": format_rfc3339(issued_at), "exp": format_rfc3339(issued_at + timedelta(seconds=30)),
                       "action": "set_valve_position", "targets": ["WAT-01"], "params": {"position": 40}, "score": 0.9})


def _commit(sim: Simulation, token: str):
    book = ActionBook(*_topology_and_config(sim), sim)
    book.preview(BRAIN, "set_valve_position", ["WAT-01"], {"position": 40})
    return book.actuate(BRAIN, "set_valve_position", ["WAT-01"], {"position": 40}, token, f"k-{uuid.uuid4().hex[:6]}")


def _topology_and_config(sim: Simulation):
    cfg = load_config()
    topo, _ = generate(cfg)
    return topo, cfg


def test_a_token_stamped_with_the_real_time_is_accepted_in_real_time_mode(monkeypatch):
    monkeypatch.setattr(clock_module, "utc_now", FakeNow(LAUNCH))
    sim = make_sim("now")
    result = _commit(sim, _guardian_token(sim, LAUNCH))
    assert result.status == "committed", result.code


def test_the_same_token_is_refused_while_the_clock_sits_on_the_fixed_date(monkeypatch):
    """This is the failure a real Guardian would hit with the old clock: its tokens look dated in the future."""
    monkeypatch.setattr(clock_module, "utc_now", FakeNow(LAUNCH))
    sim = make_sim(FIXED_START)
    result = _commit(sim, _guardian_token(sim, LAUNCH))
    assert result.status == "rejected" and result.code == "TOKEN_INVALID"
