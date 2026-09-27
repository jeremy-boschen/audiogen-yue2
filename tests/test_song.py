"""A bad manifest must fail before any weights load, not 12 minutes in."""
import json
from pathlib import Path

import pytest

from audiogen import song as song_module


def write_song(tmp_path: Path, steps, **overrides) -> Path:
    (tmp_path / "style.txt").write_text("rock, female vocal")
    (tmp_path / "lyrics.txt").write_text("[verse]\nsome words")
    spec = {"id": "t", "seed": 7, "steps": steps, **overrides}
    (tmp_path / "song.json").write_text(json.dumps(spec))
    return tmp_path


class TestTokenArithmetic:
    def test_max_tokens_is_the_new_budget_not_the_total(self, tmp_path):
        song = song_module.load(write_song(tmp_path, [
            {"id": "a", "seconds": 45.0},
            {"id": "b", "seconds": 200.0, "carry_from": "a", "carry_seconds": 45.0},
        ]))
        grown = song.step("b")
        assert grown.total_tokens == 5000
        assert grown.carry_tokens == 1125
        # The library adds the carry back on, so asking for the total would overshoot.
        assert grown.new_tokens == 3875

    def test_a_step_with_no_carry_budgets_its_whole_length(self, tmp_path):
        song = song_module.load(write_song(tmp_path, [{"id": "a", "seconds": 45.0}]))
        assert song.step("a").new_tokens == song.step("a").total_tokens == 1125

    def test_steps_inherit_the_song_seed_but_may_override(self, tmp_path):
        song = song_module.load(write_song(tmp_path, [
            {"id": "a", "seconds": 10.0},
            {"id": "b", "seconds": 20.0, "carry_from": "a", "carry_seconds": 10.0, "seed": 779},
        ]))
        assert song.step("a").seed == 7
        assert song.step("b").seed == 779


class TestValidation:
    def test_carry_from_a_later_step_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="not an earlier step"):
            song_module.load(write_song(tmp_path, [
                {"id": "a", "seconds": 20.0, "carry_from": "b", "carry_seconds": 5.0},
                {"id": "b", "seconds": 10.0},
            ]))

    def test_carry_from_an_unknown_step_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="not an earlier step"):
            song_module.load(write_song(tmp_path, [
                {"id": "a", "seconds": 20.0, "carry_from": "ghost", "carry_seconds": 5.0}]))

    def test_carrying_more_than_the_source_has_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="only 250 long"):
            song_module.load(write_song(tmp_path, [
                {"id": "a", "seconds": 10.0},
                {"id": "b", "seconds": 30.0, "carry_from": "a", "carry_seconds": 20.0},
            ]))

    def test_carry_from_without_carry_seconds_is_rejected(self, tmp_path):
        # No sentinel for "all of it": it reads fine until the source length changes.
        with pytest.raises(ValueError, match="say how much with carry_seconds or carry_tokens"):
            song_module.load(write_song(tmp_path, [
                {"id": "a", "seconds": 10.0},
                {"id": "b", "seconds": 30.0, "carry_from": "a"},
            ]))

    def test_carry_seconds_without_carry_from_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="says what to carry but not carry_from"):
            song_module.load(write_song(tmp_path, [
                {"id": "a", "seconds": 10.0, "carry_seconds": 5.0}]))

    def test_a_step_that_generates_nothing_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="leaving nothing to generate"):
            song_module.load(write_song(tmp_path, [
                {"id": "a", "seconds": 20.0},
                {"id": "b", "seconds": 20.0, "carry_from": "a", "carry_seconds": 20.0},
            ]))

    def test_duplicate_step_ids_are_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="duplicate step id"):
            song_module.load(write_song(tmp_path, [
                {"id": "a", "seconds": 10.0}, {"id": "a", "seconds": 10.0}]))

    def test_missing_lyrics_is_rejected(self, tmp_path):
        root = write_song(tmp_path, [{"id": "a", "seconds": 10.0}])
        (root / "lyrics.txt").unlink()
        with pytest.raises(FileNotFoundError):
            song_module.load(root)

    def test_no_steps_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="no steps"):
            song_module.load(write_song(tmp_path, []))


class TestScore:
    def test_score_is_optional_and_passed_through_when_present(self, tmp_path):
        root = write_song(tmp_path, [{"id": "a", "seconds": 10.0}])
        assert song_module.load(root).abc is None
        (root / "score.abc").write_text("X:1\nK:C\n")
        assert song_module.load(root).abc.startswith("X:1")


class TestEngineUnits:
    """A convenience that cannot be bypassed is a cage: every derived value has an
    escape hatch naming the engine's own unit."""

    def test_max_tokens_passes_through_untouched(self, tmp_path):
        song = song_module.load(write_song(tmp_path, [
            {"id": "a", "seconds": 45.0},
            {"id": "b", "max_tokens": 3875, "carry_from": "a", "carry_seconds": 45.0},
        ]))
        grown = song.step("b")
        assert grown.new_tokens == 3875          # not recomputed from seconds
        assert grown.total_tokens == 5000        # carry added back on
        assert grown.target_seconds == 200.0

    def test_carry_tokens_passes_through_untouched(self, tmp_path):
        song = song_module.load(write_song(tmp_path, [
            {"id": "a", "seconds": 45.0},
            {"id": "b", "seconds": 200.0, "carry_from": "a", "carry_tokens": 1125},
        ]))
        assert song.step("b").carry_tokens == 1125
        assert song.step("b").new_tokens == 3875

    def test_the_two_spellings_agree(self, tmp_path):
        by_seconds = song_module.load(write_song(tmp_path, [
            {"id": "a", "seconds": 45.0},
            {"id": "b", "seconds": 200.0, "carry_from": "a", "carry_seconds": 45.0}]))
        (tmp_path / "song.json").write_text(json.dumps({"id": "t", "seed": 7, "steps": [
            {"id": "a", "max_tokens": 1125},
            {"id": "b", "max_tokens": 3875, "carry_from": "a", "carry_tokens": 1125}]}))
        by_tokens = song_module.load(tmp_path)
        for step in ("a", "b"):
            assert by_seconds.step(step).new_tokens == by_tokens.step(step).new_tokens
            assert by_seconds.step(step).carry_tokens == by_tokens.step(step).carry_tokens

    def test_both_spellings_are_allowed_when_they_agree(self, tmp_path):
        # A manifest is allowed to spell out what it means.
        song = song_module.load(write_song(tmp_path, [
            {"id": "a", "seconds": 45.0, "max_tokens": 1125},
            {"id": "b", "seconds": 200.0, "max_tokens": 3875, "carry_from": "a",
             "carry_seconds": 45.0, "carry_tokens": 1125},
        ]))
        assert song.step("b").new_tokens == 3875
        assert song.step("b").carry_tokens == 1125

    def test_a_length_that_disagrees_with_itself_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="disagrees with itself.*3875 new tokens.*3000"):
            song_module.load(write_song(tmp_path, [
                {"id": "a", "seconds": 45.0},
                {"id": "b", "seconds": 200.0, "max_tokens": 3000,
                 "carry_from": "a", "carry_seconds": 45.0}]))

    def test_a_carry_that_disagrees_with_itself_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="disagrees with itself.*1125 tokens.*1000"):
            song_module.load(write_song(tmp_path, [
                {"id": "a", "seconds": 45.0},
                {"id": "b", "seconds": 200.0, "carry_from": "a",
                 "carry_seconds": 45.0, "carry_tokens": 1000}]))

    def test_a_mismatch_is_caught_even_when_each_half_is_plausible(self, tmp_path):
        # 200s total with 45s carried is 3875 new, not 5000; writing both catches
        # exactly the mistake the seconds spelling exists to prevent.
        with pytest.raises(ValueError, match="disagrees with itself"):
            song_module.load(write_song(tmp_path, [
                {"id": "a", "seconds": 45.0},
                {"id": "b", "seconds": 200.0, "max_tokens": 5000,
                 "carry_from": "a", "carry_seconds": 45.0}]))

    def test_a_length_must_be_given_somehow(self, tmp_path):
        with pytest.raises(ValueError, match="neither seconds nor max_tokens"):
            song_module.load(write_song(tmp_path, [{"id": "a"}]))

    def test_a_typo_in_a_key_is_rejected_not_ignored(self, tmp_path):
        with pytest.raises(ValueError, match=r"unknown keys \['carry_second'\]"):
            song_module.load(write_song(tmp_path, [
                {"id": "a", "seconds": 45.0},
                {"id": "b", "seconds": 200.0, "carry_from": "a", "carry_second": 45.0}]))
