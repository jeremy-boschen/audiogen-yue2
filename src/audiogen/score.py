"""Read a YuE2 score as music, and set its vocal melody against the lyrics.

YuE2 writes one shape of ABC: a header (M, L, Q, two V declarations, K), then a
body of `% section` comments and alternating `V: Vocal` / `V: Ins` lines, with
chord symbols on the vocal line and occasional inline meter changes. This reads
that shape. It is tolerant of truncation on purpose: a plan prefix can end in
the middle of a note, and the partial text is still worth measuring.

Nothing here judges. Counts, ratios and bins are observations; the bins in
`BINS` are for grouping phrases in a report, not a rule about what sings well.
Syllables are counted by a vowel-group heuristic and are approximate -- "fire"
counts one, "every" three -- which is why every phrase report shows the text.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from fractions import Fraction

BINS = ((0.0, 0.8, "<0.8"), (0.8, 1.0, "0.8-1.0"), (1.0, 1.3, "1.0-1.3"),
        (1.3, 1.6, "1.3-1.6"), (1.6, 2.0, "1.6-2.0"), (2.0, float("inf"), ">2.0"))

# One musical element on a body line. Order matters: chord symbols and inline
# fields first, then rests, then notes.
ELEMENT = re.compile(r"""
    (?P<chord>"[^"]*")
  | (?P<field>\[[A-Za-z]:[^\]]*\])
  | (?P<bar>:?\|[\]|:]?)
  | (?P<mrest>Z(?P<mcount>\d*))
  | (?P<rest>[zx](?P<rlen>\d*(?:/\d*)?))
  | (?P<note>(?P<acc>\^{1,2}|_{1,2}|=)?(?P<pitch>[A-Ga-g])(?P<octave>[,']*)(?P<nlen>\d*(?:/\d*)?)(?P<tie>-?))
""", re.VERBOSE)

PITCH_CLASS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def length(text: str) -> Fraction:
    """ABC note length multiplier: '' -> 1, '2' -> 2, '/2' and '/' -> 1/2, '3/2'."""
    if not text:
        return Fraction(1)
    if "/" not in text:
        return Fraction(int(text))
    top, _, bottom = text.partition("/")
    return Fraction(int(top or 1), int(bottom or 2))


def midi(pitch: str, accidental: str, octave: str) -> int:
    base = 60 + PITCH_CLASS[pitch.upper()] + (12 if pitch.islower() else 0)
    base += accidental.count("^") - accidental.count("_")
    return base + 12 * octave.count("'") - 12 * octave.count(",")


@dataclass
class Event:
    kind: str                 # "note" or "rest"
    units: Fraction           # in L: units
    section: int              # index into Score.sections
    bar: int                  # bar number within the voice, from 0
    pitch: int | None = None
    tied: bool = False        # this note ties into the next
    onset: bool = True        # False when a tie carries it from the previous note
    chord: str | None = None
    line: int = 0             # which body source line wrote it


@dataclass
class Score:
    headers: dict = field(default_factory=dict)
    sections: list = field(default_factory=list)          # labels in order
    voices: dict = field(default_factory=dict)            # name -> [Event]
    bars: dict = field(default_factory=dict)              # name -> bars completed
    chords: list = field(default_factory=list)            # chord symbols in vocal order
    meters: list = field(default_factory=list)            # every M: seen, header first

    @property
    def unit(self) -> Fraction:
        return parse_fraction(self.headers.get("L", "1/8"))

    def seconds_per_unit(self) -> float | None:
        """Wall time of one L: unit, from Q: ('1/4=105'). None when Q is absent."""
        tempo = self.headers.get("Q")
        if not tempo or "=" not in tempo:
            return None
        beat, _, bpm = tempo.partition("=")
        try:
            return float(self.unit / parse_fraction(beat)) * 60.0 / float(bpm)
        except (ValueError, ZeroDivisionError):
            return None


def parse_fraction(text: str) -> Fraction:
    text = text.strip()
    return Fraction(text) if text else Fraction(1)


def parse(text: str) -> Score:
    """Parse as much of `text` as forms whole elements. Never raises on truncation."""
    score = Score()
    in_body = False
    voice = None
    carried = {}               # voice -> an open tie waiting for its next note
    source = 0
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("%"):
            if in_body:
                score.sections.append(line.lstrip("%").strip().lower())
            continue
        header = re.match(r"^([A-Za-z]):\s*(.*)$", line)
        if header:
            key, value = header.group(1), header.group(2).strip()
            if key == "V":
                name = value.split()[0] if value.split() else ""
                if in_body:
                    voice = name
                    score.voices.setdefault(voice, [])
                    score.bars.setdefault(voice, 0)
                continue
            if key == "M":
                score.meters.append(value)
            if not in_body:
                score.headers[key] = value
                if key == "K":
                    in_body = True
            continue
        if not in_body or voice is None:
            continue
        section = max(0, len(score.sections) - 1)
        source += 1
        chord = None
        for element in ELEMENT.finditer(line):
            events = score.voices[voice]
            if element.group("chord"):
                chord = element.group("chord").strip('"')
                if voice.lower().startswith("vocal"):
                    score.chords.append(chord)
            elif element.group("field"):
                if element.group("field")[1] == "M":
                    score.meters.append(element.group("field")[3:-1].strip())
            elif element.group("bar"):
                score.bars[voice] += 1
            elif element.group("mrest"):
                count = int(element.group("mcount") or 1)
                events.append(Event("rest", bar_units(score) * count, section, score.bars[voice], line=source))
                score.bars[voice] += count - 1   # the bar line after Zn closes the last
            elif element.group("rest"):
                events.append(Event("rest", length(element.group("rlen")), section, score.bars[voice],
                                    chord=chord, line=source))
                carried.pop(voice, None)
            elif element.group("note"):
                pitch = midi(element.group("pitch"), element.group("acc") or "", element.group("octave"))
                open_tie = carried.pop(voice, None)
                tied = bool(element.group("tie"))
                events.append(Event("note", length(element.group("nlen")), section, score.bars[voice],
                                    pitch=pitch, tied=tied,
                                    onset=not (open_tie is not None and open_tie == pitch), chord=chord,
                                    line=source))
                if tied:
                    carried[voice] = pitch
    return score


def bar_units(score: Score) -> Fraction:
    meter = score.meters[-1] if score.meters else score.headers.get("M", "4/4")
    if meter in ("C", "C|"):
        meter = "4/4" if meter == "C" else "2/2"
    try:
        return parse_fraction(meter) / score.unit
    except (ValueError, ZeroDivisionError):
        return Fraction(16)


# --- how a plan develops -------------------------------------------------------

def vocal_voice(score: Score) -> str | None:
    return next((name for name in score.voices if name.lower().startswith("vocal")), None)


def describe(score: Score) -> dict:
    """The properties whose arrival the plan microscope tracks, for one text."""
    vocal = vocal_voice(score)
    notes = [e for e in score.voices.get(vocal, []) if e.kind == "note"] if vocal else []
    onsets = [e for e in notes if e.onset]
    spu = score.seconds_per_unit()
    vocal_units = sum((e.units for e in score.voices.get(vocal, [])), Fraction(0)) if vocal else Fraction(0)
    return {
        "tempo": score.headers.get("Q"),
        "meter": score.headers.get("M"),
        "unit": score.headers.get("L"),
        "key": score.headers.get("K"),
        "voices": sorted(score.voices),
        "sections": list(score.sections),
        "section_count": len(score.sections),
        "vocal_bars": score.bars.get(vocal, 0) if vocal else 0,
        "chords": len(score.chords),
        "distinct_chords": sorted(set(score.chords)),
        "vocal_notes": len(onsets),
        "vocal_contour": [e.pitch for e in onsets],
        "vocal_seconds": round(float(vocal_units) * spu, 2) if spu else None,
        "meter_changes": list(score.meters[1:]),
    }


def development(prefixes: dict[int, str]) -> dict:
    """When each property of the final plan first appears and becomes final.

    `prefixes` maps a token count to the raw decoded prefix at that count; the
    largest count is the final plan. Because generation is autoregressive,
    text once written is never revised, so a scalar header is final as soon as
    it is complete, and a growing property (bars, notes, chords) can only
    extend. `first_observable` is the first checkpoint where a property has a
    value at all; `final_from` is the first checkpoint from which it equals the
    final value; for growing properties, `share` gives the fraction of the
    final amount present at each checkpoint so "mostly established" can be read
    off rather than asserted.
    """
    marks = sorted(prefixes)
    described = {mark: describe(parse(prefixes[mark])) for mark in marks}
    final = described[marks[-1]]
    report = {}
    for key, value in final.items():
        observable = next((m for m in marks if described[m][key] not in (None, [], 0, "")), None)
        final_from = next((m for i, m in enumerate(marks)
                           if all(described[n][key] == value for n in marks[i:])), None)
        entry = {"final": value if not isinstance(value, list) or len(value) <= 24 else f"{len(value)} items",
                 "first_observable": observable, "final_from": final_from}
        if isinstance(value, (int, list)) and not isinstance(value, bool):
            size = value if isinstance(value, int) else len(value)
            if size:
                entry["share"] = {m: round((described[m][key] if isinstance(described[m][key], int)
                                            else len(described[m][key])) / size, 3) for m in marks}
        report[key] = entry
    return {"checkpoints": marks, "properties": report,
            "per_checkpoint": {m: {k: v for k, v in described[m].items() if k != "vocal_contour"}
                               for m in marks}}


def valid_prefix(text: str) -> str:
    """The prefix cut back to its last complete line. Derived, never authoritative."""
    return text[:text.rfind("\n") + 1] if "\n" in text else ""


# --- melody against lyrics -----------------------------------------------------

def syllables(word: str) -> int:
    """Approximate English syllables by vowel groups, with a silent-e rule."""
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    n = len(re.findall(r"[aeiouy]+", w))
    if n > 1 and w.endswith("e") and not w.endswith(("le", "ee", "ye")):
        n -= 1
    if n > 1 and w.endswith("ed") and not w.endswith(("ted", "ded")):
        n -= 1
    return max(1, n)


def lyric_sections(text: str) -> list[dict]:
    """[Label] blocks, each with its sung lines. Text outside a label is 'untitled'."""
    sections, current = [], None
    for raw in text.split("\n"):
        line = raw.strip()
        label = re.match(r"^\[(.+?)\]$", line)
        if label:
            current = {"label": label.group(1).strip().lower(), "lines": []}
            sections.append(current)
        elif line:
            if current is None:
                current = {"label": "untitled", "lines": []}
                sections.append(current)
            current["lines"].append(line)
    return sections


def phrases(events: list[Event], gap: Fraction) -> list[list[Event]]:
    """Split a voice into sung phrases at rests of at least `gap` units."""
    out, current, resting = [], [], Fraction(0)
    for event in events:
        if event.kind == "rest":
            resting += event.units
            if resting >= gap and current:
                out.append(current)
                current = []
            elif current:
                current.append(event)
            continue
        resting = Fraction(0)
        current.append(event)
    if current:
        out.append(current)
    # A phrase ends on its last note, not on the short rests before a split.
    return [p[:max(i for i, e in enumerate(p) if e.kind == "note") + 1] for p in out]


def by_line(events: list[Event]) -> list[list[Event]]:
    """Group a voice by the score line that wrote it, trimmed to first..last note.

    YuE2 tends to write one lyric line per `V: Vocal` line, so this segmentation
    often pairs where rest gaps do not (pickups and breaths split a line).
    """
    groups: dict[int, list[Event]] = {}
    for event in events:
        groups.setdefault(event.line, []).append(event)
    out = []
    for group in groups.values():
        notes = [i for i, e in enumerate(group) if e.kind == "note"]
        if notes:
            out.append(group[notes[0]:notes[-1] + 1])
    return out


def bin_of(ratio: float | None) -> str | None:
    if ratio is None:
        return None
    return next(label for low, high, label in BINS if low <= ratio < high)


def measure(line: str | None, notes: list[Event], score: Score) -> dict:
    words = re.findall(r"[A-Za-z']+(?:\.[A-Za-z]\.?)?", line or "")
    count = sum(syllables(w) for w in words)
    onsets = [e for e in notes if e.kind == "note" and e.onset]
    units = sum((e.units for e in notes), Fraction(0))
    bars = len({e.bar for e in notes}) if notes else 0
    spu = score.seconds_per_unit()
    ratio = round(len(onsets) / count, 3) if count else None
    return {
        "lyrics": line, "syllables": count, "words": len(words),
        "melody_notes": len(onsets),
        "notes_per_syllable": ratio,
        "notes_per_word": round(len(onsets) / len(words), 3) if words else None,
        "extra_notes_per_syllable": round(max(0, len(onsets) - count) / count, 3) if count else None,
        "bars": bars,
        "syllables_per_bar": round(count / bars, 3) if bars else None,
        "internal_rests": sum(1 for e in notes if e.kind == "rest"),
        "seconds": round(float(units) * spu, 2) if spu else None,
        "bin": bin_of(ratio),
    }


def alignment(score_text: str, lyrics_text: str, gap_beats: float = 1.0) -> dict:
    """Notes per syllable for the song, each section, and each phrase where it can be paired.

    Score sections are matched to lyric sections by label, in order; a score
    section repeated more often than the lyrics (a third chorus) reuses the
    last lyric section with that label and is marked `repeated`. Phrases are
    paired with lyric lines only when a section has exactly as many of each;
    otherwise the section is reported whole and both counts are shown, rather
    than guessing which line a phrase carries.
    """
    score = parse(score_text)
    vocal = vocal_voice(score)
    lyrics = lyric_sections(lyrics_text)
    if vocal is None:
        return {"error": "score has no vocal voice"}
    spu = score.seconds_per_unit()
    beat = parse_fraction(score.headers.get("Q", "1/4=0").partition("=")[0] or "1/4") / score.unit
    gap = beat * Fraction(gap_beats).limit_denominator(64)
    events = score.voices[vocal]

    starts, clock = {}, Fraction(0)
    for event in events:
        starts.setdefault(id(event), clock)
        clock += event.units

    used, sections = {}, []
    for index, label in enumerate(score.sections):
        body = [e for e in events if e.section == index]
        sung = [e for e in body if e.kind == "note"]
        if not sung:
            sections.append({"index": index, "label": label, "sung": False})
            continue
        candidates = [s for s in lyrics if s["label"] == label]
        seen = used.get(label, 0)
        used[label] = seen + 1
        match = candidates[min(seen, len(candidates) - 1)] if candidates else None
        lines = match["lines"] if match else []
        segmentations = {"score line": by_line(body), "rest gap": phrases(body, gap)}
        method = next((m for m, f in segmentations.items() if lines and len(f) == len(lines)), "rest gap")
        found = segmentations[method]
        entry = {"index": index, "label": label, "sung": True,
                 "lyric_section": (lyrics.index(match) if match else None),
                 "repeated": bool(match) and seen >= len(candidates),
                 "start_seconds": round(float(starts[id(body[0])]) * spu, 2) if spu else None,
                 "phrase_count": len(found), "line_count": len(lines),
                 **measure(" / ".join(lines) if lines else None, body, score)}
        if lines and len(found) == len(lines):
            entry["segmentation"] = method
            entry["phrases"] = [{"start_seconds": round(float(starts[id(p[0])]) * spu, 2) if spu else None,
                                 **measure(line, p, score)} for p, line in zip(found, lines)]
        else:
            entry["phrases"] = [{"start_seconds": round(float(starts[id(p[0])]) * spu, 2) if spu else None,
                                 **measure(None, p, score)} for p in found]
            entry["phrase_pairing"] = ("no lyric section with this label" if not lines else
                                       f"{len(lines)} lyric lines against " + ", ".join(
                                           f"{len(f)} phrases by {m}" for m, f in segmentations.items())
                                       + "; not paired")
        sections.append(entry)

    sung_lines = [line for s in lyrics for line in s["lines"]]
    all_notes = [e for e in events if e.kind == "note"]
    song = measure(" ".join(sung_lines), all_notes, score)
    song["lyrics"] = None
    song["note"] = ("song-level syllables count each lyric line once; the score may repeat or "
                    "omit sections, so compare section rows for anything precise")
    paired = [p for s in sections for p in s.get("phrases", []) if p.get("lyrics")]
    return {"song": song, "sections": sections, "phrase_gap_beats": gap_beats,
            "bins": {label: sum(1 for p in paired if p["bin"] == label) for *_, label in BINS},
            "paired_phrases": len(paired), "syllable_counter": "vowel-group heuristic (approximate)"}


def alignment_report(result: dict) -> str:
    """The human-readable form, one block per phrase."""
    out = []
    for section in result.get("sections", []):
        if not section.get("sung"):
            continue
        head = f"{section['label'].title()} (score section {section['index']}"
        head += f", from {section['start_seconds']}s score time)" if section.get("start_seconds") is not None else ")"
        out.append(head)
        out.append(f"  section: {section['syllables']} syllables, {section['melody_notes']} notes, "
                   f"notes/syllable {section['notes_per_syllable']}")
        if section.get("phrase_pairing"):
            out.append(f"  ({section['phrase_pairing']})")
        for n, phrase in enumerate(section["phrases"], 1):
            out.append(f"  phrase {n} @ {phrase['start_seconds']}s")
            if phrase.get("lyrics"):
                out.append(f"    lyrics: {phrase['lyrics']}")
                out.append(f"    syllables: {phrase['syllables']}")
            out.append(f"    melody notes: {phrase['melody_notes']}")
            if phrase.get("notes_per_syllable") is not None:
                out.append(f"    notes/syllable: {phrase['notes_per_syllable']}  [{phrase['bin']}]")
            out.append("    manual intelligibility: ")
        out.append("")
    return "\n".join(out)
