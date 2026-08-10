#!/usr/bin/env python3
"""
breath.py — the breath lead: syllable load between breath points.

Punctuation descends from delivery notation (komma, kolon, periodos were
units of breath before they were marks), so we read it back as breath
structure: load climbs syllable by syllable and punctuation vents it —
a comma partially, a period completely. The result is a sawtooth, not a
wave: per-word "syllables since the speaker last breathed."

Syllable counts come from the CMU pronouncing dictionary with a
vowel-cluster fallback for unknown words.

Usage (module):
    loads, marks = breath_lead(text)   # loads[i] = load at word i
Usage (CLI):
    python3 breath.py "text"           # print breath groups + loads
    python3 breath.py --file speech.txt
"""

import re
import sys

from construal_ekg import WORD_RE_PATTERN

try:
    import cmudict
    _CMU = cmudict.dict()
except ImportError:
    _CMU = {}

# How much load survives each breath point (0 = full reset).
RESET = {".": 0.0, "!": 0.0, "?": 0.0, "…": 0.0,
         ";": 0.2, ":": 0.3, "—": 0.4, "--": 0.4, ",": 0.5}

TOKEN_RE = re.compile(WORD_RE_PATTERN + r"|[.!?;:,…]|—|--|\n[ \t]*\n")


def syllables(word):
    w = word.lower().strip("'")
    if not w:
        return 1
    if w in _CMU:
        return max(1, sum(ph[-1].isdigit() for ph in _CMU[w][0]))
    n = len(re.findall(r"[aeiouy]+", w))
    if w.endswith("e") and not w.endswith(("le", "ee", "ye")) and n > 1:
        n -= 1                      # silent e
    return max(1, n)


def breath_lead(text):
    """Per-word breath loads plus vent marks.

    Returns (loads, marks): loads[i] is the syllable load at word i (word
    order identical to WORD_RE_PATTERN tokenization); marks are dicts
    {"after": word_index, "to": vented_load, "strength": 0..1} describing
    the breath taken after that word."""
    loads, marks = [], []
    load = 0.0
    for m in TOKEN_RE.finditer(text):
        tok = m.group(0)
        if tok[0].isalpha() or tok[0] == "'":
            load += syllables(tok)
            loads.append(load)
            continue
        factor = 0.0 if tok.startswith("\n") else RESET[tok]
        vented = load * factor
        if loads and load > vented:
            i = len(loads) - 1
            if marks and marks[-1]["after"] == i:     # e.g. ".\n\n" or ",—"
                marks[-1]["to"] = min(marks[-1]["to"], vented)
                marks[-1]["strength"] = 1 - marks[-1]["to"] / max(load, 1e-9)
            else:
                marks.append({"after": i, "to": round(vented, 1),
                              "strength": round(1 - factor, 2)})
        load = vented
    return [round(v, 1) for v in loads], marks


def breath_groups(text):
    """Split into full-breath groups (period-level) with syllable totals."""
    loads, marks = breath_lead(text)
    words = re.findall(WORD_RE_PATTERN, text)
    groups, start = [], 0
    hard = [m["after"] for m in marks if m["to"] == 0.0]
    for h in hard + ([len(words) - 1] if (not hard or hard[-1] != len(words) - 1) else []):
        chunk = words[start:h + 1]
        if chunk:
            groups.append((chunk, sum(syllables(w) for w in chunk)))
        start = h + 1
    return groups


if __name__ == "__main__":
    if sys.argv[1:2] == ["--file"]:
        text = open(sys.argv[2]).read()
    else:
        text = sys.argv[1]
    loads, marks = breath_lead(text)
    print(f"words: {len(loads)} | peak load: {max(loads):.0f} syllables | "
          f"breath points: {len(marks)}")
    for chunk, total in breath_groups(text):
        head = " ".join(chunk[:6]) + (" …" if len(chunk) > 6 else "")
        print(f"  {total:4.0f} syl | {head}")
