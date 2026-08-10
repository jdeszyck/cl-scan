#!/usr/bin/env python3
"""
coupling.py — how the breath lead and the construal lead interact.

Per specimen, reports:
  r(load,abs)  correlation between breath load and abstractness across
               content words: negative = words get more concrete as the
               breath runs out
  final-abs    mean abstractness at group-final content words (the last
               content word before a strong vent — the emphasis position,
               where nuclear stress falls)
  other-abs    mean abstractness everywhere else

plus a pooled summary. Near-zero pooled r means the leads measure
distinct things; per-specimen r shows coupling inside crafted prose.

Usage:
    python3 coupling.py file1.txt file2.txt ...
    python3 coupling.py --phrases phrases.txt --section "Classic style"
    python3 coupling.py --scorer v1 file.txt
"""

import argparse

import numpy as np

from breath import breath_lead
from construal_ekg import load_norms, score_text
from gallery import parse_phrases


def group_final_indices(scored, marks, min_strength=0.5):
    """Content-word indices in emphasis position: last content word before
    a vent of at least min_strength, plus the final content word."""
    finals = set()
    for m in marks:
        if m["strength"] < min_strength:
            continue
        j = m["after"]
        while j >= 0 and not scored[j][2]:
            j -= 1
        if j >= 0:
            finals.add(j)
    j = len(scored) - 1
    while j >= 0 and not scored[j][2]:
        j -= 1
    if j >= 0:
        finals.add(j)
    return finals


def analyze(label, text, score):
    scored = score(text)
    loads, marks = breath_lead(text)
    finals = group_final_indices(scored, marks)
    idx = [(i, s) for i, (_, s, c) in enumerate(scored) if c and s is not None]
    if len(idx) < 4:
        print(f"{label:32.32s}  (too few scored words, skipped)")
        return None
    ld = np.array([loads[i] for i, _ in idx])
    ab = np.array([s for _, s in idx])
    fin = np.array([i in finals for i, _ in idx])
    r = np.corrcoef(ld, ab)[0, 1]
    fa = ab[fin].mean() if fin.any() else float("nan")
    oa = ab[~fin].mean() if (~fin).any() else float("nan")
    print(f"{label:32.32s} {r:>+11.3f} {fa:>9.2f} {oa:>9.2f} {int(fin.sum()):>7d}")
    return ld, ab, fin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", help="text files to analyze")
    ap.add_argument("--phrases", help="also analyze each phrase in this phrases file")
    ap.add_argument("--section", help="restrict --phrases to sections whose name "
                                      "starts with this string")
    ap.add_argument("--scorer", choices=["v1", "v2"], default="v2")
    ap.add_argument("--probe", default="probe.npz")
    args = ap.parse_args()
    if not args.files and not args.phrases:
        ap.error("give text files and/or --phrases")

    if args.scorer == "v2":
        from construal_ekg2 import ContextualScorer
        score = ContextualScorer(args.probe).score_text
    else:
        norms = load_norms()
        score = lambda text: score_text(text, norms)

    specimens = [(f, open(f).read()) for f in args.files]
    if args.phrases:
        for sec in parse_phrases(args.phrases):
            if args.section and not sec["name"].startswith(args.section):
                continue
            specimens += [(i["label"], i["text"]) for i in sec["items"]]

    print(f"{'specimen':32s} {'r(load,abs)':>11s} {'final-abs':>9s} "
          f"{'other-abs':>9s} {'n_final':>7s}")
    pools = [analyze(label, text, score) for label, text in specimens]
    pools = [p for p in pools if p is not None]
    if len(pools) > 1:
        ld = np.concatenate([p[0] for p in pools])
        ab = np.concatenate([p[1] for p in pools])
        fin = np.concatenate([p[2] for p in pools])
        print(f"\npooled ({len(ab)} content words):")
        print(f"  r(breath load, abstractness) = {np.corrcoef(ld, ab)[0, 1]:+.3f}")
        print(f"  abstractness at emphasis positions: {ab[fin].mean():.2f} (n={fin.sum()})")
        print(f"  abstractness elsewhere:             {ab[~fin].mean():.2f} (n={(~fin).sum()})")


if __name__ == "__main__":
    main()
