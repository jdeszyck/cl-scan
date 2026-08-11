"""Validate the contextual probe on polysemy/metaphor minimal pairs.

Each pair in polysemy_pairs.tsv uses the SAME surface form in a
concrete-sense sentence and an abstract-sense sentence. The v1 dictionary
scorer assigns both occurrences an identical norm score by construction
(delta = 0 on every pair), so any discrimination the v2 probe shows is
contextual signal, not lexical lookup.

Reports sign accuracy (does the abstract-sense occurrence score more
abstract?), mean delta in abstractness units with a bootstrap CI, an exact
binomial test against chance, a per-POS breakdown, and the failing pairs.

Usage: python validate_probe.py [--pairs polysemy_pairs.tsv] [--probe probe.npz]
       python validate_probe.py --sheet rating_sheet.csv   # blinded rating sheet
"""
import argparse
import math
import re
import sys

import numpy as np

from construal_ekg import WORD_RE_PATTERN, load_norms, lemma_candidates

WORD_RE = re.compile(WORD_RE_PATTERN)


def load_pairs(path):
    pairs = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line or line.startswith("#") or line.startswith("word\t"):
                continue
            parts = line.split("\t")
            if len(parts) != 4:
                sys.exit(f"{path}:{lineno}: expected 4 tab-separated fields")
            pairs.append(dict(zip(("word", "pos", "concrete", "abstract"), parts)))
    return pairs


def target_score(scored, word):
    """Score of the target word, which must occur exactly once and be scored."""
    hits = [s for tok, s, is_content in scored if tok.lower() == word.lower()]
    if len(hits) != 1:
        raise ValueError(f"target '{word}' occurs {len(hits)} times")
    if hits[0] is None:
        raise ValueError(f"target '{word}' is masked as a function word")
    return hits[0]


def binom_p_one_sided(k, n):
    """P(X >= k) for X ~ Binomial(n, 0.5)."""
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n


def write_sheet(pairs, out_path):
    """Emit a blinded, shuffled human-rating sheet plus a separate answer key.

    One row per sentence (2 per pair), fixed-seed shuffle, target word
    bracketed. The sheet carries no pair grouping or sense labels; those
    live only in the key file, which raters must not see.
    """
    import csv
    import random

    items = []
    for p in pairs:
        for sense in ("concrete", "abstract"):
            marked = re.sub(rf"\b{re.escape(p['word'])}\b",
                            lambda m: f"[{m.group(0)}]",
                            p[sense], count=1, flags=re.IGNORECASE)
            items.append({"word": p["word"], "pos": p["pos"],
                          "sense": sense, "sentence": marked})
    random.Random(0).shuffle(items)

    key_path = re.sub(r"\.csv$", "", out_path) + "_key.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item", "sentence", "rating"])
        for i, it in enumerate(items, 1):
            w.writerow([i, it["sentence"], ""])
    with open(key_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item", "word", "pos", "sense"])
        for i, it in enumerate(items, 1):
            w.writerow([i, it["word"], it["pos"], it["sense"]])

    print(f"wrote {len(items)} items to {out_path} (key: {key_path})")
    print("\nRater instructions:")
    print("  Rate the meaning of the [bracketed] word as used in the sentence,")
    print("  on a 1-7 scale: 1 = highly concrete (something you can see, touch,")
    print("  hear, or physically act on), 7 = highly abstract (an idea, quality,")
    print("  or state with no direct sensory referent). Enter the number in the")
    print("  'rating' column. Keep the key file away from raters.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="polysemy_pairs.tsv")
    ap.add_argument("--probe", default="probe.npz")
    ap.add_argument("--sheet", metavar="OUT.csv",
                    help="write a blinded human-rating sheet (+ answer key) "
                         "and exit instead of running the probe")
    args = ap.parse_args()

    pairs = load_pairs(args.pairs)
    if args.sheet:
        write_sheet(pairs, args.sheet)
        return
    by_pos = {}
    for p in pairs:
        by_pos.setdefault(p["pos"], []).append(p)
    counts = ", ".join(f"{pos} {len(ps)}" for pos, ps in sorted(by_pos.items()))
    print(f"pairs: {len(pairs)} ({counts})")

    # v1 control: context-free norm lookup gives both occurrences the same
    # score, so delta is 0 on every pair the dictionary covers at all.
    norms = load_norms()
    covered = sum(1 for p in pairs
                  if any(c in norms for c in lemma_candidates(p["word"])))
    print(f"v1 dictionary control: delta = 0 by construction on all pairs "
          f"({covered}/{len(pairs)} targets in norms)")

    from construal_ekg2 import ContextualScorer
    scorer = ContextualScorer(args.probe)
    deltas = []
    for p in pairs:
        lo = target_score(scorer.score_text(p["concrete"]), p["word"])
        hi = target_score(scorer.score_text(p["abstract"]), p["word"])
        p["delta"] = hi - lo
        deltas.append(p["delta"])
    deltas = np.array(deltas)

    correct = int((deltas > 0).sum())
    n = len(deltas)
    rng = np.random.default_rng(0)
    boots = rng.choice(deltas, size=(10000, n), replace=True).mean(axis=1)
    lo_ci, hi_ci = np.percentile(boots, [2.5, 97.5])
    print("\nv2 contextual probe:")
    print(f"  sign accuracy: {correct}/{n} = {correct / n:.1%}"
          f"   (one-sided binomial p vs chance = {binom_p_one_sided(correct, n):.2g})")
    print(f"  mean delta (abstract sense - concrete sense): "
          f"{deltas.mean():+.2f}  [95% CI {lo_ci:+.2f} .. {hi_ci:+.2f}]")

    print("\n  by POS:")
    for pos, ps in sorted(by_pos.items()):
        d = np.array([p["delta"] for p in ps])
        print(f"    {pos:<5} n={len(ps):3d}  acc {int((d > 0).sum())}/{len(ps)}"
              f"  mean delta {d.mean():+.2f}")

    failures = sorted((p for p in pairs if p["delta"] <= 0), key=lambda p: p["delta"])
    if failures:
        print(f"\n  failures ({len(failures)}):")
        for p in failures:
            print(f"    {p['word']:<12} {p['delta']:+.2f}  "
                  f"\"{p['concrete']}\" / \"{p['abstract']}\"")
    else:
        print("\n  no failures")


if __name__ == "__main__":
    main()
