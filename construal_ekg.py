#!/usr/bin/env python3
"""
construal_ekg.py — trace the abstractness waveform of a text.

Lead I of the construal EKG: lexical abstractness, from the Brysbaert et al.
2014 concreteness norms (~40k human-rated lemmas). Norms are rated
1=abstract .. 5=concrete; we report the inverted scale (6 - rating), so
1=concrete .. 5=abstract — higher means more abstract.

Content words are the R-waves; function words are the isoelectric line
(interpolated through, not scored).

Usage:
    python3 construal_ekg.py "Your sentence here." out.png
    python3 construal_ekg.py --file speech.txt out.png
"""

import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

NORMS_PATH = "brysbaert.txt"

# Single source of truth for tokenization — shared by v2 and the viewer.
WORD_RE_PATTERN = r"[A-Za-z']+"

# Function words: the carrier, not the signal.
FUNCTION_WORDS = set("""
a an the this that these those some any each every no
i you he she it we they me him her us them my your his its our their
mine yours hers ours theirs myself yourself himself herself itself ourselves themselves
who whom whose which what
and or but nor so yet for
if because although though while whereas unless until since as than whether
in on at by with from to of about against between into through during before
after above below up down out off over under again further then once here there
among amongst amid toward towards upon onto within without beside besides
beyond near across along around behind beneath despite except unlike via
throughout till
is am are was were be been being have has had having do does did doing
will would shall should may might must can could
not n't never neither either both all most more less few many much
very too quite rather just only even also still
""".split())


def load_norms(path=NORMS_PATH):
    df = pd.read_csv(path, sep="\t")
    return dict(zip(df["Word"].astype(str).str.lower(), df["Conc.M"]))


def lemma_candidates(w):
    """Cheap lemmatizer: try the word, then common suffix strips."""
    yield w
    if w.endswith("ies") and len(w) > 4:
        yield w[:-3] + "y"
    if w.endswith("es") and len(w) > 3:
        yield w[:-2]
    if w.endswith("s") and len(w) > 3:
        yield w[:-1]
    if w.endswith("ing") and len(w) > 5:
        yield w[:-3]
        yield w[:-3] + "e"
    if w.endswith("ed") and len(w) > 3:
        yield w[:-2]
        yield w[:-1]
    if w.endswith("ly") and len(w) > 4:
        yield w[:-2]
        if w.endswith("bly"):
            yield w[:-2] + "e"      # nobly -> noble
    if (w.endswith("er") or w.endswith("est")) and len(w) > 4:
        stem = w[:-2] if w.endswith("er") else w[:-3]
        yield stem
        yield stem + "e"            # larger -> large


def score_text(text, norms):
    """Returns list of (token, score_or_None, is_content)."""
    tokens = re.findall(WORD_RE_PATTERN, text)
    out = []
    for tok in tokens:
        w = tok.lower().strip("'")
        if w in FUNCTION_WORDS:
            out.append((tok, None, False))
            continue
        score = None
        for cand in lemma_candidates(w):
            if cand in norms:
                score = 6.0 - norms[cand]   # invert: report abstractness
                break
        out.append((tok, score, True))
    return out


def waveform(scored, smooth_sigma=1.6):
    """Interpolate through unscored tokens, then Gaussian-smooth."""
    n = len(scored)
    xs = np.arange(n, dtype=float)
    known_x = [i for i, (_, s, _) in enumerate(scored) if s is not None]
    known_y = [s for (_, s, _) in scored if s is not None]
    if len(known_x) < 2:
        raise ValueError("Not enough scored words to trace.")
    raw = np.interp(xs, known_x, known_y)
    # Gaussian smoothing
    k = int(smooth_sigma * 4) | 1
    kx = np.arange(k) - k // 2
    kernel = np.exp(-0.5 * (kx / smooth_sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(raw, k // 2, mode="edge")
    smooth = np.convolve(padded, kernel, mode="valid")
    return xs, raw, smooth


def render(scored, title, outpath, label_all_words=False, annotate_n=8):
    xs, raw, smooth = waveform(scored)
    n = len(scored)

    fig_w = max(10, min(0.42 * n, 30)) if label_all_words else max(12, min(0.12 * n, 26))
    fig, ax = plt.subplots(figsize=(fig_w, 4.6), dpi=150)

    # ECG-paper grid
    ax.set_facecolor("#fdf6f0")
    ax.grid(True, which="major", color="#e8b4a8", lw=0.6, alpha=0.7)
    ax.grid(True, which="minor", color="#f3d9d0", lw=0.3, alpha=0.7)
    ax.minorticks_on()

    # The trace
    ax.plot(xs, smooth, color="#1a1a1a", lw=1.8, zorder=3)

    # Content-word markers on the raw signal
    cx = [i for i, (_, s, c) in enumerate(scored) if s is not None]
    cy = [s for (_, s, _) in scored if s is not None]
    ax.plot(cx, cy, "o", ms=3.5, color="#c0392b", zorder=4, alpha=0.85)

    ax.axhline(3.0, color="#888", lw=0.8, ls="--", alpha=0.6)

    ax.set_ylim(0.8, 5.2)
    ax.set_xlim(-1, n)
    ax.set_ylabel("abstractness\n(1 concrete — 5 abstract)", fontsize=9)
    ax.set_title(title, fontsize=11, fontfamily="monospace", loc="left")

    if label_all_words:
        ax.set_xticks(range(n))
        ax.set_xticklabels([t for t, _, _ in scored], rotation=55,
                           ha="right", fontsize=8, fontfamily="monospace")
    else:
        # annotate the extreme content words
        order = sorted(range(len(cx)), key=lambda i: cy[i])
        picks = order[:annotate_n // 2] + order[-annotate_n // 2:]
        seen_x = []
        for i in picks:
            x, y = cx[i], cy[i]
            if any(abs(x - sx) < n * 0.03 for sx in seen_x):
                continue
            seen_x.append(x)
            va = "bottom" if y >= 3 else "top"
            dy = 0.15 if y >= 3 else -0.15
            ax.annotate(scored[x][0].lower(), (x, y), xytext=(x, y + dy),
                        fontsize=8, fontfamily="monospace", ha="center", va=va,
                        color="#1a1a1a")
        ax.set_xlabel("token position", fontsize=9)

    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    return outpath


def coverage_report(scored):
    content = [(t, s) for t, s, c in scored if c]
    hit = [t for t, s in content if s is not None]
    miss = [t for t, s in content if s is None]
    return len(hit), len(miss), miss


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--file":
        text = open(args[1]).read()
        out = args[2] if len(args) > 2 else "trace.png"
        title = args[1]
    else:
        text = args[0]
        out = args[1] if len(args) > 1 else "trace.png"
        title = text[:70] + ("…" if len(text) > 70 else "")

    norms = load_norms()
    scored = score_text(text, norms)
    n_tokens = len(scored)
    render(scored, title, out, label_all_words=(n_tokens <= 40))
    hit, missn, miss = coverage_report(scored)
    print(f"tokens: {n_tokens} | content words scored: {hit} | unscored: {missn}")
    if miss:
        print("unscored words:", ", ".join(sorted(set(w.lower() for w in miss))))
    print(f"wrote {out}")
