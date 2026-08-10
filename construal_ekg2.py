#!/usr/bin/env python3
"""
construal_ekg2.py — contextual construal EKG.

Scores every token *in context* by projecting its contextual embedding
through the probe trained by train_probe.py. Polysemy resolves: "hard
problem" and "hard surface" diverge. Coverage is 100% of content words.

The probe predicts Brysbaert concreteness (1=abstract .. 5=concrete);
reported scores are inverted (6 - prediction) to the abstractness scale
(1=concrete .. 5=abstract), matching construal_ekg.py.

Function words remain the isoelectric line (unscored, interpolated) for
comparability with v1.

Usage:
    python3 construal_ekg2.py "The hard problem of consciousness." out.png
    python3 construal_ekg2.py --file speech.txt out.png
    python3 construal_ekg2.py --demo            # polysemy divergence test
    python3 construal_ekg2.py --file a.txt --overlay b.txt out.png

Rendering is imported from construal_ekg.py (v1) — same strip, same grid.
"""

import argparse
import re
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from construal_ekg import FUNCTION_WORDS, render, waveform

PROBE_PATH = "probe.npz"
WORD_RE = re.compile(r"[A-Za-z']+")


class ContextualScorer:
    def __init__(self, probe_path=PROBE_PATH, device=None):
        p = np.load(probe_path, allow_pickle=True)
        self.coef = torch.tensor(p["coef"], dtype=torch.float32)
        self.intercept = float(p["intercept"])
        self.layer = int(p["layer"])
        self.model_name = str(p["model_name"])
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()
        self.coef = self.coef.to(self.device)

    @torch.no_grad()
    def score_words(self, words, window=384, stride=256):
        """Contextual score per word. Long texts are chunked with overlap;
        overlapping word scores are averaged."""
        scores = np.zeros(len(words), dtype=np.float64)
        counts = np.zeros(len(words), dtype=np.float64)
        for start in range(0, len(words), stride):
            chunk = words[start:start + window]
            enc = self.tok(chunk, is_split_into_words=True, truncation=True,
                           max_length=512, return_tensors="pt").to(self.device)
            hidden = self.model(**enc, output_hidden_states=True) \
                         .hidden_states[self.layer][0]          # (seq, dim)
            tok_scores = hidden @ self.coef + self.intercept    # (seq,)
            wids = enc.word_ids(0)
            for pos, wid in enumerate(wids):
                if wid is None:
                    continue
                scores[start + wid] += float(tok_scores[pos])
                counts[start + wid] += 1
            if start + window >= len(words):
                break
        counts[counts == 0] = 1
        return 6.0 - np.clip(scores / counts, 1.0, 5.0)  # invert: abstractness

    def score_text(self, text):
        """Returns v1-compatible scored list: (token, score_or_None, is_content)."""
        tokens = WORD_RE.findall(text)
        per_word = self.score_words(tokens)
        out = []
        for tok, s in zip(tokens, per_word):
            if tok.lower().strip("'") in FUNCTION_WORDS:
                out.append((tok, None, False))
            else:
                out.append((tok, float(s), True))
        return out


def demo(scorer):
    pairs = [
        ("She struck a hard surface.", "hard"),
        ("She posed a hard problem.", "hard"),
        ("He deposited money in the bank.", "bank"),
        ("He fished from the river bank.", "bank"),
        ("The court delivered its judgment.", "court"),
        ("They played on the tennis court.", "court"),
        ("A bright light filled the room.", "light"),
        ("She made a light remark.", "light"),
    ]
    print(f"probe: {scorer.model_name} layer {scorer.layer}\n")
    for sent, target in pairs:
        scored = scorer.score_text(sent)
        for tok, s, _ in scored:
            if tok.lower() == target:
                print(f"  {target:6s} = {s:.2f}   {sent}")


def overlay_render(scored_a, scored_b, labels, outpath):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(16, 4.6), dpi=150)
    ax.set_facecolor("#fdf6f0")
    ax.grid(True, which="major", color="#e8b4a8", lw=0.6, alpha=0.7)
    ax.grid(True, which="minor", color="#f3d9d0", lw=0.3, alpha=0.7)
    ax.minorticks_on()
    for scored, label, color in zip((scored_a, scored_b), labels,
                                    ("#1a1a1a", "#c0392b")):
        xs, _, smooth = waveform(scored)
        ax.plot(xs / max(xs[-1], 1), smooth, lw=1.8, color=color, label=label)
    ax.axhline(3.0, color="#888", lw=0.8, ls="--", alpha=0.6)
    ax.set_ylim(0.8, 5.2)
    ax.set_xlabel("normalized position", fontsize=9)
    ax.set_ylabel("abstractness\n(1 concrete — 5 abstract)", fontsize=9)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    print(f"wrote {outpath}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?", help="text to trace (or use --file)")
    ap.add_argument("out", nargs="?", default="trace2.png")
    ap.add_argument("--file", help="read text from file")
    ap.add_argument("--overlay", help="second file: overlay two traces")
    ap.add_argument("--probe", default=PROBE_PATH)
    ap.add_argument("--demo", action="store_true",
                    help="run polysemy divergence test")
    args = ap.parse_args()

    scorer = ContextualScorer(args.probe)

    if args.demo:
        demo(scorer)
        return

    if args.file:
        if args.text:               # with --file, positional slot is the outpath
            args.out = args.text
        text, title = open(args.file).read(), args.file
    else:
        text, title = args.text, args.text[:70]

    scored = scorer.score_text(text)

    if args.overlay:
        scored_b = scorer.score_text(open(args.overlay).read())
        overlay_render(scored, scored_b, (title, args.overlay), args.out)
        return

    render(scored, f"{title}  [contextual: {scorer.model_name}]",
           args.out, label_all_words=(len(scored) <= 40))
    print(f"tokens: {len(scored)} | wrote {args.out}")


if __name__ == "__main__":
    main()
