#!/usr/bin/env python3
"""
train_probe.py — fit a concreteness probe on frozen transformer embeddings.

Embeds each Brysbaert lemma through a frozen pretrained model (mean-pooled
over subword tokens), fits ridge regression against the human 1-5 ratings,
reports held-out Pearson/Spearman r, and saves the probe as probe.npz.

Usage:
    python3 train_probe.py                          # distilbert-base-uncased
    python3 train_probe.py --model microsoft/deberta-v3-small
    python3 train_probe.py --min-known 0.85 --val-frac 0.1

Expect val r ~= 0.88-0.91 with distilbert-base-uncased. The saved probe is
a weight vector + intercept (~KBs) keyed to the model name and layer.
"""

import argparse
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.linear_model import RidgeCV
from scipy.stats import pearsonr, spearmanr

NORMS_PATH = "brysbaert.txt"


def load_norms(path, min_known):
    df = pd.read_csv(path, sep="\t", keep_default_na=False,
                     na_values=[""])                 # 'nan','null' are words
    df = df[df["Bigram"] == 0]                       # single words only
    df = df[df["Percent_known"] >= min_known]        # drop obscure items
    df["Word"] = df["Word"].astype(str).str.lower()
    df = df.drop_duplicates("Word")
    return df["Word"].tolist(), df["Conc.M"].to_numpy(dtype=np.float32)


@torch.no_grad()
def embed_words(words, model, tok, device, layer=-1, batch_size=256):
    """Mean-pooled subword embeddings for isolated words."""
    out = np.empty((len(words), model.config.hidden_size), dtype=np.float32)
    for i in range(0, len(words), batch_size):
        batch = words[i:i + batch_size]
        enc = tok(batch, padding=True, truncation=True, max_length=16,
                  return_tensors="pt", return_special_tokens_mask=True).to(device)
        special = enc.pop("special_tokens_mask")
        hidden = model(**enc, output_hidden_states=True).hidden_states[layer]
        mask = (enc["attention_mask"] * (1 - special)).unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        out[i:i + len(batch)] = pooled.cpu().numpy()
        if (i // batch_size) % 20 == 0:
            print(f"  embedded {i + len(batch)}/{len(words)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--norms", default=NORMS_PATH)
    ap.add_argument("--layer", type=int, default=-1,
                    help="hidden layer to probe (-1 = last)")
    ap.add_argument("--min-known", type=float, default=0.85)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", default="probe.npz")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device} | model: {args.model} | layer: {args.layer}")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()

    words, y = load_norms(args.norms, args.min_known)
    print(f"norms loaded: {len(words)} words")

    X = embed_words(words, model, tok, device, layer=args.layer)

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(words))
    n_val = int(len(words) * args.val_frac)
    val, tr = idx[:n_val], idx[n_val:]

    ridge = RidgeCV(alphas=np.logspace(-1, 4, 12))
    ridge.fit(X[tr], y[tr])
    pred = ridge.predict(X[val])

    pr = pearsonr(y[val], pred)[0]
    sr = spearmanr(y[val], pred)[0]
    mae = np.abs(y[val] - pred).mean()
    print(f"alpha: {ridge.alpha_:.3g}")
    print(f"held-out ({n_val} words): pearson r = {pr:.4f} | "
          f"spearman r = {sr:.4f} | MAE = {mae:.3f}")

    # refit on everything for the shipped probe
    ridge_full = RidgeCV(alphas=np.logspace(-1, 4, 12)).fit(X, y)
    np.savez(args.out,
             coef=ridge_full.coef_.astype(np.float32),
             intercept=np.float32(ridge_full.intercept_),
             model_name=args.model,
             layer=args.layer,
             val_pearson=pr, val_spearman=sr)
    print(f"wrote {args.out}")

    # sanity spot-checks
    for w in ["spoon", "justice", "dog", "freedom", "gravel", "irony"]:
        e = embed_words([w], model, tok, device, layer=args.layer)
        print(f"  {w:10s} -> {ridge_full.predict(e)[0]:.2f}")


if __name__ == "__main__":
    main()
