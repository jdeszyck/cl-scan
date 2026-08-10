# cl-scan — a construal EKG

Trace the concreteness/abstractness waveform of a text, word by word.
Concreteness (1 = abstract … 5 = concrete) is the standard lexical proxy for
construal level. Content words are the R-waves; function words are the
isoelectric line — unscored and interpolated through. The trace is
Gaussian-smoothed and rendered on an ECG-paper strip.

## Setup

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple \
  -r requirements.txt
```

(Drop the pytorch index flags if you want a CUDA build of torch.)

## v1 — dictionary scorer (`construal_ekg.py`)

Looks each content word up in the Brysbaert et al. (2014) concreteness norms
(`brysbaert.txt`, ~40k human-rated lemmas) with a cheap suffix-stripping
lemmatizer. Fast, transparent, but type-level: every sense of a word gets the
same score, and out-of-dictionary words go unscored.

```sh
python3 construal_ekg.py "Your sentence here." out.png
python3 construal_ekg.py --file speech.txt out.png
```

## v2 — contextual scorer (`construal_ekg2.py`)

Scores every token *in context* by projecting its contextual embedding through
a linear probe. Polysemy resolves ("hard problem" ≈ 2.0 vs. "hard surface" ≈
3.5) and coverage is 100% of content words. Requires `probe.npz` (committed;
retrain with `train_probe.py`).

```sh
python3 construal_ekg2.py "The hard problem of consciousness." out.png
python3 construal_ekg2.py --file speech.txt out.png
python3 construal_ekg2.py --file a.txt --overlay b.txt out.png   # compare two texts
python3 construal_ekg2.py --demo                                 # polysemy divergence test
```

## Training the probe (`train_probe.py`)

Embeds each Brysbaert lemma through a frozen transformer
(default `distilbert-base-uncased`, last layer, mean-pooled subwords) and fits
ridge regression against the human ratings.

```sh
python3 train_probe.py
python3 train_probe.py --model microsoft/deberta-v3-small --layer -2
```

The committed probe: distilbert-base-uncased, last layer, held-out
Pearson r = 0.864, Spearman r = 0.859, MAE = 0.41 on 3,705 words.

## Data

`brysbaert.txt` — Brysbaert, Warriner & Kuperman (2014), "Concreteness ratings
for 40 thousand generally known English word lemmas", *Behavior Research
Methods*. Tab-separated; key columns are `Word`, `Conc.M` (mean rating),
`Percent_known`, `Bigram`.

`gettysburg_trace.png`, `sentence1_trace.png` — sample v1 output.
