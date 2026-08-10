# cl-scan — a construal EKG

Trace the abstractness waveform of a text, word by word. Lexical
concreteness/abstractness is the standard proxy for construal level; scores
are reported on an abstractness scale (1 = concrete … 5 = abstract), so
higher means more abstract. (The underlying Brysbaert norms rate
concreteness 1–5; scorers invert with 6 − rating at the reporting boundary.)
Content words are the R-waves; function words are the isoelectric line —
unscored and interpolated through. The trace is Gaussian-smoothed and
rendered on an ECG-paper strip.

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
a linear probe. Polysemy resolves ("hard problem" ≈ 4.0 vs. "hard surface" ≈
2.5 abstractness) and coverage is 100% of content words. Requires `probe.npz` (committed;
retrain with `train_probe.py`).

```sh
python3 construal_ekg2.py "The hard problem of consciousness." out.png
python3 construal_ekg2.py --file speech.txt out.png
python3 construal_ekg2.py --file a.txt --overlay b.txt out.png   # compare two texts
python3 construal_ekg2.py --demo                                 # polysemy divergence test
```

## Breath lead (`breath.py`)

A second lead, purely mechanical: syllable load between breath points.
Punctuation descends from delivery notation (komma, kolon, periodos were
units of breath before they were marks), so it is read back as breath
structure — load climbs syllable by syllable (CMU pronouncing dictionary,
vowel-cluster fallback) and punctuation vents it: a comma partially, a
period completely. The trace is a sawtooth: "syllables since the speaker
last breathed."

```sh
python3 breath.py --file speech.txt      # breath groups + peak load
```

## Interactive viewer (`ekg_view.py`)

Emits a single self-contained HTML file: stacked trace strips (abstractness
wave + breath sawtooth, scroll- and crosshair-synced) over the full text,
linked both ways. Hover a word to light it up on every lead, hover a trace
to highlight the word, click either to jump the other into view. Words are
tinted by abstractness (terracotta = concrete, blue = abstract); function
words stay plain. See `gettysburg.html` for a sample.

```sh
python3 ekg_view.py "Your sentence here." out.html
python3 ekg_view.py --file speech.txt out.html
python3 ekg_view.py --file speech.txt --scorer v1 out.html   # dictionary scorer
python3 ekg_view.py --file speech.txt --no-breath out.html   # abstractness only
```

## Phrase gallery (`gallery.py`)

Renders `phrases.txt` — lines of `label :: text` under `# section` headers —
as one static HTML page of cards: mini-trace plus tinted text per phrase.
A browsable tour of the instrument across registers (concrete recipes,
corporate abstraction, polysemy pairs, proverbs). See `gallery.html`;
edit `phrases.txt` and re-run to extend it.

```sh
python3 gallery.py                    # phrases.txt -> gallery.html
python3 gallery.py myphrases.txt out.html
```

## Lead coupling (`coupling.py`)

How the breath and construal leads interact, per specimen and pooled:
correlation between breath load and abstractness (negative = words get
more concrete as the breath runs out), and mean abstractness at
emphasis positions (last content word before a strong vent) vs elsewhere.

```sh
python3 coupling.py speech.txt essay.txt
python3 coupling.py --phrases phrases.txt --section "Classic style"
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
