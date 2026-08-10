#!/usr/bin/env python3
"""
ekg_view.py — interactive HTML viewer for construal EKG traces.

Emits a single self-contained HTML file: the trace strips on top, the full
text below, linked both ways. Hover a word to light up its position on
every lead; hover a trace to highlight the word; click either to jump the
other into view. Words are tinted by abstractness (terracotta = concrete,
blue = abstract, neutral = mid); function words are plain ink.

Two leads:
  abstractness — the construal wave (v1 dictionary or v2 contextual probe)
  breath       — syllable load between breath points (see breath.py), a
                 sawtooth: load climbs word by word, punctuation vents it

Usage:
    python3 ekg_view.py "Your sentence here." out.html
    python3 ekg_view.py --file speech.txt out.html
    python3 ekg_view.py --file speech.txt --scorer v1 out.html
    python3 ekg_view.py --file speech.txt --no-breath out.html
"""

import argparse
import json
import math
import re

from breath import breath_lead
from construal_ekg import WORD_RE_PATTERN, load_norms, score_text, waveform

WORD_RE = re.compile(WORD_RE_PATTERN)


def segment(text, scored):
    """Zip original text back onto the scored tokens, preserving the
    punctuation/whitespace before each token (and after the last)."""
    out, prev_end = [], 0
    matches = list(WORD_RE.finditer(text))
    assert len(matches) == len(scored), "tokenizer drift between scorer and viewer"
    for m, (tok, s, c) in zip(matches, scored):
        assert m.group(0) == tok
        out.append({"pre": text[prev_end:m.start()], "t": tok,
                    "s": None if s is None else round(float(s), 3), "c": c})
        prev_end = m.end()
    return out, text[prev_end:]


def build_payload(text, scored, title, scorer_name, with_breath=True):
    tokens, tail = segment(text, scored)
    _, _, smooth = waveform(scored)
    content = [t for t in tokens if t["c"]]
    hit = [t for t in content if t["s"] is not None]
    mean = sum(t["s"] for t in hit) / max(len(hit), 1)

    leads = [{"name": "abstractness", "kind": "wave",
              "lo": "concrete", "hi": "abstract", "min": 1, "max": 5,
              "smooth": [round(float(v), 3) for v in smooth]}]
    if with_breath:
        loads, marks = breath_lead(text)
        assert len(loads) == len(tokens), "breath tokenizer drift"
        leads.append({"name": "breath", "kind": "saw",
                      "unit": "syllables since breath", "min": 0,
                      "max": max(5, math.ceil(max(loads) / 5) * 5),
                      "values": loads, "marks": marks})

    return {
        "title": title,
        "scorer": scorer_name,
        "stats": {"tokens": len(tokens), "content": len(content),
                  "scored": len(hit), "mean": round(mean, 2)},
        "leads": leads,
        "tokens": tokens,
        "tail": tail,
    }


TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root {
    --paper: #fdf6f0; --page: #f7efe8; --panel: #fffdfb;
    --grid-major: #e8b4a8; --grid-minor: #f3d9d0;
    --ink: #1f1c19; --ink-2: #6b635c; --trace: #1a1a1a;
    --concrete: #b5432e; --abstract: #3a6ea5; --neutral: #8a857e;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--page); color: var(--ink);
    font: 15px/1.6 Georgia, 'Times New Roman', serif;
  }
  header, main { max-width: 1100px; margin: 0 auto; padding: 0 24px; }
  header { padding-top: 28px; }
  h1 { font: 600 18px/1.3 ui-monospace, 'Cascadia Code', Menlo, monospace;
       margin: 0 0 4px; overflow-wrap: anywhere; }
  .meta { font: 12px/1.5 ui-monospace, Menlo, monospace; color: var(--ink-2); }
  .meta b { color: var(--ink); font-weight: 600; }
  .lead-cap { font: 600 11px/1.4 ui-monospace, Menlo, monospace;
              color: var(--ink-2); text-transform: uppercase;
              letter-spacing: 0.05em; margin: 16px 0 4px; }
  .strip-wrap {
    background: var(--paper); border: 1px solid var(--grid-major);
    border-radius: 6px; overflow-x: auto; position: relative;
  }
  .strip-wrap svg { display: block; }
  .tip {
    position: absolute; pointer-events: none; display: none; z-index: 3;
    background: var(--ink); color: var(--paper); border-radius: 4px;
    padding: 3px 8px; font: 12px/1.5 ui-monospace, Menlo, monospace;
    white-space: nowrap;
  }
  .tip .val { font-weight: 700; }
  .legend { font: 12px/1.6 ui-monospace, Menlo, monospace; color: var(--ink-2);
            margin: 14px 0 22px; }
  .legend .sw { display: inline-block; width: 10px; height: 10px;
                border-radius: 2px; margin: 0 4px 0 12px; vertical-align: -1px; }
  #text {
    background: var(--panel); border: 1px solid #e6ddd4; border-radius: 6px;
    padding: 22px 26px; white-space: pre-wrap; margin-bottom: 20px;
    font-size: 16px;
  }
  #text .w { border-radius: 3px; padding: 0 1px; cursor: default; }
  #text .w.fn { color: var(--ink-2); }
  #text .w.miss { border-bottom: 1px dotted var(--ink-2); }
  #text .w.hot, #text .w:hover { outline: 2px solid var(--ink); }
  details { margin-bottom: 40px; font-size: 13px; }
  summary { cursor: pointer; font: 12px ui-monospace, Menlo, monospace;
            color: var(--ink-2); }
  table { border-collapse: collapse; margin-top: 10px;
          font: 12px/1.6 ui-monospace, Menlo, monospace; }
  td, th { padding: 1px 14px 1px 0; text-align: left; }
  th { color: var(--ink-2); font-weight: 600; }
</style>
</head>
<body>
<script id="payload" type="application/json">__PAYLOAD__</script>
<header>
  <h1 id="title"></h1>
  <div class="meta" id="meta"></div>
</header>
<main>
  <div id="strips"></div>
  <div class="legend">word tint:
    <span class="sw" style="background:#f2c4b8"></span>concrete (1)
    <span class="sw" style="background:#e9e4de"></span>mid (3)
    <span class="sw" style="background:#bcd3ec"></span>abstract (5)
    &nbsp;·&nbsp; dotted underline = unscored &nbsp;·&nbsp; gray = function word
    &nbsp;·&nbsp; breath ticks = punctuation vents
  </div>
  <div id="text"></div>
  <details><summary>data table</summary><table id="table"></table></details>
</main>
<script>
const D = JSON.parse(document.getElementById('payload').textContent);
const N = D.tokens.length;

document.getElementById('title').textContent = D.title;
document.getElementById('meta').innerHTML =
  `leads: <b>${D.leads.map(l => l.name).join(' · ')}</b> · scorer: <b>${D.scorer}</b>` +
  ` · tokens: <b>${D.stats.tokens}</b> · content scored: <b>${D.stats.scored}/${D.stats.content}</b>` +
  ` · mean abstractness: <b>${D.stats.mean}</b>`;

// ---- color: diverging around 3, validated poles ----
const POLES = { strong: ['#b5432e', '#8a857e', '#3a6ea5'],
                chip:   ['#f2c4b8', '#e9e4de', '#bcd3ec'] };
function hex2rgb(h){ return [1,3,5].map(i => parseInt(h.slice(i, i+2), 16)); }
function mix(a, b, t){
  const A = hex2rgb(a), B = hex2rgb(b);
  return 'rgb(' + A.map((v,i) => Math.round(v + (B[i]-v)*t)).join(',') + ')';
}
function colorFor(s, kind){
  const [lo, mid, hi] = POLES[kind];
  const t = Math.max(-1, Math.min(1, (s - 3) / 2));
  return t < 0 ? mix(mid, lo, -t) : mix(mid, hi, t);
}

// ---- strips ----
const PX = 14, ML = 48, MR = 16, MT = 14, MB = 26;
const W = ML + N * PX + MR;
const X = i => ML + (i + 0.5) * PX;
const strips = [];
const stripsHost = document.getElementById('strips');

function makeStrip(lead) {
  const PH = lead.kind === 'wave' ? 200 : 120;
  const H = MT + PH + MB;
  const Y = v => MT + (lead.max - v) / (lead.max - lead.min) * PH;
  const S = [];
  S.push(`<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" font-family="ui-monospace,Menlo,monospace">`);

  const step = lead.kind === 'wave' ? 0.5 : (lead.max > 20 ? 10 : 5);
  for (let v = lead.min; v <= lead.max + 1e-9; v += step) {
    const major = lead.kind === 'wave' ? Number.isInteger(v) : true;
    S.push(`<line x1="${ML}" x2="${W-MR}" y1="${Y(v)}" y2="${Y(v)}"
      stroke="${major ? 'var(--grid-major)' : 'var(--grid-minor)'}" stroke-width="${major ? 0.8 : 0.5}"/>`);
    if (major) S.push(`<text x="${ML-8}" y="${Y(v)+4}" text-anchor="end" font-size="11" fill="var(--ink-2)">${v}</text>`);
  }
  for (let i = 0; i < N; i += 5) {
    const major = i % 25 === 0;
    S.push(`<line x1="${X(i)}" x2="${X(i)}" y1="${MT}" y2="${MT+PH}"
      stroke="${major ? 'var(--grid-major)' : 'var(--grid-minor)'}" stroke-width="${major ? 0.8 : 0.5}"/>`);
    if (major) S.push(`<text x="${X(i)}" y="${H-8}" text-anchor="middle" font-size="10" fill="var(--ink-2)">${i}</text>`);
  }

  if (lead.kind === 'wave') {
    S.push(`<line x1="${ML}" x2="${W-MR}" y1="${Y(3)}" y2="${Y(3)}" stroke="var(--neutral)"
      stroke-width="0.8" stroke-dasharray="5 4" opacity="0.7"/>`);
    S.push(`<path d="${lead.smooth.map((v,i) => (i ? 'L' : 'M') + X(i).toFixed(1) + ' ' + Y(v).toFixed(1)).join('')}"
      fill="none" stroke="var(--trace)" stroke-width="2" stroke-linejoin="round"/>`);
    D.tokens.forEach((tk, i) => {
      if (tk.s === null) return;
      S.push(`<circle class="dot" data-i="${i}" cx="${X(i)}" cy="${Y(tk.s)}" r="3.2"
        fill="${colorFor(tk.s, 'strong')}" stroke="var(--paper)" stroke-width="1"/>`);
    });
  } else {
    // sawtooth: ramp through word loads, near-vertical drop at each vent
    const vents = {};
    lead.marks.forEach(m => { vents[m.after] = m; });
    let d = '';
    lead.values.forEach((v, i) => {
      d += (i ? 'L' : 'M') + X(i).toFixed(1) + ' ' + Y(v).toFixed(1);
      const m = vents[i];
      if (m) d += 'L' + (X(i) + PX * 0.45).toFixed(1) + ' ' + Y(m.to).toFixed(1);
    });
    S.push(`<path d="${d}" fill="none" stroke="var(--trace)" stroke-width="1.6" stroke-linejoin="round"/>`);
    lead.marks.forEach(m => {
      const x = X(m.after) + PX * 0.45;
      S.push(`<line x1="${x}" x2="${x}" y1="${MT + PH}" y2="${MT + PH - 8 - 10 * m.strength}"
        stroke="var(--concrete)" stroke-width="${m.strength >= 1 ? 2 : 1}" opacity="0.8"/>`);
    });
  }
  S.push(`<line class="cross" y1="${MT}" y2="${MT+PH}" stroke="var(--ink)" stroke-width="1" visibility="hidden"/>`);
  S.push('</svg>');

  const cap = document.createElement('div');
  cap.className = 'lead-cap';
  cap.textContent = lead.kind === 'wave'
    ? `${lead.name} (${lead.min} ${lead.lo} — ${lead.max} ${lead.hi})`
    : `${lead.name} (${lead.unit})`;
  const wrap = document.createElement('div');
  wrap.className = 'strip-wrap';
  const tip = document.createElement('div');
  tip.className = 'tip';
  wrap.appendChild(tip);
  wrap.insertAdjacentHTML('beforeend', S.join(''));
  stripsHost.appendChild(cap);
  stripsHost.appendChild(wrap);

  const svg = wrap.querySelector('svg');
  const strip = { lead, wrap, svg, tip, Y,
                  cross: svg.querySelector('.cross'),
                  valueAt: i => lead.kind === 'wave' ? D.tokens[i].s : lead.values[i] };
  svg.addEventListener('mousemove', e => {
    const i = Math.max(0, Math.min(N - 1,
      Math.floor((e.clientX - svg.getBoundingClientRect().left - ML) / PX)));
    light(i, false, false);
  });
  svg.addEventListener('mouseleave', unlight);
  svg.addEventListener('click', e => {
    const i = Math.max(0, Math.min(N - 1,
      Math.floor((e.clientX - svg.getBoundingClientRect().left - ML) / PX)));
    light(i, false, true);
  });
  wrap.addEventListener('scroll', () => {
    strips.forEach(o => { if (o !== strip) o.wrap.scrollLeft = wrap.scrollLeft; });
  });
  return strip;
}
D.leads.forEach(lead => strips.push(makeStrip(lead)));

// ---- text panel ----
const textEl = document.getElementById('text');
textEl.innerHTML = D.tokens.map((tk, i) => {
  const esc = s => s.replace(/&/g,'&amp;').replace(/</g,'&lt;');
  let cls = 'w', style = '';
  if (!tk.c) cls += ' fn';
  else if (tk.s === null) cls += ' miss';
  else style = ` style="background:${colorFor(tk.s, 'chip')}"`;
  return esc(tk.pre) + `<span class="${cls}" id="w${i}" data-i="${i}"${style}>${esc(tk.t)}</span>`;
}).join('') + D.tail;

// ---- linking ----
let hotWord = null;
function light(i, scrollTrace, scrollText) {
  if (hotWord) hotWord.classList.remove('hot');
  hotWord = document.getElementById('w' + i);
  if (hotWord) hotWord.classList.add('hot');
  const tk = D.tokens[i];
  strips.forEach(o => {
    o.svg.querySelectorAll('.dot[r="5.5"]').forEach(d => d.setAttribute('r', 3.2));
    const dot = o.svg.querySelector(`.dot[data-i="${i}"]`);
    if (dot) dot.setAttribute('r', 5.5);
    o.cross.setAttribute('x1', X(i)); o.cross.setAttribute('x2', X(i));
    o.cross.setAttribute('visibility', 'visible');
    const v = o.valueAt(i);
    const label = o.lead.kind === 'wave'
      ? (v === null ? (tk.c ? 'unscored' : 'function word') : v.toFixed(2))
      : v + ' syl';
    o.tip.innerHTML = `${tk.t} · <span class="val">${label}</span>`;
    const ty = o.Y(o.lead.kind === 'wave' ? (v ?? 3) : v);
    o.tip.style.left = X(i) + 'px';
    o.tip.style.top = ty + 'px';
    o.tip.style.transform = ty < 55 ? 'translate(-50%, 30%)' : 'translate(-50%, -130%)';
    o.tip.style.display = 'block';
    if (scrollTrace) o.wrap.scrollTo({left: X(i) - o.wrap.clientWidth / 2, behavior: 'smooth'});
  });
  if (scrollText && hotWord) hotWord.scrollIntoView({block: 'nearest', behavior: 'smooth'});
}
function unlight() {
  if (hotWord) hotWord.classList.remove('hot');
  hotWord = null;
  strips.forEach(o => {
    o.svg.querySelectorAll('.dot[r="5.5"]').forEach(d => d.setAttribute('r', 3.2));
    o.cross.setAttribute('visibility', 'hidden');
    o.tip.style.display = 'none';
  });
}
textEl.addEventListener('mouseover', e => {
  const w = e.target.closest('.w'); if (w) light(+w.dataset.i, false, false);
});
textEl.addEventListener('mouseout', unlight);
textEl.addEventListener('click', e => {
  const w = e.target.closest('.w'); if (w) light(+w.dataset.i, true, false);
});

// ---- data table ----
const saw = D.leads.find(l => l.kind === 'saw');
document.getElementById('table').innerHTML =
  `<tr><th>#</th><th>token</th><th>abstractness</th>${saw ? '<th>breath</th>' : ''}</tr>` +
  D.tokens.map((tk, i) =>
    `<tr><td>${i}</td><td>${tk.t}</td>` +
    `<td>${tk.s === null ? (tk.c ? '—' : 'fn') : tk.s.toFixed(2)}</td>` +
    (saw ? `<td>${saw.values[i]}</td>` : '') + '</tr>'
  ).join('');
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?", help="text to trace (or use --file)")
    ap.add_argument("out", nargs="?", default="trace.html")
    ap.add_argument("--file", help="read text from file")
    ap.add_argument("--scorer", choices=["v1", "v2"], default="v2")
    ap.add_argument("--probe", default="probe.npz")
    ap.add_argument("--title", help="override the page title")
    ap.add_argument("--no-breath", action="store_true",
                    help="omit the breath lead")
    args = ap.parse_args()

    if args.file:
        if args.text:               # with --file, positional slot is the outpath
            args.out = args.text
        text, title = open(args.file).read(), args.file
    else:
        text, title = args.text, args.text[:70]
    if args.title:
        title = args.title

    if args.scorer == "v2":
        from construal_ekg2 import ContextualScorer
        scorer = ContextualScorer(args.probe)
        scored = scorer.score_text(text)
        scorer_name = f"v2 contextual ({scorer.model_name})"
    else:
        scored = score_text(text, load_norms())
        scorer_name = "v1 dictionary (Brysbaert norms)"

    payload = build_payload(text, scored, title, scorer_name,
                            with_breath=not args.no_breath)
    page = (TEMPLATE
            .replace("__TITLE__", payload["title"].replace("<", "&lt;"))
            .replace("__PAYLOAD__", json.dumps(payload).replace("</", "<\\/")))
    with open(args.out, "w") as f:
        f.write(page)
    print(f"tokens: {len(payload['tokens'])} | leads: "
          f"{', '.join(l['name'] for l in payload['leads'])} | wrote {args.out}")


if __name__ == "__main__":
    main()
