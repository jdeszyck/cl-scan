#!/usr/bin/env python3
"""
gallery.py — render a gallery of phrases, each with its mini-trace.

Reads phrases.txt (lines of "label :: text", grouped under "# section"
headers), scores every phrase, and writes one static self-contained HTML
page: per phrase, a small ECG strip plus the text tinted by abstractness.
Hover any word or dot for its score (native tooltips — no JS).

Usage:
    python3 gallery.py                          # phrases.txt -> gallery.html
    python3 gallery.py --scorer v1
    python3 gallery.py myphrases.txt out.html
"""

import argparse
import html
import re

from construal_ekg import WORD_RE_PATTERN, load_norms, score_text, waveform

WORD_RE = re.compile(WORD_RE_PATTERN)

# Diverging palette around the 3.0 midpoint (poles CVD-validated).
STRONG = {"lo": "#b5432e", "mid": "#8a857e", "hi": "#3a6ea5"}
CHIP = {"lo": "#f2c4b8", "mid": "#e9e4de", "hi": "#bcd3ec"}


def hex2rgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def mix(a, b, t):
    A, B = hex2rgb(a), hex2rgb(b)
    return "rgb(%d,%d,%d)" % tuple(round(x + (y - x) * t) for x, y in zip(A, B))


def color_for(s, pal):
    t = max(-1.0, min(1.0, (s - 3.0) / 2.0))
    return mix(pal["mid"], pal["lo"], -t) if t < 0 else mix(pal["mid"], pal["hi"], t)


def parse_phrases(path):
    sections = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            sections.append({"name": line.lstrip("# "), "items": []})
            continue
        label, _, text = line.partition("::")
        if not sections:
            sections.append({"name": "", "items": []})
        sections[-1]["items"].append({"label": label.strip(), "text": text.strip()})
    return sections


def mini_strip(scored, px=13, height=96):
    """Static SVG mini-trace for one phrase."""
    n = len(scored)
    ml, mr, mt, mb = 6, 6, 8, 8
    w = ml + n * px + mr
    ph = height - mt - mb
    x = lambda i: ml + (i + 0.5) * px
    y = lambda s: mt + (5 - s) / 4 * ph

    parts = [f'<svg width="{w}" height="{height}" viewBox="0 0 {w} {height}">']
    for s in (1, 2, 3, 4, 5):
        parts.append(f'<line x1="{ml}" x2="{w - mr}" y1="{y(s):.1f}" y2="{y(s):.1f}" '
                     f'stroke="#f3d9d0" stroke-width="0.6"/>')
    parts.append(f'<line x1="{ml}" x2="{w - mr}" y1="{y(3):.1f}" y2="{y(3):.1f}" '
                 f'stroke="#8a857e" stroke-width="0.8" stroke-dasharray="4 3" opacity="0.6"/>')
    known = [(i, s) for i, (_, s, _) in enumerate(scored) if s is not None]
    if len(known) >= 2:
        _, _, smooth = waveform(scored, smooth_sigma=1.2)
        d = "".join(f'{"L" if i else "M"}{x(i):.1f} {y(v):.1f}' for i, v in enumerate(smooth))
        parts.append(f'<path d="{d}" fill="none" stroke="#1a1a1a" '
                     f'stroke-width="1.6" stroke-linejoin="round"/>')
    for i, s in known:
        tok = scored[i][0]
        parts.append(f'<circle cx="{x(i):.1f}" cy="{y(s):.1f}" r="2.8" '
                     f'fill="{color_for(s, STRONG)}" stroke="#fdf6f0" stroke-width="0.8">'
                     f'<title>{html.escape(tok)} · {s:.2f}</title></circle>')
    parts.append("</svg>")
    return "".join(parts)


def tinted_text(text, scored):
    """Original text with each token wrapped; punctuation preserved."""
    out, prev_end = [], 0
    matches = list(WORD_RE.finditer(text))
    assert len(matches) == len(scored), "tokenizer drift"
    for m, (tok, s, c) in zip(matches, scored):
        out.append(html.escape(text[prev_end:m.start()]))
        e = html.escape(tok)
        if not c:
            out.append(f'<span class="fn">{e}</span>')
        elif s is None:
            out.append(f'<span class="miss" title="unscored">{e}</span>')
        else:
            out.append(f'<span class="w" style="background:{color_for(s, CHIP)}" '
                       f'title="{s:.2f}">{e}</span>')
        prev_end = m.end()
    out.append(html.escape(text[prev_end:]))
    return "".join(out)


PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>construal EKG — phrase gallery</title>
<style>
  * { box-sizing: border-box; }
  body { margin: 0; background: #f7efe8; color: #1f1c19;
         font: 15px/1.6 Georgia, 'Times New Roman', serif; }
  main { max-width: 1180px; margin: 0 auto; padding: 28px 24px 60px; }
  h1 { font: 600 18px/1.3 ui-monospace, Menlo, monospace; margin: 0 0 4px; }
  .sub { font: 12px/1.5 ui-monospace, Menlo, monospace; color: #6b635c; margin-bottom: 8px; }
  .legend { font: 12px/1.6 ui-monospace, Menlo, monospace; color: #6b635c; margin-bottom: 26px; }
  .legend .sw { display: inline-block; width: 10px; height: 10px; border-radius: 2px;
                margin: 0 4px 0 12px; vertical-align: -1px; }
  h2 { font: 600 13px/1.4 ui-monospace, Menlo, monospace; color: #6b635c;
       text-transform: uppercase; letter-spacing: 0.06em;
       border-bottom: 1px solid #e6ddd4; padding-bottom: 6px; margin: 34px 0 16px; }
  .cards { display: flex; flex-wrap: wrap; gap: 16px; }
  .card { background: #fffdfb; border: 1px solid #e6ddd4; border-radius: 6px;
          padding: 14px 16px 12px; max-width: 100%; }
  .card .hd { display: flex; gap: 12px; align-items: baseline; margin-bottom: 6px; }
  .card .label { font: 600 12px ui-monospace, Menlo, monospace; }
  .card .mean { font: 11px ui-monospace, Menlo, monospace; color: #6b635c; margin-left: auto; }
  .strip { background: #fdf6f0; border: 1px solid #e8b4a8; border-radius: 4px;
           overflow-x: auto; margin-bottom: 8px; }
  .strip svg { display: block; }
  .card .txt { font-size: 14.5px; max-width: 640px; }
  .txt .w, .txt .miss { border-radius: 3px; padding: 0 1px; }
  .txt .fn { color: #6b635c; }
  .txt .miss { border-bottom: 1px dotted #6b635c; }
</style>
</head>
<body>
<main>
  <h1>construal EKG — phrase gallery</h1>
  <div class="sub">__SUB__</div>
  <div class="legend">word tint:
    <span class="sw" style="background:#f2c4b8"></span>concrete (1)
    <span class="sw" style="background:#e9e4de"></span>mid (3)
    <span class="sw" style="background:#bcd3ec"></span>abstract (5)
    &nbsp;·&nbsp; hover words and dots for scores
  </div>
__BODY__
</main>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phrases", nargs="?", default="phrases.txt")
    ap.add_argument("out", nargs="?", default="gallery.html")
    ap.add_argument("--scorer", choices=["v1", "v2"], default="v2")
    ap.add_argument("--probe", default="probe.npz")
    args = ap.parse_args()

    if args.scorer == "v2":
        from construal_ekg2 import ContextualScorer
        scorer = ContextualScorer(args.probe)
        score = scorer.score_text
        scorer_name = f"v2 contextual ({scorer.model_name})"
    else:
        norms = load_norms()
        score = lambda text: score_text(text, norms)
        scorer_name = "v1 dictionary (Brysbaert norms)"

    body, n_phrases = [], 0
    for sec in parse_phrases(args.phrases):
        body.append(f"<h2>{html.escape(sec['name'])}</h2>\n<div class=\"cards\">")
        for item in sec["items"]:
            scored = score(item["text"])
            vals = [s for _, s, _ in scored if s is not None]
            mean = sum(vals) / max(len(vals), 1)
            body.append(
                '<div class="card"><div class="hd">'
                f'<span class="label">{html.escape(item["label"])}</span>'
                f'<span class="mean">mean {mean:.2f}</span></div>'
                f'<div class="strip">{mini_strip(scored)}</div>'
                f'<div class="txt">{tinted_text(item["text"], scored)}</div></div>')
            n_phrases += 1
        body.append("</div>")

    sub = (f"abstractness, 1 concrete — 5 abstract · scorer: {scorer_name} · "
           f"{n_phrases} phrases · edit phrases.txt and re-run gallery.py")
    page = PAGE.replace("__SUB__", html.escape(sub)).replace("__BODY__", "\n".join(body))
    with open(args.out, "w") as f:
        f.write(page)
    print(f"phrases: {n_phrases} | wrote {args.out}")


if __name__ == "__main__":
    main()
