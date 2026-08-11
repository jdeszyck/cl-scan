"""Generate a self-contained HTML rating page from a rating form CSV.

The page shows one sentence at a time with the target word in bold; the
participant rates it 1-7 (buttons or number keys). Per-item response times
are recorded for speeder exclusions. At the end the page tries to POST the
results as JSON to SUBMIT_URL (set it near the top of the generated HTML to
e.g. a Google Apps Script web-app endpoint); whether or not that succeeds,
the participant gets a CSV download link, and COMPLETION_CODE (if set) is
displayed for Prolific.

Usage: python rating_page.py rating_form_A.csv rating_form_A.html --form A
"""
import argparse
import csv
import json

TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  body { font-family: system-ui, sans-serif; background: #f5f4f0; color: #1a1a1a;
         margin: 0; display: flex; justify-content: center; }
  .card { max-width: 660px; width: 100%; margin: 3rem 1rem; background: #fff;
          border: 1px solid #ddd; border-radius: 10px; padding: 2rem; }
  h1 { font-size: 1.3rem; }
  .screen { display: none; }
  .screen.active { display: block; }
  #sent { font-size: 1.45rem; line-height: 1.5; margin: 2rem 0; min-height: 4.5rem; }
  #sent b { background: #fff3c4; padding: 0 .15em; border-radius: 3px; }
  .btns { display: flex; gap: .5rem; justify-content: center; }
  .btns button { width: 3rem; height: 3rem; font-size: 1.2rem; border: 1px solid #bbb;
                 border-radius: 8px; background: #fafafa; cursor: pointer; }
  .btns button:hover { background: #e8f0fe; }
  .anchors { display: flex; justify-content: space-between; font-size: .85rem;
             color: #666; margin-top: .5rem; }
  #prog { color: #888; font-size: .9rem; }
  #bar { height: 5px; background: #e5e5e5; border-radius: 3px; margin-top: .4rem; }
  #fill { height: 100%; width: 0; background: #4a7dbd; border-radius: 3px; }
  input[type=text] { font-size: 1rem; padding: .5rem; width: 100%; box-sizing: border-box;
                     border: 1px solid #bbb; border-radius: 6px; margin: .75rem 0; }
  .go { font-size: 1rem; padding: .6rem 1.4rem; border: none; border-radius: 8px;
        background: #4a7dbd; color: #fff; cursor: pointer; }
  .note { font-size: .9rem; color: #555; }
  code { background: #f0f0f0; padding: .1em .35em; border-radius: 4px; }
</style>
</head>
<body>
<div class="card">

<div id="intro" class="screen active">
  <h1>Word meaning ratings</h1>
  <p>You will see __N__ short sentences, each with one <b>highlighted</b> word.
  Rate how concrete or abstract the <em>meaning of the highlighted word</em> is,
  as it is used in that sentence:</p>
  <p><b>1 = highly concrete</b> &mdash; something you can see, touch, hear, or
  physically act on.<br>
  <b>7 = highly abstract</b> &mdash; an idea, quality, or state with no direct
  sensory referent.</p>
  <p>Use the buttons or the number keys 1&ndash;7. Go with your first
  impression; there are no right answers. The task takes about 15 minutes.</p>
  <label for="pid">Your Prolific ID:</label>
  <input type="text" id="pid" autocomplete="off">
  <button class="go" onclick="start()">Begin</button>
</div>

<div id="task" class="screen">
  <div id="prog"></div>
  <div id="bar"><div id="fill"></div></div>
  <div id="sent"></div>
  <div class="btns" id="btns"></div>
  <div class="anchors"><span>1 = highly concrete</span><span>7 = highly abstract</span></div>
</div>

<div id="done" class="screen">
  <h1>Thank you!</h1>
  <p id="status" class="note"></p>
  <p><a id="dl" download>Download your responses (CSV)</a> and keep the file
  until the study closes, in case of any submission problem.</p>
  <p id="codebox" style="display:none">Your completion code:
  <code id="code"></code></p>
</div>

</div>
<script>
const ITEMS = __ITEMS__;
const FORM = "__FORM__";
const SUBMIT_URL = "";       // optional: endpoint that accepts a JSON POST
const COMPLETION_CODE = "";  // shown at the end if set

let idx = 0, t0 = 0, pid = "";
const rows = [];

const btns = document.getElementById("btns");
for (let v = 1; v <= 7; v++) {
  const b = document.createElement("button");
  b.textContent = v;
  b.onclick = () => rate(v);
  btns.appendChild(b);
}
document.addEventListener("keydown", e => {
  if (!document.getElementById("task").classList.contains("active")) return;
  const v = parseInt(e.key, 10);
  if (v >= 1 && v <= 7) rate(v);
});

function show(id) {
  document.querySelectorAll(".screen").forEach(s => s.classList.remove("active"));
  document.getElementById(id).classList.add("active");
}
function esc(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}
function start() {
  pid = document.getElementById("pid").value.trim();
  if (!pid) { alert("Please enter your Prolific ID."); return; }
  show("task");
  render();
}
function render() {
  document.getElementById("prog").textContent = (idx + 1) + " / " + ITEMS.length;
  document.getElementById("fill").style.width = (100 * idx / ITEMS.length) + "%";
  document.getElementById("sent").innerHTML =
    esc(ITEMS[idx].sentence).replace(/\\[([^\\]]+)\\]/, "<b>$1</b>");
  t0 = performance.now();
}
function rate(v) {
  rows.push([ITEMS[idx].item, v, Math.round(performance.now() - t0)]);
  idx++;
  if (idx < ITEMS.length) render(); else finish();
}
function toCsv() {
  const lines = ["participant,form,item,rating,rt_ms"];
  for (const [item, rating, rt] of rows)
    lines.push([pid, FORM, item, rating, rt].join(","));
  return lines.join("\\n") + "\\n";
}
function finish() {
  show("done");
  const blob = new Blob([toCsv()], { type: "text/csv" });
  const dl = document.getElementById("dl");
  dl.href = URL.createObjectURL(blob);
  dl.download = "ratings_" + FORM + "_" + pid + ".csv";
  if (COMPLETION_CODE) {
    document.getElementById("codebox").style.display = "block";
    document.getElementById("code").textContent = COMPLETION_CODE;
  }
  const status = document.getElementById("status");
  if (!SUBMIT_URL) {
    status.textContent = "Your responses are ready below.";
    return;
  }
  status.textContent = "Submitting your responses\\u2026";
  fetch(SUBMIT_URL, {
    method: "POST",
    headers: { "Content-Type": "text/plain" },
    body: JSON.stringify({ participant: pid, form: FORM, rows: rows }),
  }).then(r => {
    status.textContent = r.ok ? "Responses submitted successfully."
      : "Automatic submission failed \\u2014 please download the CSV below and "
        + "return it via the study page.";
  }).catch(() => {
    status.textContent = "Automatic submission failed \\u2014 please download "
      + "the CSV below and return it via the study page.";
  });
}
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("form_csv", help="rating form CSV (item, sentence, rating)")
    ap.add_argument("out_html")
    ap.add_argument("--form", default="A", help="form label recorded in the data")
    ap.add_argument("--title", default="Word meaning ratings")
    args = ap.parse_args()

    items = []
    with open(args.form_csv) as f:
        for row in csv.DictReader(f):
            items.append({"item": int(row["item"]), "sentence": row["sentence"]})

    page = (TEMPLATE
            .replace("__ITEMS__", json.dumps(items))
            .replace("__FORM__", args.form)
            .replace("__TITLE__", args.title)
            .replace("__N__", str(len(items))))
    with open(args.out_html, "w") as f:
        f.write(page)
    print(f"wrote {args.out_html} ({len(items)} items, form {args.form})")
    print("Before deploying: set SUBMIT_URL (and COMPLETION_CODE) near the top "
          "of the <script> block.")


if __name__ == "__main__":
    main()
