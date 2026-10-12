"""Write a self-contained HTML page for checking free will and fate by eye.

A map of the elementary rules (free will across, fate up) with a look-back
slider, and, if given, the tipping-point sweep. Points are shown only at
look-backs where their estimate is trusted.
"""

import argparse
import json
from pathlib import Path

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Free will and fate</title>
<style>
:root { --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#8a8984; --grid:#e6e5e0;
  --c1:#2a78d6; --c2:#eb6834; --c3:#1baf7a; --c4:#eda100; --mark:#2a78d6; --mark2:#eb6834; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --surface:#1a1a19; --ink:#fff; --ink2:#c3c2b7; --muted:#8f8e86; --grid:#33332f;
  --c1:#3987e5; --c2:#d95926; --c3:#199e70; --c4:#c98500; --mark:#3987e5; --mark2:#d95926; } }
:root[data-theme="dark"] { --surface:#1a1a19; --ink:#fff; --ink2:#c3c2b7; --muted:#8f8e86;
  --grid:#33332f; --c1:#3987e5; --c2:#d95926; --c3:#199e70; --c4:#c98500; --mark:#3987e5;
  --mark2:#d95926; }
body { margin:0; background:var(--surface); color:var(--ink);
  font:15px/1.5 system-ui, -apple-system, sans-serif; }
main { max-width:960px; margin:0 auto; padding:24px 16px 48px; }
h1 { font-size:22px; margin:0 0 4px; } h2 { font-size:17px; margin:32px 0 4px; }
p, li { color:var(--ink2); max-width:70ch; }
.controls { display:flex; gap:16px; align-items:center; flex-wrap:wrap; margin:12px 0; }
.legend { display:flex; gap:14px; flex-wrap:wrap; font-size:13px; color:var(--ink2); }
.legend span { display:inline-flex; align-items:center; gap:6px; }
svg { width:100%; height:auto; display:block; overflow:visible; }
.axis text { fill:var(--muted); font-size:11px; } .axis line { stroke:var(--grid); }
.label { fill:var(--ink2); font-size:11px; pointer-events:none; }
#tip { position:fixed; pointer-events:none; background:var(--surface); color:var(--ink);
  border:1px solid var(--grid); border-radius:6px; padding:6px 8px; font-size:12px;
  display:none; box-shadow:0 2px 8px rgba(0,0,0,.15); }
.multiples { display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:16px; }
details { margin-top:16px; color:var(--ink2); } table { border-collapse:collapse; font-size:12px; }
td, th { padding:2px 8px; text-align:right; border-bottom:1px solid var(--grid); }
</style></head><body><main>
<h1>Free will and fate</h1>
<p><b>Free will</b> (entropy rate): how much is still left to guess about the next cell once you
know the cells before it. <b>Fate</b> (excess entropy): how much those earlier cells told you.
Both are in bits and depend on how far back you look; the slider sets that look-back. A rule is
drawn only where its estimate is trusted.</p>
<div class="controls"><label>Look-back: <input id="L" type="range" min="2" max="24" value="12">
<b id="Lval">12</b> cells</label>
<div class="legend" id="legend"></div></div>
<svg id="map" viewBox="0 0 900 520" role="img" aria-label="Map of rules by free will and fate"></svg>
<details><summary>Table</summary><div id="table"></div></details>
__DP__
<div id="tip"></div>
</main><script>
const RULES = __RULES__, DP = __DPDATA__;
const CLASS = {1:["I","--c1","circle"],2:["II","--c2","square"],3:["III","--c3","diamond"],
  4:["IV","--c4","triangle"]};
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const tip = document.getElementById("tip");
function showTip(e, html) { tip.innerHTML = html; tip.style.display = "block";
  tip.style.left = Math.min(e.clientX + 12, innerWidth - 180) + "px"; tip.style.top = (e.clientY + 12) + "px"; }
function hideTip() { tip.style.display = "none"; }
const NS = "http://www.w3.org/2000/svg";
function el(tag, attrs, parent) { const n = document.createElementNS(NS, tag);
  for (const k in attrs) n.setAttribute(k, attrs[k]); if (parent) parent.appendChild(n); return n; }
function shape(kind, x, y, r, fill, parent) {
  if (kind === "circle") return el("circle", {cx:x, cy:y, r, fill}, parent);
  if (kind === "square") return el("rect", {x:x-r, y:y-r, width:2*r, height:2*r, rx:1, fill}, parent);
  const pts = kind === "diamond" ? [[x,y-r*1.3],[x+r*1.3,y],[x,y+r*1.3],[x-r*1.3,y]]
    : [[x,y-r*1.3],[x+r*1.2,y+r],[x-r*1.2,y+r]];
  return el("polygon", {points: pts.map(p => p.join(",")).join(" "), fill}, parent);
}
function axes(svg, m, w, h, xmax, ymax, xlab, ylab) {
  const g = el("g", {class:"axis"}, svg);
  for (let i = 0; i <= 5; i++) {
    const x = m.l + (w - m.l - m.r) * i / 5, y = h - m.b - (h - m.t - m.b) * i / 5;
    el("line", {x1:x, x2:x, y1:m.t, y2:h-m.b}, g); el("line", {x1:m.l, x2:w-m.r, y1:y, y2:y}, g);
    el("text", {x, y:h-m.b+16, "text-anchor":"middle"}, g).textContent = (xmax*i/5).toFixed(2).replace(/\\.?0+$/,"");
    el("text", {x:m.l-6, y:y+4, "text-anchor":"end"}, g).textContent = (ymax*i/5).toFixed(1).replace(/\\.0$/,"");
  }
  el("text", {x:(m.l+w-m.r)/2, y:h-4, "text-anchor":"middle"}, g).textContent = xlab;
  el("text", {x:12, y:(m.t+h-m.b)/2, "text-anchor":"middle", transform:`rotate(-90 12 ${(m.t+h-m.b)/2})`}, g).textContent = ylab;
}
function drawMap() {
  const L = +document.getElementById("L").value; document.getElementById("Lval").textContent = L;
  const svg = document.getElementById("map"); svg.innerHTML = "";
  const m = {l:52, r:16, t:12, b:40}, w = 900, h = 520;
  const pts = RULES.filter(r => r.trusted[L-1]).map(r => ({...r, fw:r.free_will[L-1], fa:r.fate[L-1]}));
  const ymax = Math.max(1, Math.ceil(Math.max(...pts.map(p => p.fa)) + 0.5));
  axes(svg, m, w, h, 1, ymax, "free will (bits per cell)", "fate (bits)");
  const X = v => m.l + (w-m.l-m.r) * v, Y = v => h - m.b - (h-m.t-m.b) * v / ymax;
  for (const p of pts) {
    const [name, col, kind] = CLASS[p.wolfram_class];
    const s = shape(kind, X(p.fw), Y(p.fa), 5, css(col), svg);
    s.setAttribute("stroke", css("--surface")); s.setAttribute("stroke-width", 2);
    const hit = el("circle", {cx:X(p.fw), cy:Y(p.fa), r:10, fill:"transparent"}, svg);
    hit.addEventListener("mousemove", e => showTip(e, `<b>Rule ${p.rule}</b> · class ${name}<br>free will ${p.fw.toFixed(3)}<br>fate ${p.fa.toFixed(2)}`));
    hit.addEventListener("mouseleave", hideTip);
  }
  // one label per cluster of nearby class III and IV points, so labels never stack
  const groups = [];
  for (const p of pts.filter(q => q.wolfram_class >= 3)) {
    const x = X(p.fw), y = Y(p.fa), g = groups.find(g => Math.hypot(g.x - x, g.y - y) < 12);
    if (g) g.rules.push(p.rule); else groups.push({x, y, rules:[p.rule]});
  }
  for (const g of groups) {
    const left = g.x > w - 140;
    el("text", {x: left ? g.x - 8 : g.x + 8, y: g.y - 8, "text-anchor": left ? "end" : "start",
      class:"label"}, svg).textContent = g.rules.join(", ");
  }
  const hidden = RULES.length - pts.length;
  el("text", {x:w-m.r, y:m.t+12, "text-anchor":"end", class:"label"}, svg).textContent =
    hidden ? `${hidden} rules not trusted at this look-back` : "all rules trusted";
  document.getElementById("table").innerHTML = "<table><tr><th>rule</th><th>class</th><th>free will</th><th>fate</th></tr>" +
    pts.map(p => `<tr><td>${p.rule}</td><td>${CLASS[p.wolfram_class][0]}</td><td>${p.fw.toFixed(3)}</td><td>${p.fa.toFixed(2)}</td></tr>`).join("") + "</table>";
  drawDP(L);
}
function drawDP(L) {
  if (!DP.length) return;
  for (const [id, key, label, col] of [["dpfw","free_will","free will (bits per cell)","--mark"],["dpfa","fate","fate (bits)","--mark2"]]) {
    const svg = document.getElementById(id); svg.innerHTML = "";
    const m = {l:44, r:12, t:12, b:40}, w = 440, h = 300;
    const pts = DP.filter(d => d.surviving_runs > 0 && d.trusted[L-1]).map(d => ({p:d.p, v:d[key][L-1], n:d.surviving_runs}));
    const ymax = key === "free_will" ? 1 : Math.max(1, Math.ceil(Math.max(0, ...pts.map(q => q.v)) + 0.5));
    const p0 = 0.6, p1 = 0.9;
    const X = v => m.l + (w-m.l-m.r) * (v - p0) / (p1 - p0), Y = v => h - m.b - (h-m.t-m.b) * v / ymax;
    const g = el("g", {class:"axis"}, svg);
    for (let i = 0; i <= 3; i++) { const v = p0 + (p1-p0)*i/3, x = X(v), y = h - m.b - (h-m.t-m.b)*i/4;
      el("line", {x1:x, x2:x, y1:m.t, y2:h-m.b}, g); el("line", {x1:m.l, x2:w-m.r, y1:y, y2:y}, g);
      el("text", {x, y:h-m.b+16, "text-anchor":"middle"}, g).textContent = v.toFixed(1);
      el("text", {x:m.l-6, y:y+4, "text-anchor":"end"}, g).textContent = (ymax*i/4).toFixed(2); }
    { const y = m.t; el("line", {x1:m.l, x2:w-m.r, y1:y, y2:y}, g);
      el("text", {x:m.l-6, y:y+4, "text-anchor":"end"}, g).textContent = ymax.toFixed(2); }
    el("text", {x:(m.l+w-m.r)/2, y:h-4, "text-anchor":"middle"}, g).textContent = "spreading probability p";
    el("line", {x1:X(0.705485), x2:X(0.705485), y1:m.t, y2:h-m.b, stroke:css("--muted"), "stroke-dasharray":"4 3"}, svg);
    el("text", {x:X(0.705485)+4, y:m.t+10, class:"label"}, svg).textContent = "tipping point";
    el("text", {x:m.l, y:m.t-2, class:"label"}, svg).textContent = label;
    el("polyline", {points: pts.map(q => `${X(q.p)},${Y(q.v)}`).join(" "), fill:"none", stroke:css(col), "stroke-width":2}, svg);
    for (const q of pts) { const c = el("circle", {cx:X(q.p), cy:Y(q.v), r:4, fill:css(col), stroke:css("--surface"), "stroke-width":2}, svg);
      const hit = el("circle", {cx:X(q.p), cy:Y(q.v), r:9, fill:"transparent"}, svg);
      hit.addEventListener("mousemove", e => showTip(e, `p = ${q.p}<br>${key.replace("_"," ")} ${q.v.toFixed(3)}<br>${q.n} of 200 lattices alive`));
      hit.addEventListener("mouseleave", hideTip); }
  }
}
document.getElementById("legend").innerHTML = Object.values(CLASS).map(([n, c, k]) => {
  const sym = {circle:"●", square:"■", diamond:"◆", triangle:"▲"}[k];
  return `<span><span style="color:var(${c})">${sym}</span>class ${n}</span>`; }).join("");
document.getElementById("L").addEventListener("input", drawMap);
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", drawMap);
drawMap();
</script></body></html>
"""

DP_BLOCK = """<h2>Across a tipping point</h2>
<p>Directed percolation: a cell switches on with probability p if it or its right-hand neighbour is
on. Below the tipping point (dashed) everything dies out. Only lattices still alive after 1,000
steps are measured, and the same look-back slider applies.</p>
<div class="multiples"><svg id="dpfw" viewBox="0 0 440 300"></svg><svg id="dpfa" viewBox="0 0 440 300"></svg></div>"""


def load(d: Path) -> list[dict]:
    return [json.loads(f.read_text()) for f in sorted(d.glob("*.json"))]


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("rules_dir", type=Path)
    a.add_argument("out", type=Path)
    a.add_argument("--dp-dir", type=Path)
    a = a.parse_args()
    keep = ("rule", "wolfram_class", "free_will", "fate", "trusted")
    rules = [{k: r[k] for k in keep} for r in load(a.rules_dir)]
    dp = [{k: r[k] for k in ("p", "surviving_runs", "free_will", "fate", "trusted")}
          for r in load(a.dp_dir)] if a.dp_dir else []
    html = (PAGE.replace("__RULES__", json.dumps(rules)).replace("__DPDATA__", json.dumps(dp))
            .replace("__DP__", DP_BLOCK if dp else ""))
    a.out.write_text(html)


if __name__ == "__main__":
    main()
