#!/bin/bash
# Regenerates git-viz.html from current repo state
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# Gather data
TOTAL_COMMITS=$(git log --oneline | wc -l | tr -d ' ')
DAYS_ACTIVE=$(git log --format='%ad' --date=format:'%Y-%m-%d' | sort -u | wc -l | tr -d ' ')
DATE_NOW=$(date +%Y-%m-%d)

# LOC by language via cloc
if command -v cloc &>/dev/null; then
  CLOC_JSON=$(cloc --quiet --json --exclude-dir=target,docs,.git,node_modules --not-match-f='\.(svg|json)$' . 2>/dev/null)
else
  echo "cloc not found, install with: brew install cloc"
  exit 1
fi

# Extract top languages as JS array (monochrome shades)
LANGS=$(echo "$CLOC_JSON" | python3 -c "
import json, sys
d = json.load(sys.stdin)
skip = {'header','SUM'}
langs = [(k, d[k]['code']) for k in d if k not in skip]
langs.sort(key=lambda x: -x[1])
shades = ['#fff','#ccc','#aaa','#888','#777','#666','#555','#4a4a4a','#444','#3a3a3a','#333','#2a2a2a']
out = []
for i,(name,loc) in enumerate(langs[:12]):
    if loc < 100: continue
    out.append('{name:%s,loc:%d,color:%s}' % (json.dumps(name), loc, json.dumps(shades[i % len(shades)])))
print('[' + ','.join(out) + ']')
")

# Daily stats
DAILY=$(git log --numstat --format='DATE:%ad' --date=format:'%Y-%m-%d' | awk '
/^DATE:/{date=$0; sub(/^DATE:/,"",date); commits[date]++; if(!(date in seen)){seen[date]=1; days[++n]=date}}
/^[0-9]+\t[0-9]+/{ins[date]+=$1; del[date]+=$2}
END{
  for(i=n;i>=1;i--){d=days[i]; printf "{date:\"%s\",commits:%d,ins:%d,del:%d},",d,commits[d],ins[d]+0,del[d]+0}
}')
DAILY="[${DAILY%,}]"

# Hour distribution
HOURS=$(git log --format='%ad' --date=format:'%H' | awk '{h[$1]++} END{for(i=0;i<24;i++) printf "%d,",h[sprintf("%02d",i)]+0}')
HOURS="[${HOURS%,}]"

# Recent commits
COMMITS=$(git log --oneline -20 | python3 -c "
import json, sys
out = []
for line in sys.stdin:
    line = line.strip()
    h = line[:7]
    msg = line[8:]
    out.append('[%s,%s]' % (json.dumps(h), json.dumps(msg)))
print('[' + ','.join(out) + ']')
")

# Top Rust/CUDA LOC
RUST_LOC=$(echo "$CLOC_JSON" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('Rust',{}).get('code',0))")
CUDA_LOC=$(echo "$CLOC_JSON" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('CUDA',{}).get('code',0))")
TOTAL_LOC=$(echo "$CLOC_JSON" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('SUM',{}).get('code',0))")

fmt_k() { python3 -c "n=$1; print(f'{n/1000:.1f}K') if n>=1000 else print(n)"; }
RUST_DISPLAY=$(fmt_k "$RUST_LOC")
CUDA_DISPLAY=$(fmt_k "$CUDA_LOC")
TOTAL_DISPLAY=$(fmt_k "$TOTAL_LOC")

cat > docs/git-viz.html <<HTMLEOF
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>rvLLM -- Git Activity</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
html{font-size:17px;-webkit-font-smoothing:antialiased;-moz-osx-font-smoothing:grayscale}
body{background:#000;color:#d4d4d4;font-family:Helvetica Neue,Helvetica,Arial,sans-serif;font-weight:300;line-height:1.65;letter-spacing:-0.01em}
a{color:#d4d4d4;text-decoration:none}
a:hover{color:#fff}
::selection{background:#333;color:#fff}

.container{max-width:740px;margin:0 auto;padding:0 2rem}

header{padding:3rem 0 2rem}
header h1{font-size:2.6rem;font-weight:700;letter-spacing:-0.04em;color:#fff;margin-bottom:0.3rem}
header .sub{font-size:1rem;color:#888;font-weight:300;margin-bottom:0.5rem}
header .links{font-size:0.8rem;color:#777;letter-spacing:0.04em}
header .links a{color:#999;margin-right:1.2rem}
header .links a:hover{color:#fff}

section{padding:2rem 0;border-bottom:1px solid #111}
section:last-of-type{border-bottom:none}

h2{font-size:1.3rem;font-weight:600;letter-spacing:-0.03em;color:#fff;margin-bottom:0.3rem}
.section-sub{color:#888;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:1.8rem;font-weight:400}
h3{font-size:0.75rem;text-transform:uppercase;letter-spacing:0.1em;color:#888;font-weight:500;margin:2rem 0 0.8rem}

p{margin-bottom:0.8rem;font-size:1rem;color:#ccc}
strong{font-weight:500;color:#ccc}

.tbl{overflow-x:auto;margin:0.6rem 0 1.4rem}
table{width:100%;border-collapse:collapse;font-size:0.92rem;font-variant-numeric:tabular-nums}
thead th{padding:0.5rem 0.7rem;text-align:left;font-size:0.68rem;text-transform:uppercase;letter-spacing:0.06em;color:#888;font-weight:500;border-bottom:1px solid #222}
th.r,td.r{text-align:right}
tbody td{padding:0.4rem 0.7rem;color:#ccc;border-bottom:1px solid #0d0d0d}
tbody tr:hover{background:#0a0a0a}
tr.peak td{color:#fff;font-weight:500}

code{font-family:SF Mono,Menlo,Consolas,monospace;font-size:0.85em;color:#b0b0b0}

/* Stats row */
.stats-row{display:flex;gap:2rem;margin:1.5rem 0 0.5rem}
.stat{text-align:center;flex:1}
.stat .val{font-size:2.4rem;font-weight:700;color:#fff;letter-spacing:-0.04em;line-height:1.1}
.stat .lbl{font-size:0.68rem;text-transform:uppercase;letter-spacing:0.08em;color:#888;font-weight:400;margin-top:0.15rem}

/* Bar chart */
.bar-chart{display:flex;align-items:flex-end;gap:3px;height:160px;margin:1rem 0}
.bar-col{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:flex-end;height:100%}
.bar{width:100%;border-radius:1px 1px 0 0;transition:height 1.2s cubic-bezier(0.16,1,0.3,1);background:#d4d4d4}
.bar-label{font-size:0.6rem;color:#888;margin-top:0.4rem;letter-spacing:0.02em}
.bar-val{font-size:0.6rem;color:#888;margin-bottom:0.2rem}

/* Lang bars */
.lang-row{display:flex;align-items:center;margin-bottom:0.5rem;font-size:0.82rem}
.lang-name{width:110px;text-align:right;padding-right:1rem;color:#888;flex-shrink:0;font-size:0.78rem}
.lang-bar-bg{flex:1;height:18px;background:#0a0a0a;overflow:hidden}
.lang-bar-fill{height:100%;transition:width 1.2s cubic-bezier(0.16,1,0.3,1)}
.lang-loc{width:70px;text-align:right;padding-left:0.8rem;color:#888;font-size:0.78rem;font-variant-numeric:tabular-nums;flex-shrink:0}

/* Composition bar */
.comp-bar{height:18px;display:flex;overflow:hidden;margin:0.8rem 0 0.6rem}
.comp-seg{height:100%;transition:width 1.2s cubic-bezier(0.16,1,0.3,1)}
.comp-seg:hover{opacity:0.7}
.legend{display:flex;gap:1rem;flex-wrap:wrap;font-size:0.7rem;color:#888}
.legend-item{display:flex;align-items:center;gap:0.3rem}
.legend-dot{width:8px;height:8px}

/* Dual bar chart (ins/del) */
.dual-bar-chart{display:flex;align-items:flex-end;gap:6px;height:140px;margin:1rem 0}
.dual-col{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:flex-end;height:100%}
.dual-pair{width:100%;display:flex;gap:2px;align-items:flex-end}
.dual-pair .bar{flex:1}
.bar.ins{background:#ccc}
.bar.del{background:#555}

/* Commit log */
.commit-entry{display:flex;align-items:baseline;padding:0.35rem 0;border-bottom:1px solid #0d0d0d;font-family:SF Mono,Menlo,Consolas,monospace;font-size:0.72rem}
.commit-entry:last-child{border-bottom:none}
.commit-hash{color:#888;margin-right:0.8rem;flex-shrink:0}
.commit-msg{color:#ccc;flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}

footer{padding:2rem 0 4rem;text-align:center}
footer p{font-size:0.72rem;color:#666;letter-spacing:0.02em}
footer a{color:#666}
footer a:hover{color:#999}

.fade-in{opacity:0;transform:translateY(12px);animation:fadeUp 0.6s forwards}
@keyframes fadeUp{to{opacity:1;transform:translateY(0)}}

@media(max-width:600px){
  html{font-size:13px}
  header h1{font-size:2rem}
  .stats-row{gap:1rem}
  .stat .val{font-size:1.8rem}
  .container{padding:0 1.2rem}
}
</style>
</head>
<body>
<div class="container">

<header>
  <h1>rvLLM</h1>
  <p class="sub">Git activity and lines of code.</p>
  <p class="links">
    <a href="https://github.com/m0at/rvllm">GitHub</a>
    <a href="index.html">Benchmarks</a>
  </p>
</header>

<section class="fade-in">
  <div class="stats-row">
    <div class="stat"><div class="val">${TOTAL_COMMITS}</div><div class="lbl">Commits</div></div>
    <div class="stat"><div class="val">${TOTAL_DISPLAY}</div><div class="lbl">Total LOC</div></div>
    <div class="stat"><div class="val">${RUST_DISPLAY}</div><div class="lbl">Rust</div></div>
    <div class="stat"><div class="val">${CUDA_DISPLAY}</div><div class="lbl">CUDA</div></div>
    <div class="stat"><div class="val">${DAYS_ACTIVE}</div><div class="lbl">Days</div></div>
  </div>
</section>

<section class="fade-in" style="animation-delay:0.1s">
  <h2>Code Composition</h2>
  <p class="section-sub">Lines of code by language</p>
  <div class="comp-bar" id="compBar"></div>
  <div class="legend" id="compLegend"></div>
  <h3>Breakdown</h3>
  <div id="langBars"></div>
</section>

<section class="fade-in" style="animation-delay:0.2s">
  <h2>Daily Commits</h2>
  <p class="section-sub">Commits per day</p>
  <div class="bar-chart" id="dailyChart"></div>
</section>

<section class="fade-in" style="animation-delay:0.3s">
  <h2>Lines Changed</h2>
  <p class="section-sub">Insertions (light) and deletions (dark) per day</p>
  <div class="dual-bar-chart" id="locChart"></div>
</section>

<section class="fade-in" style="animation-delay:0.4s">
  <h2>Hour of Day</h2>
  <p class="section-sub">Commit distribution by hour (UTC)</p>
  <div class="bar-chart" id="hourChart" style="height:120px"></div>
</section>

<section class="fade-in" style="animation-delay:0.5s">
  <h2>Recent Commits</h2>
  <p class="section-sub">Last 20</p>
  <div id="commitLog"></div>
</section>

<footer>
<p>generated ${DATE_NOW} -- run <code>./gen-viz.sh</code> to update</p>
</footer>

</div>
<script>
const langs = ${LANGS};
const daily = ${DAILY};
const hours = ${HOURS};
const commits = ${COMMITS};

// Composition bar + legend
const compBar = document.getElementById('compBar');
const compLegend = document.getElementById('compLegend');
const codeTotal = langs.reduce((s,l) => s+l.loc, 0);
langs.forEach(l => {
  const seg = document.createElement('div');
  seg.className = 'comp-seg';
  seg.style.width = '0%';
  seg.dataset.width = (l.loc / codeTotal * 100) + '%';
  seg.style.background = l.color;
  seg.title = l.name + ': ' + l.loc.toLocaleString();
  compBar.appendChild(seg);
  const li = document.createElement('div');
  li.className = 'legend-item';
  li.innerHTML = '<div class="legend-dot" style="background:'+l.color+'"></div>'+l.name+' '+(l.loc/codeTotal*100).toFixed(1)+'%';
  compLegend.appendChild(li);
});

// Language bars
const barContainer = document.getElementById('langBars');
langs.forEach(l => {
  const pct = l.loc / langs[0].loc * 100;
  const row = document.createElement('div');
  row.className = 'lang-row';
  row.innerHTML = '<span class="lang-name">'+l.name+'</span><div class="lang-bar-bg"><div class="lang-bar-fill" style="width:0%;background:'+l.color+';" data-width="'+pct+'%"></div></div><span class="lang-loc">'+l.loc.toLocaleString()+'</span>';
  barContainer.appendChild(row);
});

// Daily commits
const maxCommits = Math.max(...daily.map(d => d.commits));
const dailyChart = document.getElementById('dailyChart');
daily.forEach(d => {
  const col = document.createElement('div');
  col.className = 'bar-col';
  col.innerHTML = '<div class="bar-val">'+d.commits+'</div><div class="bar" style="height:0%" data-height="'+(d.commits/maxCommits*90)+'%"></div><div class="bar-label">'+d.date+'</div>';
  dailyChart.appendChild(col);
});

// LOC chart (insertions + deletions)
const maxLoc = Math.max(...daily.map(d => Math.max(d.ins, d.del)));
const locChart = document.getElementById('locChart');
daily.forEach(d => {
  const col = document.createElement('div');
  col.className = 'dual-col';
  const insH = d.ins / maxLoc * 85;
  const delH = d.del / maxLoc * 85;
  col.innerHTML = '<div class="bar-val" style="font-size:0.55rem">+'+(d.ins/1000).toFixed(1)+'K / -'+(d.del/1000).toFixed(1)+'K</div><div class="dual-pair"><div class="bar ins" style="height:0%" data-height="'+insH+'%"></div><div class="bar del" style="height:0%" data-height="'+delH+'%"></div></div><div class="bar-label">'+d.date+'</div>';
  locChart.appendChild(col);
});

// Hour chart
const maxHour = Math.max(...hours);
const hourChart = document.getElementById('hourChart');
hours.forEach((h, i) => {
  const col = document.createElement('div');
  col.className = 'bar-col';
  const pct = h / maxHour * 85;
  col.innerHTML = (h > 0 ? '<div class="bar-val">'+h+'</div>' : '<div class="bar-val"></div>') +
    '<div class="bar" style="height:0%;background:#888" data-height="'+pct+'%"></div>' +
    '<div class="bar-label">'+String(i).padStart(2,'0')+'</div>';
  hourChart.appendChild(col);
});

// Commit log
const log = document.getElementById('commitLog');
commits.forEach(([hash, msg]) => {
  const entry = document.createElement('div');
  entry.className = 'commit-entry';
  entry.innerHTML = '<span class="commit-hash">'+hash+'</span><span class="commit-msg">'+msg+'</span>';
  log.appendChild(entry);
});

// Animate
setTimeout(() => {
  document.querySelectorAll('[data-width]').forEach(el => { el.style.width = el.dataset.width; });
  document.querySelectorAll('[data-height]').forEach(el => { el.style.height = el.dataset.height; });
}, 200);
</script>
</body>
</html>
HTMLEOF

echo "docs/git-viz.html regenerated from $(git rev-parse --short HEAD)"
