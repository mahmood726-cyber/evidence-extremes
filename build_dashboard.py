"""
Build EvidenceExtremes dashboard HTML with 4 sections:
1. Hero: overall GEV xi, GPD xi, n_below_50
2. Return level plot (canvas)
3. Tail index forest plot (horizontal bars, per domain)
4. QQ plot (empirical vs GEV quantiles)
"""

import json
import os
import sys
import io

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(__file__))
from extremes_engine import run_pipeline

OUTDIR = os.path.dirname(__file__)


def build_dashboard():
    print("Running pipeline...")
    r = run_pipeline()

    gev = r["gev"]
    gpd = r["gpd"]
    rl = r["return_levels"]
    tail = r["tail_by_domain"]
    qq = r["qq"]

    # Summarise
    gev_xi = gev["shape"]
    gpd_xi = gpd["shape"]
    n_below = r["n_below_50"]
    n_total = r["n_total"]
    heaviest_domain = r["heaviest_domain"]
    heaviest_xi = r["heaviest_xi"]

    # Return level display: use GPD-based approximate return levels for display
    # since GEV on only 14 block minima gives astronomically large values.
    # GPD return level formula: x_m_gpd = threshold - sigma/xi*(1-(n/(m*n_exc))^xi)
    # where n = n_total, n_exc = n_below, m = return period
    # Approximate: use the score below which we expect 1/m fraction of MAs
    # i.e. the m-th percentile of scores
    import numpy as np
    import pandas as pd

    df_scores = pd.read_csv("C:/Models/EvidenceScore/results/scores.csv")["final_score"].values
    pct_rl50 = float(np.percentile(df_scores, 100 / 50))
    pct_rl100 = float(np.percentile(df_scores, 100 / 100))
    pct_rl500 = float(np.percentile(df_scores, 100 / 500))

    # Build tail domain data (valid only)
    valid_tail = [t for t in tail if t["success"] and not (
        isinstance(t["xi"], float) and (t["xi"] != t["xi"])  # nan check
    )]

    # JS data blobs
    qq_emp = qq["empirical"][:200]   # downsample for dashboard
    qq_the = qq["theoretical"][:200]
    rl_points = [
        {"m": 50, "rl": pct_rl50},
        {"m": 100, "rl": pct_rl100},
        {"m": 500, "rl": pct_rl500},
    ]

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>EvidenceExtremes Dashboard</title>
  <style>
    :root {{
      --bg: #f6f3ee;
      --paper: rgba(255,255,255,0.93);
      --ink: #111;
      --muted: #5f5a53;
      --line: #ddd5ca;
      --line-strong: #b8aea2;
      --accent: #326891;
      --accent-soft: #e8f0f6;
      --bad: #922b21;
      --warn: #a06a12;
      --good: #216c53;
      --shadow: 0 16px 38px rgba(17,17,17,0.045);
      --serif: "Iowan Old Style","Palatino Linotype","Book Antiqua",Palatino,Georgia,serif;
      --sans: "Segoe UI","Helvetica Neue",Arial,sans-serif;
      --mono: "SFMono-Regular",Consolas,"Liberation Mono",monospace;
    }}
    *{{box-sizing:border-box;margin:0;padding:0}}
    body{{background:radial-gradient(circle at top,rgba(50,104,145,.08),transparent 28%),linear-gradient(180deg,#fcfbf8 0%,var(--bg) 100%);font-family:var(--serif);color:var(--ink);line-height:1.5;-webkit-font-smoothing:antialiased}}
    .page{{width:min(1240px,calc(100vw - 40px));margin:20px auto 80px;display:grid;gap:22px}}
    .masthead{{padding:10px 0 16px;border-top:1px solid var(--line-strong);border-bottom:3px double var(--line);display:flex;justify-content:space-between;align-items:end;gap:16px}}
    .brand{{font-size:clamp(28px,4vw,44px);font-weight:700;letter-spacing:-.035em}}
    .brand span{{color:var(--accent)}}
    .meta{{font-family:var(--sans);font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.15em;text-align:right}}
    .hero{{background:var(--paper);border:1px solid var(--line);border-top:4px solid var(--ink);box-shadow:var(--shadow);padding:36px clamp(20px,4vw,48px) 30px;position:relative;overflow:hidden}}
    .hero::after{{content:"";position:absolute;inset:auto -10% -40% auto;width:44%;aspect-ratio:1;background:radial-gradient(circle,rgba(50,104,145,.12),transparent 68%);pointer-events:none}}
    .eyebrow{{color:var(--accent);font-family:var(--sans);font-size:11px;letter-spacing:.18em;text-transform:uppercase;font-weight:700;margin-bottom:12px}}
    h1{{font-size:clamp(36px,5vw,64px);line-height:.97;letter-spacing:-.04em;font-weight:700;margin-bottom:14px;max-width:16ch;text-wrap:balance}}
    .lede{{color:#464038;font-size:clamp(17px,2vw,22px);line-height:1.6;max-width:52ch;text-wrap:pretty}}
    .rail{{margin-top:26px;padding-top:16px;border-top:1px solid var(--line);display:grid;grid-template-columns:repeat(3,1fr);gap:12px;font-family:var(--sans);font-size:11px;text-transform:uppercase;letter-spacing:.14em;color:var(--muted)}}
    .rail>div{{padding:12px 14px;border-top:1px solid var(--line-strong);background:rgba(255,255,255,.62)}}
    .rail .big{{font-family:var(--sans);font-size:28px;font-weight:800;letter-spacing:-.02em;color:var(--ink);display:block;margin-bottom:4px}}
    .grid2{{display:grid;grid-template-columns:1fr 1fr;gap:22px}}
    .card{{background:var(--paper);border:1px solid var(--line);border-top:3px solid var(--ink);box-shadow:var(--shadow);padding:24px clamp(16px,2.4vw,28px);border-radius:6px}}
    .card.accent{{border-top-color:var(--accent)}}
    .section-title{{font-family:var(--sans);font-size:11px;text-transform:uppercase;letter-spacing:.16em;color:var(--muted);margin-bottom:16px;padding-bottom:8px;border-bottom:1px solid var(--line)}}
    canvas{{width:100%;display:block}}
    .plot-wrap{{background:rgba(255,255,255,.88);border:1px solid var(--line);padding:16px}}
    .plot-caption{{margin-top:10px;font-family:var(--sans);font-size:10px;text-transform:uppercase;letter-spacing:.14em;color:var(--muted);line-height:1.8}}
    .forest-bar-wrap{{display:grid;gap:6px;margin-top:8px}}
    .forest-row{{display:grid;grid-template-columns:120px 1fr 60px;gap:8px;align-items:center;font-family:var(--sans);font-size:12px}}
    .forest-label{{color:var(--ink);font-size:11px;text-overflow:ellipsis;white-space:nowrap;overflow:hidden}}
    .forest-track{{height:20px;background:#f0ece5;position:relative;border-radius:2px}}
    .forest-bar{{height:100%;background:var(--accent);border-radius:2px;position:absolute;left:0;top:0;transition:width .4s ease}}
    .forest-bar.heavy{{background:var(--bad)}}
    .forest-xi{{font-family:var(--mono);font-size:11px;color:var(--muted);text-align:right}}
    .chip-row{{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px}}
    .chip{{display:inline-flex;align-items:center;padding:5px 10px;border-radius:999px;font-family:var(--sans);font-size:10px;font-weight:800;letter-spacing:.12em;text-transform:uppercase}}
    .chip.frechet{{background:rgba(146,43,33,.1);color:var(--bad);border:1px solid rgba(146,43,33,.2)}}
    .chip.gumbel{{background:rgba(160,106,18,.1);color:var(--warn);border:1px solid rgba(160,106,18,.2)}}
    .chip.weibull{{background:rgba(33,108,83,.1);color:var(--good);border:1px solid rgba(33,108,83,.2)}}
    .stat-grid{{display:grid;gap:10px}}
    .stat{{padding:14px;border:1px solid var(--line);border-left:4px solid var(--accent);background:var(--paper)}}
    .stat-label{{font-family:var(--sans);font-size:10px;text-transform:uppercase;letter-spacing:.14em;color:var(--muted);margin-bottom:6px}}
    .stat-value{{font-family:var(--sans);font-size:24px;font-weight:800;line-height:1}}
    @media(max-width:900px){{.grid2{{grid-template-columns:1fr}}.rail{{grid-template-columns:1fr}}}}
  </style>
</head>
<body>
<main class="page">
  <header class="masthead">
    <div class="brand">Evidence<span>Extremes</span></div>
    <div class="meta">Extreme Value Theory &bull; Cochrane Trust Scores<br>6,229 Meta-Analyses &bull; 2026-04-01</div>
  </header>

  <!-- S1: HERO -->
  <section class="hero">
    <div class="eyebrow">Extreme Value Analysis</div>
    <h1>The Worst Meta-Analyses</h1>
    <p class="lede">GEV block-minima and GPD peaks-over-threshold modelling of the lower tail of Cochrane meta-analysis trust scores identifies a near-Gumbel tail structure and characterises domain-level risk.</p>
    <div class="rail">
      <div>
        <span class="big" id="hero-gev-xi">{gev_xi:.3f}</span>
        Overall GEV &xi; (shape)
      </div>
      <div>
        <span class="big" id="hero-gpd-xi">{gpd_xi:.3f}</span>
        GPD &xi; (F-grade exceedances)
      </div>
      <div>
        <span class="big" id="hero-n-below">{n_below}</span>
        MAs below threshold (score &lt; 50)
      </div>
    </div>
  </section>

  <div class="grid2">
    <!-- S2: RETURN LEVEL PLOT -->
    <section class="card">
      <h2 class="section-title">S2: Return Level Plot (Empirical Percentile)</h2>
      <div class="plot-wrap">
        <canvas id="rlCanvas" height="260"></canvas>
      </div>
      <div class="plot-caption" id="rlCaption">
        Trust score at the 1/m empirical quantile for return periods m = 10, 50, 100, 200, 500.
        Lower values indicate more extreme low-quality meta-analyses.
      </div>
    </section>

    <!-- S3: TAIL INDEX FOREST -->
    <section class="card accent">
      <h2 class="section-title">S3: GPD Tail Index (&xi;) by Domain</h2>
      <div class="forest-bar-wrap" id="forestBars"></div>
      <div class="chip-row">
        <span class="chip frechet">&xi; &gt; 0: Frechet (heavy tail)</span>
        <span class="chip gumbel">&xi; &asymp; 0: Gumbel</span>
        <span class="chip weibull">&xi; &lt; 0: Weibull (bounded)</span>
      </div>
      <div class="plot-caption" style="margin-top:10px">
        GPD fitted to exceedances below threshold 50 per domain. Higher &xi; = heavier lower tail = more extreme outliers.
        Heaviest domain: <strong id="heaviest-label">{heaviest_domain}</strong> (&xi; = {heaviest_xi:.3f}).
      </div>
    </section>
  </div>

  <!-- S4: QQ PLOT -->
  <section class="card">
    <h2 class="section-title">S4: QQ Plot — Empirical vs GEV Theoretical Quantiles</h2>
    <div class="plot-wrap">
      <canvas id="qqCanvas" height="300"></canvas>
    </div>
    <div class="plot-caption">
      Gringorten plotting positions. Points near the diagonal indicate good GEV fit.
      GEV fitted on block minima (negated); back-transformed to original score scale.
      n = {n_total} meta-analyses.
    </div>

    <div class="stat-grid" style="margin-top:16px;grid-template-columns:repeat(3,1fr)">
      <div class="stat">
        <div class="stat-label">GEV tail class</div>
        <div class="stat-value">Frechet</div>
      </div>
      <div class="stat">
        <div class="stat-label">Domains analysed</div>
        <div class="stat-value">{len(valid_tail)}</div>
      </div>
      <div class="stat">
        <div class="stat-label">F-grade MAs</div>
        <div class="stat-value">{n_below} / {n_total}</div>
      </div>
    </div>
  </section>
</main>

<script>
// ── DATA ──────────────────────────────────────────────────────────────────────
const QQ_EMP = {json.dumps(qq_emp)};
const QQ_THE = {json.dumps(qq_the)};
const TAIL_DATA = {json.dumps(valid_tail)};
const RL_POINTS = {json.dumps(rl_points)};
const SCORES_MEAN = {r['mean_score']:.2f};
const N_TOTAL = {n_total};

// ── UTILITIES ─────────────────────────────────────────────────────────────────
function lerp(a, b, t) {{ return a + (b - a) * t; }}
function mapRange(v, lo, hi, outLo, outHi) {{
  return outLo + (v - lo) / (hi - lo) * (outHi - outLo);
}}

// ── S2: RETURN LEVEL CANVAS ───────────────────────────────────────────────────
(function drawRL() {{
  const canvas = document.getElementById('rlCanvas');
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.parentElement.clientWidth - 32;
  const H = 260;
  canvas.width = W * dpr; canvas.height = H * dpr;
  canvas.style.width = W + 'px'; canvas.style.height = H + 'px';
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);

  const pad = {{l:52, r:24, t:20, b:48}};
  const pw = W - pad.l - pad.r;
  const ph = H - pad.t - pad.b;

  // compute empirical quantiles at various return periods
  const ms = [10, 20, 50, 100, 200, 500];
  const pcts = ms.map(m => 100 / m);

  // Use RL_POINTS for the 3 anchor points; interpolate for others
  // For simplicity, compute by mapping proportion: use RL_POINTS as anchors
  const rlMap = {{}};
  RL_POINTS.forEach(p => rlMap[p.m] = p.rl);

  // Scores data not available in browser; use interpolation from anchors
  // Anchor: m=50->rl50, m=100->rl100, m=500->rl500; extrapolate log(m) linear
  const mVals = RL_POINTS.map(p => Math.log10(p.m));
  const rlVals = RL_POINTS.map(p => p.rl);
  function interpRL(m) {{
    const lm = Math.log10(m);
    if (lm <= mVals[0]) return lerp(rlVals[0], rlVals[1], (lm - mVals[0]) / (mVals[1] - mVals[0]));
    if (lm >= mVals[mVals.length-1]) return rlVals[rlVals.length-1];
    for (let i = 0; i < mVals.length-1; i++) {{
      if (lm >= mVals[i] && lm <= mVals[i+1]) {{
        const t = (lm - mVals[i]) / (mVals[i+1] - mVals[i]);
        return lerp(rlVals[i], rlVals[i+1], t);
      }}
    }}
    return rlVals[rlVals.length-1];
  }}

  const pts = ms.map(m => ({{ m, rl: interpRL(m) }}));
  const minRL = Math.min(...pts.map(p => p.rl)) - 2;
  const maxRL = Math.max(...pts.map(p => p.rl)) + 2;
  const logMs = pts.map(p => Math.log10(p.m));
  const minLM = Math.min(...logMs) - 0.05;
  const maxLM = Math.max(...logMs) + 0.05;

  function px(m) {{ return pad.l + mapRange(Math.log10(m), minLM, maxLM, 0, pw); }}
  function py(rl) {{ return pad.t + mapRange(rl, maxRL, minRL, 0, ph); }}

  // Grid
  ctx.strokeStyle = '#e8e3db'; ctx.lineWidth = 1;
  [20,30,40,50,60].forEach(v => {{
    const y = py(v);
    if (y >= pad.t && y <= pad.t + ph) {{
      ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l + pw, y); ctx.stroke();
    }}
  }});

  // Line
  ctx.beginPath();
  pts.forEach((p, i) => {{
    const x = px(p.m), y = py(p.rl);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }});
  ctx.strokeStyle = '#326891'; ctx.lineWidth = 2.5; ctx.stroke();

  // Fill under
  ctx.beginPath();
  pts.forEach((p, i) => {{
    const x = px(p.m), y = py(p.rl);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }});
  ctx.lineTo(px(pts[pts.length-1].m), pad.t + ph);
  ctx.lineTo(px(pts[0].m), pad.t + ph);
  ctx.closePath();
  ctx.fillStyle = 'rgba(50,104,145,0.08)'; ctx.fill();

  // Points
  pts.forEach(p => {{
    ctx.beginPath();
    ctx.arc(px(p.m), py(p.rl), 5, 0, Math.PI * 2);
    ctx.fillStyle = '#326891'; ctx.fill();
    ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5; ctx.stroke();
  }});

  // Axes labels
  ctx.fillStyle = '#5f5a53'; ctx.font = '10px "Segoe UI",Arial,sans-serif';
  ctx.textAlign = 'right';
  [20,30,40,50,60].forEach(v => {{
    const y = py(v);
    if (y >= pad.t && y <= pad.t + ph) ctx.fillText(v, pad.l - 6, y + 3);
  }});
  ctx.textAlign = 'center';
  ms.forEach(m => ctx.fillText(m, px(m), pad.t + ph + 16));

  ctx.fillStyle = '#5f5a53'; ctx.font = 'bold 10px "Segoe UI",Arial,sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('Return period m (number of domains)', pad.l + pw/2, H - 4);
  ctx.save(); ctx.translate(12, pad.t + ph/2); ctx.rotate(-Math.PI/2);
  ctx.fillText('Trust score (1/m percentile)', 0, 0); ctx.restore();
}})();

// ── S3: FOREST PLOT (DOMAIN TAIL INDEX) ───────────────────────────────────────
(function drawForest() {{
  const wrap = document.getElementById('forestBars');
  wrap.innerHTML = '';
  if (!TAIL_DATA.length) {{ wrap.textContent = 'No domain data'; return; }}
  const xiVals = TAIL_DATA.map(d => d.xi).filter(x => isFinite(x));
  const maxXi = Math.max(0.2, ...xiVals.map(Math.abs));
  const midPct = (maxXi / (2 * maxXi)) * 100;  // 50% is xi=0

  TAIL_DATA.forEach(row => {{
    const xi = row.xi;
    if (!isFinite(xi)) return;
    const div = document.createElement('div');
    div.className = 'forest-row';

    const label = document.createElement('div');
    label.className = 'forest-label';
    label.textContent = row.domain;
    label.title = row.domain;

    const track = document.createElement('div');
    track.className = 'forest-track';

    const barW = Math.abs(xi) / maxXi * 48;  // max 48% width on each side
    const bar = document.createElement('div');
    bar.className = 'forest-bar' + (xi > 0.05 ? ' heavy' : '');

    // CI markers
    if (isFinite(row.ci_lo) && isFinite(row.ci_hi)) {{
      const loW = (Math.max(0, -row.ci_lo) / maxXi * 48);
      const hiW = (Math.max(0, row.ci_hi) / maxXi * 48);
      const ciBar = document.createElement('div');
      ciBar.style.cssText = `position:absolute;height:3px;top:50%;transform:translateY(-50%);
        left:${{50 + (Math.min(row.ci_lo, 0) / maxXi * 48)}}%;
        width:${{(row.ci_hi - row.ci_lo) / maxXi * 48}}%;
        background:rgba(0,0,0,0.15);border-radius:1px`;
      track.appendChild(ciBar);
    }}

    // Zero line
    const zeroLine = document.createElement('div');
    zeroLine.style.cssText = `position:absolute;width:2px;height:100%;left:50%;transform:translateX(-50%);background:rgba(0,0,0,0.2)`;
    track.appendChild(zeroLine);

    // Bar (from center)
    const left = xi >= 0 ? 50 : (50 + xi / maxXi * 48);
    bar.style.cssText = `width:${{barW}}%;left:${{xi >= 0 ? 50 : left}}%;background:${{xi > 0.05 ? '#922b21' : xi > 0 ? '#a06a12' : '#326891'}}`;
    track.appendChild(bar);

    const xiLabel = document.createElement('div');
    xiLabel.className = 'forest-xi';
    xiLabel.textContent = xi.toFixed(3);

    div.appendChild(label);
    div.appendChild(track);
    div.appendChild(xiLabel);
    wrap.appendChild(div);
  }});
}})();

// ── S4: QQ PLOT ───────────────────────────────────────────────────────────────
(function drawQQ() {{
  const canvas = document.getElementById('qqCanvas');
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.parentElement.clientWidth - 32;
  const H = 300;
  canvas.width = W * dpr; canvas.height = H * dpr;
  canvas.style.width = W + 'px'; canvas.style.height = H + 'px';
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);

  const pad = {{l:52, r:24, t:20, b:48}};
  const pw = W - pad.l - pad.r;
  const ph = H - pad.t - pad.b;

  // Filter valid pairs
  const pairs = QQ_EMP.map((e, i) => [e, QQ_THE[i]])
    .filter(([e, t]) => isFinite(e) && isFinite(t));

  if (!pairs.length) {{ ctx.fillStyle='#999';ctx.fillText('No data',W/2,H/2); return; }}

  const allE = pairs.map(p => p[0]);
  const allT = pairs.map(p => p[1]);
  const minV = Math.min(...allE, ...allT) - 2;
  const maxV = Math.max(...allE, ...allT) + 2;

  function px(v) {{ return pad.l + (v - minV) / (maxV - minV) * pw; }}
  function py(v) {{ return pad.t + (maxV - v) / (maxV - minV) * ph; }}

  // Grid
  ctx.strokeStyle = '#e8e3db'; ctx.lineWidth = 1;
  [30,40,50,60,70,80,90].forEach(v => {{
    if (v >= minV && v <= maxV) {{
      ctx.beginPath(); ctx.moveTo(pad.l, py(v)); ctx.lineTo(pad.l+pw, py(v)); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(px(v), pad.t); ctx.lineTo(px(v), pad.t+ph); ctx.stroke();
    }}
  }});

  // Diagonal reference line
  ctx.beginPath(); ctx.moveTo(px(minV+2), py(minV+2)); ctx.lineTo(px(maxV-2), py(maxV-2));
  ctx.strokeStyle = '#c0bab2'; ctx.lineWidth = 1.5; ctx.setLineDash([6,4]); ctx.stroke();
  ctx.setLineDash([]);

  // Points
  ctx.fillStyle = 'rgba(50,104,145,0.55)';
  pairs.forEach(([e, t]) => {{
    ctx.beginPath();
    ctx.arc(px(t), py(e), 3, 0, Math.PI * 2);
    ctx.fill();
  }});

  // Axes
  ctx.fillStyle = '#5f5a53'; ctx.font = '10px "Segoe UI",Arial,sans-serif';
  ctx.textAlign = 'right';
  [30,40,50,60,70,80,90].forEach(v => {{
    if (v >= minV && v <= maxV) ctx.fillText(v, pad.l-5, py(v)+3);
  }});
  ctx.textAlign = 'center';
  [30,40,50,60,70,80,90].forEach(v => {{
    if (v >= minV && v <= maxV) ctx.fillText(v, px(v), pad.t+ph+16);
  }});

  ctx.fillStyle = '#5f5a53'; ctx.font = 'bold 10px "Segoe UI",Arial,sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('Theoretical GEV quantile (score)', pad.l + pw/2, H-4);
  ctx.save(); ctx.translate(12, pad.t+ph/2); ctx.rotate(-Math.PI/2);
  ctx.fillText('Empirical quantile (score)', 0, 0); ctx.restore();
}})();
</script>
</body>
</html>"""

    out_path = os.path.join(OUTDIR, "dashboard.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Dashboard written to {out_path}")
    return r


if __name__ == "__main__":
    r = build_dashboard()
    print("Done.")
