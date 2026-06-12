import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

# ── Load submission once at startup ──────────────────────────────────────────
SUBMISSION_PATH = os.environ.get("SUBMISSION_PATH", "submission.csv")

def _load() -> pd.DataFrame:
    df = pd.read_csv(SUBMISSION_PATH)
    df["SK_ID_CURR"] = df["SK_ID_CURR"].astype(int)
    df["TARGET"]     = df["TARGET"].astype(float)
    return df.set_index("SK_ID_CURR")

try:
    DATA = _load()
    print(f"✅ Loaded {len(DATA):,} records from {SUBMISSION_PATH}")
except FileNotFoundError:
    raise RuntimeError(f"submission.csv not found at {SUBMISSION_PATH}")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Home Credit Default Risk API",
    description="Look up default probability by client ID",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Helpers ───────────────────────────────────────────────────────────────────
def _risk_label(prob: float) -> str:
    if prob >= 0.70: return "🔴 Very High Risk"
    if prob >= 0.50: return "🟠 High Risk"
    if prob >= 0.30: return "🟡 Medium Risk"
    return "🟢 Low Risk"

def _risk_color(prob: float) -> str:
    if prob >= 0.70: return "#ef4444"
    if prob >= 0.50: return "#f97316"
    if prob >= 0.30: return "#eab308"
    return "#22c55e"

# ── API Routes ────────────────────────────────────────────────────────────────
@app.get("/predict/{client_id}")
def predict(client_id: int):
    """Return default probability for a single client."""
    if client_id not in DATA.index:
        raise HTTPException(status_code=404, detail=f"Client ID {client_id} not found.")
    prob = float(DATA.loc[client_id, "TARGET"])
    return {
        "SK_ID_CURR":        client_id,
        "default_probability": round(prob, 6),
        "risk_label":        _risk_label(prob),
        "will_default":      prob >= 0.5,
    }

@app.get("/stats")
def stats():
    """Summary statistics of the submission."""
    return {
        "total_clients":  len(DATA),
        "mean_prob":      round(float(DATA["TARGET"].mean()), 6),
        "high_risk_count": int((DATA["TARGET"] >= 0.5).sum()),
        "high_risk_pct":  round(float((DATA["TARGET"] >= 0.5).mean()) * 100, 2),
        "thresholds": {
            "very_high": int((DATA["TARGET"] >= 0.70).sum()),
            "high":      int(((DATA["TARGET"] >= 0.50) & (DATA["TARGET"] < 0.70)).sum()),
            "medium":    int(((DATA["TARGET"] >= 0.30) & (DATA["TARGET"] < 0.50)).sum()),
            "low":       int((DATA["TARGET"] < 0.30).sum()),
        }
    }

@app.get("/health")
def health():
    return {"status": "ok", "records": len(DATA)}

# ── Dashboard (single-page HTML served at /) ─────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def dashboard():
    s = stats()
    t = s["thresholds"]
    total = s["total_clients"]

    def pct(n): return round(n / total * 100, 1)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Home Credit Risk Dashboard</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:'Segoe UI',system-ui,sans-serif;background:#0f172a;color:#e2e8f0;min-height:100vh}}
  header{{background:linear-gradient(135deg,#1e40af,#7c3aed);padding:2rem;text-align:center}}
  header h1{{font-size:1.8rem;font-weight:700;letter-spacing:.5px}}
  header p{{opacity:.8;margin-top:.3rem;font-size:.95rem}}
  .container{{max-width:960px;margin:2rem auto;padding:0 1rem}}

  /* stat cards */
  .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:1rem;margin-bottom:2rem}}
  .card{{background:#1e293b;border-radius:12px;padding:1.2rem;border:1px solid #334155}}
  .card .label{{font-size:.75rem;text-transform:uppercase;letter-spacing:1px;color:#94a3b8;margin-bottom:.4rem}}
  .card .value{{font-size:1.8rem;font-weight:700}}
  .card .sub{{font-size:.8rem;color:#64748b;margin-top:.2rem}}

  /* risk bar */
  .section{{background:#1e293b;border-radius:12px;padding:1.5rem;margin-bottom:1.5rem;border:1px solid #334155}}
  .section h2{{font-size:1rem;font-weight:600;margin-bottom:1rem;color:#cbd5e1}}
  .bar-row{{display:flex;align-items:center;gap:.8rem;margin-bottom:.7rem}}
  .bar-label{{width:120px;font-size:.85rem;color:#94a3b8}}
  .bar-track{{flex:1;background:#0f172a;border-radius:99px;height:22px;overflow:hidden}}
  .bar-fill{{height:100%;border-radius:99px;display:flex;align-items:center;padding-left:.6rem;font-size:.75rem;font-weight:600;color:#fff;transition:width .6s ease}}
  .bar-count{{width:90px;text-align:right;font-size:.85rem;color:#cbd5e1}}

  /* lookup */
  .lookup{{background:#1e293b;border-radius:12px;padding:1.5rem;border:1px solid #334155}}
  .lookup h2{{font-size:1rem;font-weight:600;margin-bottom:1rem;color:#cbd5e1}}
  .input-row{{display:flex;gap:.7rem;flex-wrap:wrap}}
  .input-row input{{flex:1;min-width:180px;padding:.65rem 1rem;background:#0f172a;border:1px solid #334155;border-radius:8px;color:#e2e8f0;font-size:1rem;outline:none}}
  .input-row input:focus{{border-color:#6366f1}}
  .input-row button{{padding:.65rem 1.4rem;background:#6366f1;border:none;border-radius:8px;color:#fff;font-size:1rem;font-weight:600;cursor:pointer}}
  .input-row button:hover{{background:#4f46e5}}
  #result{{margin-top:1rem;padding:1rem;border-radius:8px;display:none}}
  #result.show{{display:block}}
  .result-grid{{display:grid;grid-template-columns:1fr 1fr;gap:.7rem;margin-top:.7rem}}
  .result-item .rl{{font-size:.72rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.8px}}
  .result-item .rv{{font-size:1.1rem;font-weight:600;margin-top:.2rem}}
  .gauge-wrap{{text-align:center;margin-top:.5rem}}
  .gauge-wrap .big-prob{{font-size:3rem;font-weight:800}}
  footer{{text-align:center;padding:1.5rem;color:#475569;font-size:.8rem}}
</style>
</head>
<body>
<header>
  <h1>🏦 Home Credit Default Risk</h1>
  <p>Ensemble model predictions · {total:,} clients</p>
</header>

<div class="container">

  <!-- Stat cards -->
  <div class="cards">
    <div class="card">
      <div class="label">Total Clients</div>
      <div class="value" style="color:#818cf8">{total:,}</div>
    </div>
    <div class="card">
      <div class="label">Mean Default Prob</div>
      <div class="value" style="color:#f59e0b">{s['mean_prob']:.1%}</div>
    </div>
    <div class="card">
      <div class="label">High Risk (≥50%)</div>
      <div class="value" style="color:#f87171">{s['high_risk_count']:,}</div>
      <div class="sub">{s['high_risk_pct']}% of portfolio</div>
    </div>
    <div class="card">
      <div class="label">Low Risk (&lt;30%)</div>
      <div class="value" style="color:#4ade80">{t['low']:,}</div>
      <div class="sub">{pct(t['low'])}% of portfolio</div>
    </div>
  </div>

  <!-- Risk distribution bar -->
  <div class="section">
    <h2>Risk Distribution</h2>
    <div class="bar-row">
      <div class="bar-label">🔴 Very High ≥70%</div>
      <div class="bar-track"><div class="bar-fill" style="width:{pct(t['very_high'])}%;background:#ef4444">{pct(t['very_high'])}%</div></div>
      <div class="bar-count">{t['very_high']:,}</div>
    </div>
    <div class="bar-row">
      <div class="bar-label">🟠 High 50–70%</div>
      <div class="bar-track"><div class="bar-fill" style="width:{pct(t['high'])}%;background:#f97316">{pct(t['high'])}%</div></div>
      <div class="bar-count">{t['high']:,}</div>
    </div>
    <div class="bar-row">
      <div class="bar-label">🟡 Medium 30–50%</div>
      <div class="bar-track"><div class="bar-fill" style="width:{pct(t['medium'])}%;background:#eab308">{pct(t['medium'])}%</div></div>
      <div class="bar-count">{t['medium']:,}</div>
    </div>
    <div class="bar-row">
      <div class="bar-label">🟢 Low &lt;30%</div>
      <div class="bar-track"><div class="bar-fill" style="width:{pct(t['low'])}%;background:#22c55e">{pct(t['low'])}%</div></div>
      <div class="bar-count">{t['low']:,}</div>
    </div>
  </div>

  <!-- Client lookup -->
  <div class="lookup">
    <h2>🔍 Client Lookup</h2>
    <div class="input-row">
      <input id="clientInput" type="number" placeholder="Enter SK_ID_CURR  e.g. 100001" />
      <button onclick="lookup()">Check Risk</button>
    </div>
    <div id="result"></div>
  </div>

</div>

<footer>Home Credit Default Risk · Deployed via GitHub Actions → Docker → Hugging Face</footer>

<script>
async function lookup() {{
  const id  = document.getElementById('clientInput').value.trim();
  const box = document.getElementById('result');
  if (!id) return;
  box.className = 'show';
  box.innerHTML = '<span style="color:#94a3b8">Loading…</span>';
  try {{
    const r = await fetch(`/predict/${{id}}`);
    if (!r.ok) {{
      const e = await r.json();
      box.style.background = '#1e1e2e';
      box.innerHTML = `<span style="color:#f87171">⚠️ ${{e.detail}}</span>`;
      return;
    }}
    const d = await r.json();
    const prob  = d.default_probability;
    const color = prob >= .7 ? '#ef4444' : prob >= .5 ? '#f97316' : prob >= .3 ? '#eab308' : '#22c55e';
    box.style.background = '#0f172a';
    box.style.border = `1px solid ${{color}}44`;
    box.innerHTML = `
      <div class="gauge-wrap">
        <div class="big-prob" style="color:${{color}}">${{(prob*100).toFixed(2)}}%</div>
        <div style="font-size:1.1rem;margin-top:.3rem">${{d.risk_label}}</div>
      </div>
      <div class="result-grid" style="margin-top:1rem">
        <div class="result-item"><div class="rl">Client ID</div><div class="rv">${{d.SK_ID_CURR}}</div></div>
        <div class="result-item"><div class="rl">Will Default</div><div class="rv" style="color:${{d.will_default?'#f87171':'#4ade80'}}">${{d.will_default ? 'Yes ⚠️' : 'No ✅'}}</div></div>
        <div class="result-item"><div class="rl">Raw Score</div><div class="rv">${{prob.toFixed(6)}}</div></div>
        <div class="result-item"><div class="rl">Threshold</div><div class="rv">0.50</div></div>
      </div>`;
  }} catch(e) {{
    box.innerHTML = `<span style="color:#f87171">Error: ${{e.message}}</span>`;
  }}
}}
document.getElementById('clientInput').addEventListener('keydown', e => e.key==='Enter' && lookup());
</script>
</body>
</html>"""
    return HTMLResponse(content=html)
