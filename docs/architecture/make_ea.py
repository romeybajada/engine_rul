"""Draws the four-layer EA diagram (Step 5) in the style of Session 1, Figure 2:
python docs/architecture/make_ea.py docs/architecture/ea-layers.svg, then rsvg-convert -z 2 to PNG."""
import sys
from xml.sax.saxutils import escape

W = 1100
LEFT = 40                                       # band x
X0, CW, GAP = 226, 262, 18                      # first box x, box width, gap
COLS = [X0 + i * (CW + GAP) for i in range(3)]
BAND_H, BAND_GAP, TOP = 96, 18, 66
BOX_H = 62
ORDER = ["Business", "Data", "Application", "Technology"]   # Session 1 order
BANDS = {b: TOP + i * (BAND_H + BAND_GAP) for i, b in enumerate(ORDER)}
H = BANDS["Technology"] + BAND_H + 62

INK, MUTED, LINE = "#1d2433", "#5b6475", "#c9ced8"
MINE, MINE_FILL = "#1f5fbf", "#e8f0fb"
PATTERN = {"online": "#0b7a5c", "batch": "#9a5b00", "human": "#8a2f7a"}
QUESTION = {"Business": "Why, and who?", "Data": "What information?", "Application": "Which systems?", "Technology": "Running on what?"}

out = []
def add(s): out.append(s)

def text(x, y, s, size=13, weight="normal", fill=INK, anchor="start", style=""):
    add(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}" {style}>{escape(s)}</text>')

def pill(x, y, label, key):
    pw = 8 + 6.4 * len(label)
    add(f'<rect x="{x - pw}" y="{y}" width="{pw}" height="17" rx="8.5" fill="{PATTERN[key]}"/>')
    text(x - pw / 2, y + 12, label, 9.5, "bold", "#ffffff", "middle", 'letter-spacing="0.4"')

def box(col, band, title, line, kind="ext", pattern=None):
    """kind: 'model' (the new component), 'mine' (built in Entry 1) or 'ext' (existing estate)."""
    x, y = COLS[col], BANDS[band] + (BAND_H - BOX_H) / 2
    if kind == "ext":
        add(f'<rect x="{x}" y="{y}" width="{CW}" height="{BOX_H}" rx="6" fill="#ffffff" stroke="{LINE}" stroke-width="1.3" stroke-dasharray="5 4"/>')
    else:
        width = 2 if kind == "model" else 1.3
        fill = MINE_FILL if kind == "model" else "#ffffff"
        add(f'<rect x="{x}" y="{y}" width="{CW}" height="{BOX_H}" rx="6" fill="{fill}" stroke="{MINE}" stroke-width="{width}"/>')
    text(x + 12, y + 24, title, 13.5, "bold")
    text(x + 12, y + 44, line, 12, fill=MUTED)
    if pattern:
        pill(x + CW - 10, y + 9, *pattern)

add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Helvetica, Arial, sans-serif">')
add('<defs><marker id="head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
    f'<path d="M0,0 L10,5 L0,10 z" fill="{MUTED}"/></marker></defs>')
add(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')
text(LEFT, 38, "The RUL service sits in one layer but touches all four", 17, "bold")

for band, y in BANDS.items():
    add(f'<rect x="{LEFT}" y="{y}" width="{W - 2 * LEFT}" height="{BAND_H}" rx="8" fill="#ffffff" stroke="{LINE}" stroke-width="1.3"/>')
    text(LEFT + 18, y + 42, band, 15, "bold")
    text(LEFT + 18, y + 62, QUESTION[band], 12, fill=MUTED)

# The middle column is the model's own chain, joined by one dashed line, as in Session 1
cx = COLS[1] + CW / 2
for upper, lower in zip(ORDER, ORDER[1:]):
    y1 = BANDS[upper] + (BAND_H + BOX_H) / 2
    y2 = BANDS[lower] + (BAND_H - BOX_H) / 2
    add(f'<line x1="{cx}" y1="{y1}" x2="{cx}" y2="{y2}" stroke="{MINE}" stroke-width="1.5" stroke-dasharray="4 4"/>')

box(0, "Business", "Certifying staff", "Decide fitness to fly", pattern=("HUMAN-IN-THE-LOOP", "human"))
box(1, "Business", "CAMO planners", "Act on its output: 4 slots a week", pattern=("HUMAN-IN-THE-LOOP", "human"))
box(2, "Business", "Service owner (me)", "Acts on drift alerts", kind="mine")

box(0, "Data", "Engine-health feed", "One snapshot per engine per flight")
box(1, "Data", "Model artefact", "Trained on linux/amd64, versioned", kind="mine")
box(2, "Data", "Prediction log", "Every answer with its model version", kind="mine")

box(0, "Application", "Planning system", "Sends each engine's last 30 flights", pattern=("BATCH · WEEKLY", "batch"))
box(1, "Application", "RUL service", "POST /rul over the engine_rul package", kind="model", pattern=("ONLINE", "online"))
box(2, "Application", "Grafana dashboard", "p95 and regime share", kind="mine")

box(0, "Technology", "CAMO hosting", "Feed and planning system")
box(1, "Technology", "API container", "Model baked in at build", kind="mine")
box(2, "Technology", "Postgres, Prometheus", "Log volume and metrics scrape", kind="mine")

# Two flows in the application layer
ya = BANDS["Application"] + BAND_H / 2
for c in (0, 1):
    add(f'<line x1="{COLS[c] + CW + 1}" y1="{ya}" x2="{COLS[c + 1] - 2}" y2="{ya}" stroke="{MUTED}" stroke-width="1.4" marker-end="url(#head)"/>')

# Legend
ly = H - 26
add(f'<rect x="{LEFT}" y="{ly - 12}" width="28" height="16" rx="4" fill="#ffffff" stroke="{MINE}" stroke-width="1.3"/>')
text(LEFT + 36, ly, "built in Entry 1", 12, fill=MUTED)
add(f'<rect x="{LEFT + 150}" y="{ly - 12}" width="28" height="16" rx="4" fill="#ffffff" stroke="{LINE}" stroke-width="1.3" stroke-dasharray="5 4"/>')
text(LEFT + 186, ly, "existing estate", 12, fill=MUTED)
x = LEFT + 310
for label, key in [("ONLINE", "online"), ("BATCH · WEEKLY", "batch"), ("HUMAN-IN-THE-LOOP", "human")]:
    pw = 8 + 6.4 * len(label)
    pill(x + pw, ly - 12, label, key)
    x += pw + 8
text(x + 4, ly, "capability patterns (Session 1, §3.3)", 12, fill=MUTED)
add('</svg>')
open(sys.argv[1], "w").write("\n".join(out))
