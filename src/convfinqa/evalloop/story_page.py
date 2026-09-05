"""Render `story.json` as the published page. One file, no runtime dependencies.

Self-contained by design: the page is served by GitHub Pages from `docs/`, so
every chart is hand-authored inline SVG and every style is inline CSS. A page
that fetches a charting library at view time is a page that breaks quietly when
that CDN moves, years after the campaign it describes.

The page's argument, in order: what the loop is, what the rule is, what it did,
and — the part most write-ups omit — what it rejected. Rejections are most of
the record and they are what make the promotions credible.
"""

from __future__ import annotations

import html
import json
from typing import Any

AGENTS = ("triage", "preprocess", "retriever", "calculator")
AGENT_COLOR = {
    "triage": "#7aa2f7",
    "preprocess": "#bb9af7",
    "retriever": "#e0af68",
    "calculator": "#73daca",
}

CSS = """
:root{--bg:#0e1116;--panel:#151a21;--panel-2:#1b212a;--line:#262d38;--text:#dbe3ee;
--muted:#93a1b5;--faint:#6b7a90;--good:#73daca;--bad:#f7768e;--amber:#e0af68;--info:#7aa2f7;
--violet:#bb9af7;--mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);
font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,sans-serif}
.wrap{max-width:980px;margin:0 auto;padding:0 24px 96px}
header{padding:72px 0 40px;border-bottom:1px solid var(--line);margin-bottom:44px}
h1{font-size:38px;line-height:1.15;margin:0 0 14px;letter-spacing:-.02em}
h2{font-size:23px;margin:56px 0 14px;letter-spacing:-.01em}
h3{font-size:16px;margin:30px 0 10px}
p{margin:0 0 14px;color:var(--muted)}
.lede{font-size:17px;color:var(--text);max-width:66ch}
.eyebrow{font:600 11px/1 var(--mono);letter-spacing:.14em;text-transform:uppercase;
color:var(--faint);margin:0 0 10px}
code,.mono{font-family:var(--mono);font-size:.9em}
.grid{display:grid;gap:14px}
.g3{grid-template-columns:repeat(auto-fit,minmax(200px,1fr))}
.g2{grid-template-columns:repeat(auto-fit,minmax(280px,1fr))}
.card{background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:16px;min-width:0}
.stat{font:600 30px/1 var(--mono);letter-spacing:-.02em}
.stat.good{color:var(--good)}.stat.bad{color:var(--bad)}.stat.amber{color:var(--amber)}
.k{font:600 10px/1 var(--mono);letter-spacing:.12em;text-transform:uppercase;color:var(--faint);
margin-bottom:8px}
.sub{font-size:12.5px;color:var(--faint);margin-top:6px}
table{width:100%;border-collapse:collapse;font-size:13.5px}
th{text-align:left;font:600 10px/1 var(--mono);letter-spacing:.1em;text-transform:uppercase;
color:var(--faint);padding:0 10px 8px;border-bottom:1px solid var(--line);white-space:nowrap}
td{padding:9px 10px;border-bottom:1px solid var(--line);vertical-align:top}
tr:last-child td{border-bottom:0}
.num{font-family:var(--mono);text-align:right;white-space:nowrap}
.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:9px;background:var(--panel);
margin:0 0 18px}
.scroll table{min-width:640px}
.pill{display:inline-block;font:600 10px/1.7 var(--mono);letter-spacing:.06em;text-transform:uppercase;
padding:0 8px;border-radius:99px;border:1px solid}
.pill.ok{color:var(--good);border-color:#2c5f57;background:#132420}
.pill.no{color:var(--faint);border-color:var(--line);background:var(--panel-2)}
.pill.agent{color:var(--info);border-color:#2b3d5e;background:#141b28;text-transform:none}
figure{margin:22px 0 26px}
svg{display:block;width:100%;height:auto}
figcaption{font-size:12.5px;color:var(--faint);margin-top:10px}
.exp{border:1px solid var(--line);border-radius:9px;background:var(--panel);margin:0 0 12px;
overflow:hidden}
.exp>summary{cursor:pointer;padding:13px 16px;display:flex;gap:12px;align-items:center;
flex-wrap:wrap;list-style:none}
.exp>summary::-webkit-details-marker{display:none}
.exp[open]>summary{border-bottom:1px solid var(--line);background:var(--panel-2)}
.exp .body{padding:16px}
.exp .label{font:600 12px/1 var(--mono);color:var(--text)}
.exp .head-delta{margin-left:auto;font-family:var(--mono);font-size:12.5px}
pre{background:#0b0e13;border:1px solid var(--line);border-radius:7px;padding:13px;
overflow-x:auto;font-family:var(--mono);font-size:12px;line-height:1.5;margin:10px 0 0;color:var(--muted)}
pre .add{color:var(--good)}pre .del{color:var(--bad)}pre .hunk{color:var(--violet)}
.note{border-left:2px solid var(--amber);background:var(--panel);padding:12px 16px;
border-radius:0 7px 7px 0;margin:0 0 16px}
.note strong{color:var(--text)}
footer{margin-top:70px;padding-top:22px;border-top:1px solid var(--line);
font-size:12.5px;color:var(--faint)}
.legend{display:flex;gap:16px;flex-wrap:wrap;font:11px/1 var(--mono);color:var(--muted);margin-top:10px}
.legend i{display:inline-block;width:14px;height:2px;vertical-align:middle;margin-right:6px}
"""


def _e(text: Any) -> str:
    return html.escape(str(text if text is not None else ""))


def _pct(value: Any, digits: int = 1) -> str:
    if value is None:
        return "—"
    return f"{float(value) * 100:.{digits}f}%"


def _pp(value: Any) -> str:
    if value is None:
        return "—"
    return f"{float(value) * 100:+.2f}pp"


def harness_svg() -> str:
    """The loop, as one figure: what runs, on which split, and what decides."""
    # Sub-labels are kept short enough to sit inside a 128px box at 9.5px mono
    # — roughly 22 characters. Longer ones overflow the stroke and read as a
    # rendering bug rather than a caption.
    boxes = [
        (20, "train draw", "pool − gate, seeded"),
        (170, "run", "stop at first wrong"),
        (320, "diagnose", "attribution from gold"),
        (470, "rewrite", "ONE subagent, in full"),
        (620, "gate", "fixed split, both arms"),
        (770, "decide", "one-sided p < 0.05"),
    ]
    parts = [
        '<svg viewBox="0 0 940 200" role="img" aria-labelledby="harness-t">',
        '<title id="harness-t">The optimisation loop: draw, run, diagnose, '
        "rewrite, gate, decide</title>",
        '<defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
        'markerHeight="6" orient="auto-start-reverse">'
        '<path d="M0 0L10 5L0 10z" fill="#6b7a90"/></marker></defs>',
        '<g font-family="IBM Plex Mono, monospace" font-size="11">',
    ]
    for i, (x, title, sub) in enumerate(boxes):
        stroke = "#e0af68" if title in {"gate", "decide"} else "#7aa2f7"
        fill = "#1b212a" if title in {"gate", "decide"} else "#151a21"
        parts.append(
            f'<rect x="{x}" y="52" width="128" height="66" rx="8" fill="{fill}" '
            f'stroke="{stroke}"/>'
            f'<text x="{x + 64}" y="80" text-anchor="middle" fill="#dbe3ee" '
            f'font-weight="600">{title}</text>'
            f'<text x="{x + 64}" y="99" text-anchor="middle" fill="#93a1b5" '
            f'font-size="9">{sub}</text>'
        )
        if i < len(boxes) - 1:
            parts.append(
                f'<line x1="{x + 128}" y1="85" x2="{boxes[i + 1][0] - 4}" y2="85" '
                'stroke="#6b7a90" stroke-width="1.2" marker-end="url(#ar)"/>'
            )
    parts.append(
        '<path d="M884 118 L884 158 L84 158 L84 122" fill="none" stroke="#6b7a90" '
        'stroke-width="1.2" stroke-dasharray="4 4" marker-end="url(#ar)"/>'
        '<text x="484" y="174" text-anchor="middle" fill="#93a1b5" font-size="10">'
        "promoted or not, the verdict is recorded — and the next rewrite reads it"
        "</text>"
    )
    parts.append(
        '<text x="20" y="32" fill="#6b7a90" font-size="10">TRAIN — resampled every '
        "cycle, teacher reads it</text>"
        '<text x="620" y="32" fill="#e0af68" font-size="10">GATE — fixed, never '
        "tuned against</text>"
    )
    parts.append("</g></svg>")
    return "".join(parts)


def track_chart(track: list[dict[str, Any]]) -> str:
    """Overall accuracy and the four per-agent metrics across champion moves."""
    points = [p for p in track if p.get("accuracy") is not None]
    if len(points) < 2:
        return ""
    w, h = 940, 320
    left, right, top, bottom = 64, 150, 28, 52
    series: dict[str, list[float | None]] = {
        "overall": [p["accuracy"] for p in points],
        **{a: [(p["panel"] or {}).get(a) for p in points] for a in AGENTS},
    }
    values = [v for row in series.values() for v in row if v is not None]
    lo, hi = min(values), max(values)
    pad = max(0.02, (hi - lo) * 0.25)
    lo, hi = max(0.0, lo - pad), min(1.0, hi + pad)

    def x_of(i: int) -> float:
        span = max(1, len(points) - 1)
        return left + i * (w - left - right) / span

    def y_of(v: float) -> float:
        return top + (hi - v) / (hi - lo) * (h - top - bottom)

    parts = [
        f'<svg viewBox="0 0 {w} {h}" role="img" aria-labelledby="track-t">',
        '<title id="track-t">Overall accuracy and each subagent\'s metric at every '
        "champion move</title>",
        '<g font-family="IBM Plex Mono, monospace" font-size="10">',
    ]
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        v = lo + frac * (hi - lo)
        y = y_of(v)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{w - right}" y2="{y:.1f}" '
            'stroke="#262d38" stroke-width="1"/>'
            f'<text x="{left - 10}" y="{y + 3:.1f}" text-anchor="end" fill="#6b7a90">'
            f"{v * 100:.0f}%</text>"
        )
    for i, p in enumerate(points):
        parts.append(
            f'<text x="{x_of(i):.1f}" y="{h - 30}" text-anchor="middle" '
            f'fill="#dbe3ee">{_e(p["version"])}</text>'
        )
        if p.get("target_agent"):
            parts.append(
                f'<text x="{x_of(i):.1f}" y="{h - 16}" text-anchor="middle" '
                f'fill="{AGENT_COLOR.get(p["target_agent"], "#6b7a90")}" '
                f'font-size="9">↑ {_e(p["target_agent"])}</text>'
            )
    labels: list[tuple[float, str, str]] = []
    for name, row in series.items():
        colour = "#dbe3ee" if name == "overall" else AGENT_COLOR[name]
        width = 2.4 if name == "overall" else 1.4
        drawn: list[tuple[int, float]] = [
            (i, float(value)) for i, value in enumerate(row) if value is not None
        ]
        if len(drawn) < 2:
            continue
        segment = " ".join(f"{x_of(i):.1f},{y_of(value):.1f}" for i, value in drawn)
        parts.append(
            f'<polyline points="{segment}" fill="none" stroke="{colour}" '
            f'stroke-width="{width}" stroke-linejoin="round"/>'
        )
        for i, value in drawn:
            parts.append(
                f'<circle cx="{x_of(i):.1f}" cy="{y_of(value):.1f}" r="3" '
                f'fill="{colour}"/>'
            )
        labels.append((y_of(drawn[-1][1]), colour, name))
    # Two series can end within a few tenths of a point of each other, and their
    # labels then overprint into an unreadable smudge. Nudge them apart from the
    # top down, keeping each label's colour tied to its line.
    labels.sort()
    min_gap = 12.0
    for i in range(1, len(labels)):
        y, colour, name = labels[i]
        prev_y = labels[i - 1][0]
        if y - prev_y < min_gap:
            labels[i] = (prev_y + min_gap, colour, name)
    for y, colour, name in labels:
        parts.append(
            f'<text x="{w - right + 10}" y="{y + 3:.1f}" fill="{colour}">'
            f"{_e(name)}</text>"
        )
    parts.append("</g></svg>")
    return "".join(parts)


def _diff_html(diff: str, limit: int = 90) -> str:
    lines = diff.splitlines()
    shown, rest = lines[:limit], len(lines) - limit
    out = []
    for line in shown:
        cls = ""
        if line.startswith("+") and not line.startswith("+++"):
            cls = "add"
        elif line.startswith("-") and not line.startswith("---"):
            cls = "del"
        elif line.startswith("@@"):
            cls = "hunk"
        out.append(f'<span class="{cls}">{_e(line)}</span>' if cls else _e(line))
    if rest > 0:
        out.append(f"… {rest} more diff lines")
    return "<pre>" + "\n".join(out) + "</pre>"


def _experiment(exp: dict[str, Any]) -> str:
    promoted = exp["promoted"]
    pill = (
        '<span class="pill ok">promoted</span>'
        if promoted
        else '<span class="pill no">rejected</span>'
    )
    p = exp.get("cluster_p_one_sided")
    ci = exp.get("delta_ci") or [None, None]
    rows = "".join(
        f"<tr><td>{_e(a)}</td>"
        f'<td class="num">{_pct((exp["panel_baseline"] or {}).get(a))}</td>'
        f'<td class="num">{_pct((exp["panel_candidate"] or {}).get(a))}</td></tr>'
        for a in AGENTS
    )
    chars = exp.get("prompt_chars") or {}
    char_note = ""
    if chars.get("before") and chars.get("after"):
        char_note = (
            f'<p class="sub">prompt {int(chars["before"])} → {int(chars["after"])} '
            "characters</p>"
        )
    # Every interpolated value is bound to a local first. Nesting quotes inside
    # an f-string is a 3.12 feature, and `ruff format` will happily rewrite a
    # valid single-quoted inner string into an invalid double-quoted one — so
    # the expressions stay out of the template.
    rationale = str(exp.get("rationale") or "")
    rationale_html = f"<p>{_e(rationale)}</p>" if rationale else ""
    diff_text = str(exp.get("diff") or "")
    diff_html = (
        f"<h3>The prompt change</h3>{_diff_html(diff_text)}" if diff_text else ""
    )
    delta = exp.get("accuracy_delta") or 0
    delta_tone = "good" if delta > 0 else "bad"
    p_tone = "good" if p is not None and float(p) < 0.05 else "amber"
    p_text = "—" if p is None else f"{float(p):.3f}"
    heading = _e(exp["label"] or exp["candidate_version"])
    agent = _e(exp["target_agent"])
    baseline = _e(exp["baseline_version"])
    candidate = _e(exp["candidate_version"])
    summary = _e(exp.get("summary_of_changes") or "no summary recorded")
    n_fixed = int(exp.get("fixed") or 0)
    n_broken = int(exp.get("broken") or 0)
    n_compared = int(exp.get("n_compared") or 0)
    return f"""<details class="exp">
<summary><span class="label">{heading}</span>
{pill}<span class="pill agent">{agent}</span>
<span class="head-delta">{_pp(delta)}&nbsp;·&nbsp;p={p_text}</span></summary>
<div class="body">
<p><strong>{baseline} → {candidate}</strong> — {summary}</p>
{rationale_html}
<div class="grid g3">
<div class="card"><div class="k">paired delta</div>
<div class="stat {delta_tone}">{_pp(delta)}</div>
<div class="sub">95% CI [{_pp(ci[0])}, {_pp(ci[1])}]</div></div>
<div class="card"><div class="k">one-sided clustered p</div>
<div class="stat {p_tone}">{p_text}</div>
<div class="sub">α = 0.05</div></div>
<div class="card"><div class="k">flips</div>
<div class="stat">{n_fixed}&thinsp;/&thinsp;{n_broken}</div>
<div class="sub">fixed / broken of {n_compared}</div></div>
</div>
{char_note}
<h3>Per-agent panel on the gate split</h3>
<div class="scroll"><table><thead><tr><th>subagent</th><th class="num">before</th>
<th class="num">after</th></tr></thead><tbody>{rows}</tbody></table></div>
{diff_html}
</div></details>"""


def _lineage_row(ev: dict[str, Any]) -> str:
    when = _e(str(ev.get("at") or "")[:19])
    previous = _e(ev.get("previous") or ev.get("previous_champion") or "—")
    version = _e(ev.get("version"))
    reason = _e(str(ev.get("reason") or "")[:180])
    return (
        f'<tr><td class="mono">{when}</td>'
        f'<td class="mono">{previous} → <strong>{version}</strong></td>'
        f"<td>{reason}</td></tr>"
    )


def render_page(data: dict[str, Any]) -> str:
    """The whole page."""
    campaigns = data.get("campaigns") or []
    experiments = [e for c in campaigns for e in c["experiments"]]
    promoted = [e for e in experiments if e["promoted"]]
    track = data.get("champion_track") or []
    split = data.get("split") or {}
    first_acc = track[0]["accuracy"] if track else None
    last_acc = track[-1]["accuracy"] if track else None
    if first_acc is not None and last_acc is not None:
        moved = f"gate accuracy {_pct(first_acc)} → {_pct(last_acc)}"
    elif data.get("champion_accuracy") is not None:
        moved = (
            f"{_pct(data['champion_accuracy'])} on the gate split — "
            "not yet moved by an experiment"
        )
    else:
        moved = "no gate run recorded"

    chart = track_chart(track)
    campaign_sections = "".join(
        f"<h3>{_e(c['name'])} — {len(c['experiments'])} experiments, "
        f"{sum(1 for e in c['experiments'] if e['promoted'])} promoted</h3>"
        + "".join(_experiment(e) for e in c["experiments"])
        for c in campaigns
    )
    # `.get` throughout: this renders a committed JSON file that an older
    # version of the collector may have written, and a missing key should cost
    # a dash in one cell rather than the whole page.
    lineage_rows = "".join(
        _lineage_row(ev) for ev in reversed(data.get("lineage") or [])
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Optimising a multi-agent financial QA system</title>
<meta name="description" content="A prompt-optimisation loop for a four-agent
ConvFinQA pipeline: gold-derived attribution, one subagent per experiment, and a
significance gate that rejects most of what it is offered.">
<style>{CSS}</style></head><body><div class="wrap">
<header>
<p class="eyebrow">ConvFinQA · agent optimisation</p>
<h1>Optimising a multi-agent system,<br>one subagent at a time</h1>
<p class="lede">A four-stage financial question-answering pipeline that improves
itself: a teacher reads its failures, rewrites exactly one subagent's prompt, and
a paired significance test on a fixed held-back split decides whether that
rewrite becomes the champion. Every number here is read out of the tracking
store that recorded the runs — nothing on this page is typed by hand.</p>
</header>

<div class="grid g3">
<div class="card"><div class="k">champion</div>
<div class="stat">{_e(data.get("champion") or "—")}</div>
<div class="sub">{moved}</div></div>
<div class="card"><div class="k">experiments run</div>
<div class="stat">{len(experiments)}</div>
<div class="sub">{len(promoted)} promoted · {
        len(experiments) - len(promoted)
    } rejected</div></div>
<div class="card"><div class="k">gate split</div>
<div class="stat">{split.get("gate_questions", "—")}</div>
<div class="sub">questions across {split.get("gate_reports", "—")} conversations,
fixed for the campaign</div></div>
</div>

<h2>How the loop works</h2>
<p>One experiment changes <strong>one subagent's prompt</strong> and nothing else.
That constraint is what makes a result readable: when the champion moves, the diff
between it and the version before it is a single prompt, so the improvement has a
named cause rather than an assertion attached to it.</p>
<figure>{harness_svg()}
<figcaption>Train is resampled from the pool every cycle and read only by the
teacher. The gate split never moves and is never tuned against. Both arms of every
comparison run every question on it — the early-stopping that saves a quarter of
the train pass is refused here, because a paired test needs a counterpart for each
question.</figcaption></figure>

<div class="note"><strong>The promotion rule.</strong>
<p>{_e(data.get("rule"))}. One-sided because the gate only ever promotes
improvements, so spending half the rejection region on a direction it will never
act in buys nothing. Cluster-corrected because flips are not independent — a
conversation's turns share a report, a history and usually an error, so four fixed
turns in one report are one piece of evidence, not four.</p></div>

{
        f"<h2>What moved, and which subagent moved it</h2><figure>{chart}<figcaption>Overall gate accuracy in white; each subagent&rsquo;s own gold-derived metric in colour. The arrow under a version names the single subagent that experiment rewrote.</figcaption></figure>"
        if chart
        else ""
    }

<h2>Every experiment, including the ones that failed</h2>
<p>Most challengers are rejected, and that is the system working. At this split
size a genuine improvement of two or three points is not distinguishable from
noise, so the gate refuses it — and the rejection is recorded, fed back to the
prompt writer, and counts against the campaign's budget.</p>
{campaign_sections or "<p>No campaigns recorded yet.</p>"}

{
        f'<h2>Champion lineage</h2><div class="scroll"><table><thead><tr><th>when</th><th>move</th><th>why</th></tr></thead><tbody>{lineage_rows}</tbody></table></div>'
        if lineage_rows
        else ""
    }

<footer>Generated {_e(data.get("generated_at"))} from
<code>evaluation/registry.json</code> and the MLflow tracking store ·
split manifest <code>{_e(split.get("name") or "—")}</code> ·
rebuild with <code>convfinqa-evalloop story</code>
<script type="application/json" id="story-data">{
        json.dumps(
            {
                "champion": data.get("champion"),
                "n_experiments": len(experiments),
                "generated_at": data.get("generated_at"),
            }
        )
    }</script>
</footer>
</div></body></html>"""
