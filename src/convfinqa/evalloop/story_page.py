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
a{color:var(--info);text-decoration:none}a:hover{text-decoration:underline}
.arms{display:grid;gap:14px;grid-template-columns:repeat(auto-fit,minmax(260px,1fr))}
.arm h3{margin:0 0 6px}
.pill.cls{color:var(--violet);border-color:#43386b;background:#191627;text-transform:none}
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
<p class="eyebrow">ConvFinQA · agent optimisation · <a href="agent-sdk.html">the Agent SDK experiment →</a></p>
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


# ── The Agent SDK experiment page ─────────────────────────────────────────


def _arm_card(title: str, arm: dict[str, Any] | None, blurb: str) -> str:
    arm = arm or {}
    version = arm.get("version")
    if not version:
        return f"""<div class="card arm"><div class="k">{_e(title)}</div>
<div class="stat">—</div><div class="sub">not yet run</div><p class="sub">{_e(blurb)}</p></div>"""
    panel = arm.get("panel") or {}
    rows = "".join(
        f'<tr><td>{_e(a)}</td><td class="num">{_pct(panel.get(a))}</td></tr>'
        for a in AGENTS
    )
    cost = arm.get("cost")
    wall = arm.get("wall")
    cost_text = "—" if cost is None else f"${float(cost):.2f}"
    wall_text = "—" if wall is None else f"{float(wall) / 60:.0f} min"
    by_type = arm.get("by_turn_type") or {}
    type_rows = "".join(
        f'<tr><td>{_e(label)}</td><td class="num">{_pct(by_type.get(key))}</td></tr>'
        for key, label in (("number", "number turns"), ("program", "program turns"))
    )
    return f"""<div class="card arm"><div class="k">{_e(title)}</div>
<div class="stat">{_pct(arm.get("accuracy"))}</div>
<div class="sub"><span class="mono">{_e(version)}</span> · {_e(arm.get("run_name") or "")}</div>
<p class="sub">{_e(blurb)}</p>
<table><tbody>{type_rows}{rows}
<tr><td>cost</td><td class="num">{cost_text}</td></tr>
<tr><td>wall</td><td class="num">{wall_text}</td></tr></tbody></table></div>"""


def _turn_type_verdict(gate: dict[str, Any]) -> str:
    """The paired verdict split by turn type — where the difference actually is.

    Rendered as its own block rather than folded into the headline: the aggregate
    delta is an average over two populations that behave nothing alike, and a
    reader who sees only the average will attribute the gain to both.
    """
    by_type = gate.get("by_turn_type") or {}
    if not by_type:
        return ""
    order = [("program", "program"), ("number", "number")]
    rows = []
    for key, label in order:
        row = by_type.get(key)
        if not row:
            continue
        delta = float(row.get("delta_pp") or 0.0)
        p = row.get("cluster_p_one_sided")
        p_text = "—" if p is None else f"{float(p):.4g}"
        verdict = (
            '<span class="pill ok">significant</span>'
            if p is not None and float(p) < 0.05
            else '<span class="pill no">no effect</span>'
        )
        rows.append(
            f"<tr><td>{_e(label)}</td>"
            f'<td class="num">{row.get("n")}</td>'
            f'<td class="num">{_pct(row.get("baseline_accuracy"))}</td>'
            f'<td class="num">{_pct(row.get("candidate_accuracy"))}</td>'
            f'<td class="num {"good" if delta > 0 else ("bad" if delta < 0 else "")}">'
            f"{delta:+.2f}pp</td>"
            f'<td class="num">{row.get("fixed")}&thinsp;/&thinsp;{row.get("broken")}</td>'
            f'<td class="num">{p_text}</td>'
            f"<td>{verdict}</td></tr>"
        )
    if not rows:
        return ""
    return f"""<h2>Where the difference is: number turns vs program turns</h2>
<p class="sub">The dataset splits turns into a lookup (<em>number</em>) and a
computation (<em>program</em>). Both arms have saturated the lookup, so the
aggregate delta is carried entirely by the reasoning turns — reported separately
because an average over the two describes neither.</p>
<div class="scroll"><table>
<thead><tr><th>turn type</th><th class="num">n</th><th class="num">pipeline</th>
<th class="num">sdk</th><th class="num">delta</th><th class="num">fixed / broken</th>
<th class="num">one-sided clustered p</th><th>verdict</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table></div>"""


def _sdk_experiment(exp: dict[str, Any]) -> str:
    """One SDK experiment: like `_experiment`, with the class and its edits."""
    promoted = exp.get("promoted")
    pill = (
        '<span class="pill ok">promoted</span>'
        if promoted
        else '<span class="pill no">rejected</span>'
    )
    p = exp.get("cluster_p_one_sided")
    ci = exp.get("delta_ci") or [None, None]
    rows = "".join(
        f"<tr><td>{_e(a)}</td>"
        f'<td class="num">{_pct((exp.get("panel_baseline") or {}).get(a))}</td>'
        f'<td class="num">{_pct((exp.get("panel_candidate") or {}).get(a))}</td></tr>'
        for a in AGENTS
    )
    edits = exp.get("edits") or []
    edit_rows = "".join(
        f'<tr><td class="mono">{_e(ed.get("failure_class") or ed.get("target"))}</td>'
        f"<td>{_e(ed.get('change_kind'))}</td>"
        f'<td class="num">{_e(ed.get("n_diagnoses") if ed.get("n_diagnoses") is not None else "—")}</td>'
        f"<td>{_e(str(ed.get('rationale') or '')[:220])}</td></tr>"
        for ed in edits
    )
    edits_html = (
        '<h3>Edits in this rewrite</h3><div class="scroll"><table><thead><tr>'
        '<th>failure class</th><th>kind</th><th class="num">cases</th><th>why</th>'
        f"</tr></thead><tbody>{edit_rows}</tbody></table></div>"
        if edit_rows
        else ""
    )
    diff_text = str(exp.get("diff") or "")
    diff_html = (
        f"<h3>The prompt change</h3>{_diff_html(diff_text)}" if diff_text else ""
    )
    delta = exp.get("accuracy_delta") or 0
    delta_tone = "good" if delta > 0 else "bad"
    p_tone = "good" if p is not None and float(p) < 0.05 else "amber"
    p_text = "—" if p is None else f"{float(p):.3f}"
    heading = _e(exp.get("label") or exp.get("candidate_version"))
    target_class = _e(exp.get("target_class") or exp.get("target_agent") or "—")
    baseline = _e(exp.get("baseline_version"))
    candidate = _e(exp.get("candidate_version"))
    summary = _e(exp.get("summary_of_changes") or "no summary recorded")
    n_fixed = int(exp.get("fixed") or 0)
    n_broken = int(exp.get("broken") or 0)
    n_compared = int(exp.get("n_compared") or 0)
    return f"""<details class="exp">
<summary><span class="label">{heading}</span>
{pill}<span class="pill cls">{target_class}</span>
<span class="head-delta">{_pp(delta)}&nbsp;·&nbsp;p={p_text}</span></summary>
<div class="body">
<p><strong>{baseline} → {candidate}</strong> — {summary}</p>
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
{edits_html}
<h3>Per-stage panel on the gate split</h3>
<div class="scroll"><table><thead><tr><th>stage</th><th class="num">before</th>
<th class="num">after</th></tr></thead><tbody>{rows}</tbody></table></div>
{diff_html}
</div></details>"""


def render_sdk_page(data: dict[str, Any]) -> str:
    """The Agent SDK experiment: one prompt, one session, the same gate.

    Renders from the same `story.json` as the campaign page. Everything that
    has not happened yet renders as "not yet run" — never as a zero, because a
    zero on this page would be read as a result.
    """
    comparison = data.get("runtime_comparison") or {}
    pipeline = comparison.get("pipeline") or {}
    sdk = comparison.get("agent_sdk") or {}
    gate = comparison.get("gate") or {}
    sdk_campaigns = data.get("sdk_campaigns") or []
    experiments = [e for c in sdk_campaigns for e in c.get("experiments", [])]
    promoted = [e for e in experiments if e.get("promoted")]
    split = data.get("split") or {}

    if gate.get("delta_pp") is not None:
        d = float(gate["delta_pp"])
        p = gate.get("p_value")
        ci = gate.get("ci") or [None, None]
        verdict_html = f"""<div class="grid g3">
<div class="card"><div class="k">paired delta (sdk − pipeline)</div>
<div class="stat {"good" if d > 0 else "bad"}">{d:+.2f}pp</div>
<div class="sub">95% CI [{_pp(ci[0])}, {_pp(ci[1])}]</div></div>
<div class="card"><div class="k">one-sided clustered p</div>
<div class="stat {"good" if p is not None and float(p) < 0.05 else "amber"}">{"—" if p is None else f"{float(p):.3f}"}</div>
<div class="sub">α = 0.05</div></div>
<div class="card"><div class="k">flips</div>
<div class="stat">{gate.get("fixed") if gate.get("fixed") is not None else "—"}&thinsp;/&thinsp;{gate.get("broken") if gate.get("broken") is not None else "—"}</div>
<div class="sub">fixed / broken · {_e(gate.get("candidate_version") or "")} vs {_e(pipeline.get("version") or data.get("champion") or "—")}</div></div>
</div>"""
    else:
        verdict_html = (
            '<div class="note"><strong>No cross-runtime gate yet.</strong>'
            "<p>The comparison appears here once an <code>sdk_vN</code> run on "
            "the gate split has been gated against the pipeline champion — a "
            "paired, one-sided, cluster-corrected McNemar test, the same rule "
            "the campaigns use.</p></div>"
        )

    campaign_sections = "".join(
        f"<h3>{_e(c['name'])} — {len(c.get('experiments', []))} experiments, "
        f"{sum(1 for e in c.get('experiments', []) if e.get('promoted'))} promoted</h3>"
        + "".join(_sdk_experiment(e) for e in c.get("experiments", []))
        for c in sdk_campaigns
    )
    sdk_champion = data.get("sdk_champion")

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>The Agent SDK experiment — one session against four agents</title>
<meta name="description" content="A single Claude Agent SDK session with calculator
tools, run through the same eval loop and the same significance gate as the
four-agent ConvFinQA pipeline.">
<style>{CSS}</style></head><body><div class="wrap">
<header>
<p class="eyebrow"><a href="index.html">← the campaign write-up</a> · ConvFinQA · Agent SDK experiment</p>
<h1>One session, one prompt,<br>the same gate</h1>
<p class="lede">The pipeline answers a conversation with four prompted agents in
sequence. This experiment gives one Claude Agent SDK session the calculator
tools and one system prompt, distilled from the pipeline's four, and runs it
through the same loop: a fresh train draw, a diagnosis agent that files each
first-wrong case under a failure class, a teacher that edits the one prompt —
several tagged areas at once, until two rejections in a row drop it to one — and
the same paired significance test on the same fixed gate split. Every number is
read out of the record; nothing here is typed by hand.</p>
</header>

<div class="grid g3">
<div class="card"><div class="k">sdk champion</div>
<div class="stat">{_e(sdk_champion or "—")}</div>
<div class="sub">alias <code>sdk_champion</code> — never <code>champion</code></div></div>
<div class="card"><div class="k">sdk experiments</div>
<div class="stat">{len(experiments)}</div>
<div class="sub">{len(promoted)} promoted · {
        len(experiments) - len(promoted)
    } rejected</div></div>
<div class="card"><div class="k">gate split</div>
<div class="stat">{split.get("gate_questions", "—")}</div>
<div class="sub">questions across {split.get("gate_reports", "—")} conversations —
the same split the pipeline campaigns gate on</div></div>
</div>

<h2>The two arms on the gate split</h2>
<div class="arms">
{
        _arm_card(
            "pipeline · four agents",
            pipeline,
            "triage → preprocess → retriever → calculator, four prompts, DeepSeek.",
        )
    }
{
        _arm_card(
            "agent_sdk · one session",
            sdk,
            "one Claude session per conversation, calculator tools only, one prompt; stages reported by the agent so the same panel applies.",
        )
    }
</div>
<h3>The verdict between them</h3>
{verdict_html}

{_turn_type_verdict(gate)}

<h2>What is different, and what is not</h2>
<div class="note"><strong>Same loop, same rule.</strong>
<p>Draw, run, diagnose, rewrite, gate, decide — in that order, on the same
splits, with the same one-sided cluster-corrected McNemar at α = 0.05
(<code>{_e(data.get("rule"))}</code>). Promotion evidence comes only from the
gate split, and it moves <code>sdk_champion</code> — the pipeline's
<code>champion</code> is never touched by this arm.</p></div>
<div class="note"><strong>The unit of change is the prompt, with tagged edits inside it.</strong>
<p>The pipeline rewrites exactly one subagent per experiment so a champion move
has a named cause. There is one prompt here, so the teacher may edit several
areas per cycle — one edit per failure class it addresses, each tagged with the
class and the case ids behind it. The cost is attribution: after a gate, each
edit is read against the flips in its own class. Two rejections in a row switch
the lineage to one area per cycle for the rest of the campaign.</p></div>
<div class="note"><strong>Skipped stages are failures of that stage.</strong>
<p>A program turn the session answered without a plan attributes to preprocess;
one it answered without a calculator call attributes to calculator. Arithmetic
happens only inside the tools, so the calculator column means the same thing in
both arms.</p></div>

<h2>Every SDK experiment, including the ones that failed</h2>
{
        campaign_sections
        or "<p>No SDK campaign recorded yet. The first cycle is <code>convfinqa-evalloop cycle --campaign s01 --runtime agent_sdk</code>.</p>"
    }

<footer>Generated {_e(data.get("generated_at"))} from
<code>evaluation/registry.json</code>, the three ledgers under
<code>evaluation/diagnostics/evalloop/</code> and the MLflow tracking store ·
rebuild with <code>convfinqa-evalloop story</code>
<script type="application/json" id="story-data">{
        json.dumps(
            {
                "champion": data.get("champion"),
                "sdk_champion": sdk_champion,
                "n_sdk_experiments": len(experiments),
                "generated_at": data.get("generated_at"),
            }
        )
    }</script>
</footer>
</div></body></html>"""
