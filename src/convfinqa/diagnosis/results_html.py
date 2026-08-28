"""Render diagnostic_results_<variant>.html — dark-theme clone of pydantic predictions."""

from __future__ import annotations

import html as _html
from pathlib import Path
from typing import Any

import pandas as pd

from convfinqa.reporting import render_cell, render_page, viewer_panel_html

_JSON_COLUMNS = {
    "triage_io",
    "preprocess_io",
    "retriever_io",
    "calculator_io",
    "pred_sub_questions",
    "supporting_evidence",
    "harness_turn_results",
    "harness_triage_io",
    "harness_preprocess_io",
    "harness_retriever_io",
    "harness_calculator_io",
}
_LONG_TEXT_COLUMNS = {
    "history_text",
    "question",
    "system_prompt_fix",
    "failure_explanation",
}

_FILTER_SCRIPT = """
function applyFilters() {
  const onlyResolved = document.getElementById('only-resolved').checked;
  const onlyUnresolved = document.getElementById('only-unresolved').checked;
  const fa = document.getElementById('fa-filter').value;
  const att = document.getElementById('att-filter').value;
  const search = document.getElementById('search').value.toLowerCase();
  document.querySelectorAll('tbody tr').forEach(tr => {
    let show = true;
    if (onlyResolved && tr.dataset.resolved !== 'true') show = false;
    if (onlyUnresolved && tr.dataset.resolved === 'true') show = false;
    if (fa && tr.dataset.fa !== fa) show = false;
    if (att && tr.dataset.att !== att) show = false;
    if (search && !tr.textContent.toLowerCase().includes(search)) show = false;
    tr.classList.toggle('hidden', !show);
  });
}
function attach() {
  ['only-resolved', 'only-unresolved', 'fa-filter', 'att-filter', 'search'].forEach(id => {
    const el = document.getElementById(id);
    if (!el) return;
    el.addEventListener('input', applyFilters);
    el.addEventListener('change', applyFilters);
  });
  attachViewer();
}
document.addEventListener('DOMContentLoaded', attach);
"""


def _fmt_acc(n_correct: int, n_all: int) -> str:
    if n_all <= 0:
        return '<span class="placeholder">—</span>'
    return f"{n_correct / n_all:.1%}"


def _pivot_by_agent(df: pd.DataFrame) -> str:
    """Counts per failed_agent: all rows / harness_correct / accuracy."""
    if "failed_agent" not in df.columns:
        return ""
    correct_mask = (
        df["harness_correct"].astype(str).str.lower().isin({"true", "1"})
        if "harness_correct" in df.columns
        else pd.Series([False] * len(df))
    )
    agents = sorted(a for a in df["failed_agent"].fillna("").unique() if a != "")
    rows: list[str] = []
    total_all = 0
    total_correct = 0
    for agent in agents:
        m = df["failed_agent"] == agent
        n_all = int(m.sum())
        n_correct = int((m & correct_mask).sum())
        total_all += n_all
        total_correct += n_correct
        rows.append(
            f"<tr><td>{_html.escape(agent)}</td>"
            f'<td class="num">{n_all}</td>'
            f'<td class="num">{n_correct}</td>'
            f'<td class="num">{_fmt_acc(n_correct, n_all)}</td></tr>'
        )
    rows.append(
        f'<tr class="total"><td>TOTAL</td>'
        f'<td class="num">{total_all}</td>'
        f'<td class="num">{total_correct}</td>'
        f'<td class="num">{_fmt_acc(total_correct, total_all)}</td></tr>'
    )
    body = "\n".join(rows)
    return (
        '<div class="pivot">'
        "<h2>by failed_agent</h2>"
        "<table><thead><tr>"
        "<th>failed_agent</th><th>all</th><th>harness_correct</th><th>accuracy</th>"
        "</tr></thead>"
        f"<tbody>{body}</tbody></table></div>"
    )


def _pivot_by_agent_mode(df: pd.DataFrame) -> str:
    """Counts per (failed_agent, failure_mode): all / harness_correct / accuracy."""
    if not {"failed_agent", "failure_mode"}.issubset(df.columns):
        return ""
    correct_mask = (
        df["harness_correct"].astype(str).str.lower().isin({"true", "1"})
        if "harness_correct" in df.columns
        else pd.Series([False] * len(df))
    )
    rows: list[str] = []
    total_all = 0
    total_correct = 0
    agents = sorted(a for a in df["failed_agent"].fillna("").unique() if a != "")
    for agent in agents:
        agent_mask = df["failed_agent"] == agent
        modes = sorted(
            m for m in df.loc[agent_mask, "failure_mode"].fillna("").unique() if m != ""
        )
        agent_all = 0
        agent_correct = 0
        agent_rows: list[str] = []
        for mode in modes:
            m = agent_mask & (df["failure_mode"] == mode)
            n_all = int(m.sum())
            n_correct = int((m & correct_mask).sum())
            agent_all += n_all
            agent_correct += n_correct
            agent_rows.append(
                f"<tr><td>{_html.escape(agent)}</td>"
                f"<td>{_html.escape(mode)}</td>"
                f'<td class="num">{n_all}</td>'
                f'<td class="num">{n_correct}</td>'
                f'<td class="num">{_fmt_acc(n_correct, n_all)}</td></tr>'
            )
        total_all += agent_all
        total_correct += agent_correct
        rows.extend(agent_rows)
        rows.append(
            f'<tr class="subtotal"><td>{_html.escape(agent)}</td>'
            f"<td><em>subtotal</em></td>"
            f'<td class="num">{agent_all}</td>'
            f'<td class="num">{agent_correct}</td>'
            f'<td class="num">{_fmt_acc(agent_correct, agent_all)}</td></tr>'
        )
    rows.append(
        f'<tr class="total"><td colspan="2">TOTAL</td>'
        f'<td class="num">{total_all}</td>'
        f'<td class="num">{total_correct}</td>'
        f'<td class="num">{_fmt_acc(total_correct, total_all)}</td></tr>'
    )
    body = "\n".join(rows)
    return (
        '<div class="pivot">'
        "<h2>by failed_agent × failure_mode</h2>"
        "<table><thead><tr>"
        "<th>failed_agent</th><th>failure_mode</th>"
        "<th>all</th><th>harness_correct</th><th>accuracy</th>"
        "</tr></thead>"
        f"<tbody>{body}</tbody></table></div>"
    )


def _fix_box(s: str) -> str:
    return f'<div class="fix-box">{_html.escape(s)}</div>'


def _cell(col: str, value: Any, *, row_id: str) -> str:
    return render_cell(
        col,
        value,
        row_id=row_id,
        json_columns=_JSON_COLUMNS,
        long_text_columns=_LONG_TEXT_COLUMNS,
        empty_placeholder=True,
        special={"system_prompt_fix": _fix_box},
    )


def write_diagnostic_html(
    csv_path: Path,
    *,
    output_path: Path | None = None,
    title: str | None = None,
    prompts_version: str | None = None,
    variant: str | None = None,
) -> Path:
    """Render the s7 diagnostic CSV as the sticky-inspector HTML report."""
    from convfinqa.config import settings as _s

    if title is None:
        title = f"diagnostic_results_{variant or _s.variant}"
    # Resolved fields for the subtitle under the h1.
    _variant = variant or _s.variant
    _prompts_version = prompts_version or _s.prompts_version or "v2"
    df = pd.read_csv(csv_path).fillna("")
    out_path = output_path or csv_path.with_suffix(".html")
    columns = list(df.columns)

    n_rows = len(df)
    n_resolved = int(
        df["resolved"].astype(str).str.lower().isin({"true", "1"}).sum()
        if "resolved" in df.columns
        else 0
    )
    n_correct = int(
        df["harness_correct"].astype(str).str.lower().isin({"true", "1"}).sum()
        if "harness_correct" in df.columns
        else 0
    )
    n_unique_cases = (
        df[["report_id", "turn_index"]].drop_duplicates().shape[0]
        if {"report_id", "turn_index"}.issubset(df.columns)
        else 0
    )

    body_rows: list[str] = []
    fa_values: set[str] = set()
    att_values: set[str] = set()
    for _, row in df.iterrows():
        verify = str(row.get("verify_result", "")).strip().lower()
        if verify == "passed":
            row_class = "row-correct"
        elif verify == "failed":
            row_class = "row-wrong"
        else:
            row_class = "row-pending"
        fa = _html.escape(str(row.get("failed_agent", "")))
        att = _html.escape(str(row.get("attempt_id", "")))
        resolved = str(row.get("resolved", "")).lower() in {"true", "1"}
        fa_values.add(str(row.get("failed_agent", "")))
        att_values.add(str(row.get("attempt_id", "")))
        row_id = (
            f"{row.get('report_id', '')}#turn={row.get('turn_index', '')}"
            f"#att={row.get('attempt_id', '')}"
        )
        cells = "".join(f"<td>{_cell(c, row[c], row_id=row_id)}</td>" for c in columns)
        body_rows.append(
            f'<tr class="{row_class}" data-fa="{fa}" data-att="{att}" '
            f'data-resolved="{str(resolved).lower()}">{cells}</tr>'
        )

    head_cells = "".join(f"<th>{_html.escape(c)}</th>" for c in columns)
    body = "\n".join(body_rows)

    fa_options = "".join(
        f'<option value="{_html.escape(v)}">{_html.escape(v or "—")}</option>'
        for v in sorted(x for x in fa_values if x)
    )
    att_options = "".join(
        f'<option value="{_html.escape(str(v))}">{_html.escape(str(v))}</option>'
        for v in sorted(x for x in att_values if x)
    )

    summary_html = (
        f"<div><strong>file:</strong> {_html.escape(csv_path.name)}</div>"
        f"<div><strong>cases:</strong> {n_unique_cases}</div>"
        f"<div><strong>rows:</strong> {n_rows}</div>"
        f"<div><strong>resolved:</strong> {n_resolved}</div>"
        f"<div><strong>harness_correct:</strong> {n_correct}</div>"
    )

    pivots_html = (
        f'<div class="pivots">{_pivot_by_agent(df)}{_pivot_by_agent_mode(df)}</div>'
    )

    filters_html = f"""
    <div class="filters">
      <label><input type="checkbox" id="only-resolved"> only resolved</label>
      <label><input type="checkbox" id="only-unresolved"> only unresolved</label>
      <label>failed_agent:
        <select id="fa-filter">
          <option value="">any</option>
          {fa_options}
        </select>
      </label>
      <label>attempt_id:
        <select id="att-filter">
          <option value="">any</option>
          {att_options}
        </select>
      </label>
      <label>search:
        <input type="text" id="search" placeholder="text in any cell" size="40">
      </label>
    </div>
    """

    report_body = f"""  <h1>{_html.escape(title)}</h1>
  <div class="subhead">
    <span class="pill"><strong>variant:</strong> {_html.escape(_variant)}</span>
    <span class="pill"><strong>prompts_version (input):</strong> {_html.escape(_prompts_version)}</span>
    <span class="muted">optimising failures of {_html.escape(_prompts_version)} → producing {_html.escape(_variant)}</span>
  </div>
  <div class="summary">{summary_html}</div>
  {pivots_html}
  {filters_html}
{viewer_panel_html()}
  <table>
    <thead><tr>{head_cells}</tr></thead>
    <tbody>{body}</tbody>
  </table>"""
    page = render_page(title=title, body=report_body, filter_script=_FILTER_SCRIPT)
    out_path.write_text(page)
    return out_path
