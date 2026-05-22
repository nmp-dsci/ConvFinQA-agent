"""Render diagnostic_results_v3_opt.html — dark-theme clone of pydantic predictions."""

from __future__ import annotations

import html as _html
import json
from pathlib import Path
from typing import Any

import pandas as pd

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

_STYLE = """
:root {
  --bg: #0b141a;
  --panel: #111b21;
  --panel2: #202c33;
  --border: #2a3942;
  --accent: #005c4b;
  --accent2: #00a884;
  --text-main: #e9edef;
  --text-muted: #8696a0;
  --danger: #f15c6d;
  --row-correct: rgba(0, 168, 132, 0.10);
  --row-wrong: rgba(241, 92, 109, 0.10);
  --row-pending: rgba(150, 150, 150, 0.07);
}
html, body { background: var(--bg); }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       margin: 0.75em 1em; color: var(--text-main); font-size: 13px; }
h1 { font-size: 18px; margin: 0.5em 0; color: var(--text-main); font-weight: 600; }
.summary { display: flex; gap: 0.75em; margin-bottom: 0.75em; flex-wrap: wrap; }
.summary div { background: var(--panel2); border-radius: 6px; padding: 6px 12px;
               color: var(--text-main); border: 1px solid var(--border); }
.summary strong { color: var(--text-muted); font-weight: 500; margin-right: 0.4em; }
.filters { display: flex; gap: 1em; margin-bottom: 0.75em; flex-wrap: wrap;
           align-items: center; background: var(--panel); padding: 8px 12px;
           border-radius: 6px; border: 1px solid var(--border); }
.filters label { color: var(--text-muted); display: inline-flex; align-items: center;
                 gap: 0.4em; }
.filters input[type="text"], .filters select {
  background: var(--panel2); color: var(--text-main); border: 1px solid var(--border);
  border-radius: 4px; padding: 3px 6px; font: inherit;
}
.filters input[type="checkbox"] { accent-color: var(--accent2); }
table { border-collapse: collapse; width: 100%; table-layout: auto;
        background: var(--panel); border: 1px solid var(--border); }
th, td { border: 1px solid var(--border); padding: 5px 8px; vertical-align: top;
         max-width: 420px; word-wrap: break-word; }
th { background: var(--panel2); color: var(--text-muted); position: sticky;
     top: 0; z-index: 1; cursor: pointer; font-weight: 500; text-align: left;
     text-transform: uppercase; font-size: 11px; letter-spacing: 0.03em; }
td { color: var(--text-main); }
tbody tr:hover td { background: var(--panel2); }
tr.row-correct td { background: var(--row-correct); }
tr.row-wrong td { background: var(--row-wrong); }
tr.row-pending td { background: var(--row-pending); }
tr.row-correct:hover td { background: rgba(0, 168, 132, 0.18); }
tr.row-wrong:hover td { background: rgba(241, 92, 109, 0.18); }
tr.hidden { display: none; }
.hidden-content { display: none; }
pre { background: var(--bg); padding: 8px 10px; border-radius: 4px;
      border: 1px solid var(--border); color: var(--text-main);
      max-height: 320px; overflow: auto; font-size: 12px;
      white-space: pre-wrap; word-break: break-word;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.view-btn { background: transparent; color: var(--accent2); border: 1px solid var(--border);
            border-radius: 4px; padding: 2px 8px; cursor: pointer; font: inherit;
            font-size: 12px; }
.view-btn:hover { background: var(--panel2); color: var(--text-main); }
.view-btn.active { background: var(--accent); color: var(--text-main);
                   border-color: var(--accent2); }
.viewer { position: sticky; top: 0; z-index: 5; background: var(--panel);
          border: 1px solid var(--border); border-radius: 6px;
          padding: 8px 12px; margin-bottom: 0.75em; }
.viewer.hidden { display: none; }
.viewer-header { display: flex; justify-content: space-between; align-items: center;
                 margin-bottom: 6px; gap: 1em; }
.viewer-label { color: var(--text-muted); font-size: 12px;
                text-transform: uppercase; letter-spacing: 0.04em; }
.viewer-label strong { color: var(--text-main); margin-right: 0.4em;
                       text-transform: none; letter-spacing: 0; font-weight: 500; }
.viewer-close { background: transparent; color: var(--text-muted); border: 0;
                font-size: 18px; cursor: pointer; padding: 0 6px; }
.viewer-close:hover { color: var(--danger); }
.viewer pre { margin: 0; max-height: 50vh; }
.placeholder { color: var(--text-muted); font-style: italic; }
.fix-box { background: rgba(0, 168, 132, 0.08); border-left: 3px solid var(--accent2);
           padding: 4px 8px; border-radius: 3px; white-space: pre-wrap;
           font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 6px; }
::-webkit-scrollbar-track { background: transparent; }
"""

_SCRIPT = """
let _activeViewBtn = null;
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
function closeViewer() {
  const v = document.getElementById('viewer-panel');
  if (v) v.classList.add('hidden');
  if (_activeViewBtn) {
    _activeViewBtn.classList.remove('active');
    _activeViewBtn = null;
  }
}
function openViewer(btn) {
  const content = btn.nextElementSibling;
  if (!content) return;
  const colName = btn.dataset.col || '';
  const rowId = btn.dataset.row || '';
  document.getElementById('viewer-col').textContent = colName;
  document.getElementById('viewer-row').textContent = rowId;
  document.getElementById('viewer-content').textContent = content.textContent;
  document.getElementById('viewer-panel').classList.remove('hidden');
  if (_activeViewBtn) _activeViewBtn.classList.remove('active');
  _activeViewBtn = btn;
  btn.classList.add('active');
}
function attach() {
  ['only-resolved', 'only-unresolved', 'fa-filter', 'att-filter', 'search'].forEach(id => {
    const el = document.getElementById(id);
    if (!el) return;
    el.addEventListener('input', applyFilters);
    el.addEventListener('change', applyFilters);
  });
  document.querySelectorAll('.view-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      if (_activeViewBtn === btn) { closeViewer(); return; }
      openViewer(btn);
    });
  });
  const closeBtn = document.getElementById('viewer-close');
  if (closeBtn) closeBtn.addEventListener('click', closeViewer);
  document.addEventListener('keydown', e => { if (e.key === 'Escape') closeViewer(); });
}
document.addEventListener('DOMContentLoaded', attach);
"""


def _viewable(col: str, label: str, body: str, row_id: str) -> str:
    return (
        f'<button class="view-btn" type="button" data-col="{_html.escape(col)}" '
        f'data-row="{_html.escape(row_id)}">{_html.escape(label)}</button>'
        f'<pre class="hidden-content">{_html.escape(body)}</pre>'
    )


def _cell(col: str, value: Any, *, row_id: str) -> str:
    s = "" if value is None else str(value)
    if s.strip() == "":
        return '<span class="placeholder">—</span>'
    if col == "system_prompt_fix" and s.strip():
        return f'<div class="fix-box">{_html.escape(s)}</div>'
    if col in _JSON_COLUMNS and s.strip():
        try:
            parsed = json.loads(s)
            pretty = json.dumps(parsed, indent=2, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError):
            pretty = s
        return _viewable(col, "view", pretty, row_id)
    if col in _LONG_TEXT_COLUMNS and len(s) > 200:
        return _viewable(col, s[:120] + "…", s, row_id)
    return _html.escape(s)


def write_diagnostic_html(
    csv_path: Path,
    *,
    output_path: Path | None = None,
    title: str = "diagnostic_results_v3_opt",
) -> Path:
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

    page = f"""<!doctype html>
<html lang="en" class="dark">
<head>
  <meta charset="utf-8">
  <meta name="color-scheme" content="dark">
  <title>{_html.escape(title)}</title>
  <style>{_STYLE}</style>
</head>
<body>
  <h1>{_html.escape(title)}</h1>
  <div class="summary">{summary_html}</div>
  {filters_html}
  <div id="viewer-panel" class="viewer hidden">
    <div class="viewer-header">
      <span class="viewer-label">
        <strong id="viewer-col"></strong>
        <span id="viewer-row"></span>
      </span>
      <button class="viewer-close" id="viewer-close" type="button" aria-label="close">×</button>
    </div>
    <pre id="viewer-content"></pre>
  </div>
  <table>
    <thead><tr>{head_cells}</tr></thead>
    <tbody>{body}</tbody>
  </table>
  <script>{_SCRIPT}</script>
</body>
</html>
"""
    out_path.write_text(page)
    return out_path
