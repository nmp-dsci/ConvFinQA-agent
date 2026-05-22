"""HTML report and terminal accuracy-table renderers for predictions CSVs."""

from __future__ import annotations

import html as _html
import json
from pathlib import Path
from typing import Any

import pandas as pd

from convfinqa.pipeline.prompts_loader import GEPA_NAME


def write_predictions_html(csv_path: Path, output_path: Path | None = None) -> Path:
    """Render an HTML report from a predictions CSV."""
    df = pd.read_csv(csv_path).fillna("")
    out_path = output_path or csv_path.with_suffix(".html")
    columns = list(df.columns)

    json_columns = {
        "triage_io",
        "preprocess_io",
        "retriever_io",
        "calculator_io",
        "pred_sub_questions",
    }
    long_text_columns = {"history_text", "question"}

    def _viewable(col: str, label: str, body: str, row_id: str) -> str:
        return (
            f'<button class="view-btn" type="button" data-col="{_html.escape(col)}" '
            f'data-row="{_html.escape(row_id)}">{_html.escape(label)}</button>'
            f'<pre class="hidden-content">{_html.escape(body)}</pre>'
        )

    def _cell(col: str, value: Any, *, row_id: str) -> str:
        s = "" if value is None else str(value)
        if col in json_columns and s.strip():
            try:
                parsed = json.loads(s)
                pretty = json.dumps(parsed, indent=2, ensure_ascii=False)
            except (json.JSONDecodeError, TypeError):
                pretty = s
            return _viewable(col, "view", pretty, row_id)
        if col in long_text_columns and len(s) > 200:
            return _viewable(col, s[:120] + "…", s, row_id)
        return _html.escape(s)

    correct_series = df["correct"] if "correct" in df.columns else None
    n_rows = len(df)
    n_correct = (
        int(correct_series.astype(str).str.lower().isin({"true", "1"}).sum())
        if correct_series is not None
        else 0
    )
    accuracy = n_correct / n_rows if n_rows else 0.0

    body_rows: list[str] = []
    for _, row in df.iterrows():
        ok = str(row.get("correct", "")).lower() in {"true", "1"}
        row_class = "row-correct" if ok else "row-wrong"
        tt = _html.escape(str(row.get("pred_turn_type", "")))
        ct = _html.escape(str(row.get("pred_conv_type", "")))
        row_id = f"{row.get('report_id', '')}#turn={row.get('turn_index', '')}"
        cells = "".join(f"<td>{_cell(c, row[c], row_id=row_id)}</td>" for c in columns)
        body_rows.append(
            f'<tr class="{row_class}" data-tt="{tt}" data-ct="{ct}" '
            f'data-correct="{str(ok).lower()}">{cells}</tr>'
        )

    head_cells = "".join(f"<th>{_html.escape(c)}</th>" for c in columns)
    body = "\n".join(body_rows)

    style = """
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
    tr.row-correct:hover td { background: rgba(0, 168, 132, 0.18); }
    tr.row-wrong:hover td { background: rgba(241, 92, 109, 0.18); }
    tr.hidden { display: none; }
    .hidden-content { display: none; }
    pre { background: var(--bg); padding: 8px 10px; border-radius: 4px;
          border: 1px solid var(--border); color: var(--text-main);
          max-height: 320px; overflow: auto; font-size: 12px;
          white-space: pre-wrap; word-break: break-word;
          font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
    .view-btn { background: transparent; color: var(--accent2);
                border: 1px solid var(--border); border-radius: 4px;
                padding: 2px 8px; cursor: pointer; font: inherit; font-size: 12px; }
    .view-btn:hover { background: var(--panel2); color: var(--text-main); }
    .view-btn.active { background: var(--accent); color: var(--text-main);
                       border-color: var(--accent2); }
    .viewer { position: sticky; top: 0; z-index: 5; background: var(--panel);
              border: 1px solid var(--border); border-radius: 6px;
              padding: 8px 12px; margin-bottom: 0.75em; }
    .viewer.hidden { display: none; }
    .viewer-header { display: flex; justify-content: space-between;
                     align-items: center; margin-bottom: 6px; gap: 1em; }
    .viewer-label { color: var(--text-muted); font-size: 12px;
                    text-transform: uppercase; letter-spacing: 0.04em; }
    .viewer-label strong { color: var(--text-main); margin-right: 0.4em;
                           text-transform: none; letter-spacing: 0;
                           font-weight: 500; }
    .viewer-close { background: transparent; color: var(--text-muted);
                    border: 0; font-size: 18px; cursor: pointer; padding: 0 6px; }
    .viewer-close:hover { color: var(--danger); }
    .viewer pre { margin: 0; max-height: 50vh; }
    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    """

    script = """
    let _activeViewBtn = null;
    function applyFilters() {
      const onlyWrong = document.getElementById('only-wrong').checked;
      const tt = document.getElementById('tt-filter').value;
      const ct = document.getElementById('ct-filter').value;
      const search = document.getElementById('search').value.toLowerCase();
      document.querySelectorAll('tbody tr').forEach(tr => {
        let show = true;
        if (onlyWrong && tr.dataset.correct !== 'false') show = false;
        if (tt && tr.dataset.tt !== tt) show = false;
        if (ct && tr.dataset.ct !== ct) show = false;
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
      document.getElementById('viewer-col').textContent = btn.dataset.col || '';
      document.getElementById('viewer-row').textContent = btn.dataset.row || '';
      document.getElementById('viewer-content').textContent = content.textContent;
      document.getElementById('viewer-panel').classList.remove('hidden');
      if (_activeViewBtn) _activeViewBtn.classList.remove('active');
      _activeViewBtn = btn;
      btn.classList.add('active');
    }
    function attach() {
      ['only-wrong', 'tt-filter', 'ct-filter', 'search'].forEach(id => {
        document.getElementById(id).addEventListener('input', applyFilters);
        document.getElementById(id).addEventListener('change', applyFilters);
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

    summary_html = (
        f"<div><strong>file:</strong> {_html.escape(csv_path.name)}</div>"
        f"<div><strong>turns:</strong> {n_rows}</div>"
        f"<div><strong>correct:</strong> {n_correct}</div>"
        f"<div><strong>accuracy:</strong> {accuracy:.1%}</div>"
    )

    filters_html = """
    <div class="filters">
      <label><input type="checkbox" id="only-wrong"> only wrong</label>
      <label>turn_type:
        <select id="tt-filter">
          <option value="">any</option>
          <option value="number">number</option>
          <option value="program">program</option>
        </select>
      </label>
      <label>conv_type:
        <select id="ct-filter">
          <option value="">any</option>
          <option value="Type I">Type I</option>
          <option value="Type II">Type II</option>
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
  <title>Pydantic predictions — {_html.escape(GEPA_NAME)}</title>
  <style>{style}</style>
</head>
<body>
  <h1>Pydantic predictions — {_html.escape(GEPA_NAME)}</h1>
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
  <script>{script}</script>
</body>
</html>
"""
    out_path.write_text(page)
    print(f"Wrote {out_path}")  # noqa: T201
    return out_path


def print_accuracy_table(csv_paths: dict[str, Path]) -> None:
    """Print a terminal accuracy comparison table across prompt versions."""
    versions = list(csv_paths.keys())
    dfs: dict[str, pd.DataFrame] = {}
    for v, path in csv_paths.items():
        df = pd.read_csv(path).fillna("")
        df["_ok"] = df["correct"].astype(str).str.lower().isin({"true", "1"})
        dfs[v] = df

    def _row(label: str, masks: dict[str, pd.Series]) -> tuple:
        count = int(masks[versions[0]].sum())
        accs = []
        for v in versions:
            m = masks[v]
            total = int(m.sum())
            correct = int(dfs[v].loc[m, "_ok"].sum())
            accs.append(correct / total if total else 0.0)
        return (label, count, *accs)

    rows: list[tuple] = []
    rows.append(_row("Overall", {v: pd.Series([True] * len(dfs[v])) for v in versions}))
    for tt in ["Number", "Program"]:
        rows.append(
            _row(
                f"turn_type={tt}",
                {v: dfs[v]["gold_turn_type"].str.lower() == tt.lower() for v in versions},
            )
        )
    for ct in ["Type I", "Type II"]:
        rows.append(
            _row(f"conv_type={ct}", {v: dfs[v]["gold_conv_type"] == ct for v in versions})
        )
    max_turn = max(int(dfs[v]["turn_index"].max()) for v in versions)
    for ti in range(max_turn + 1):
        rows.append(_row(f"question={ti}", {v: dfs[v]["turn_index"] == ti for v in versions}))

    col_w = 22
    ver_w = 12
    header = f"{'Cut':<{col_w}}  {'Count':>6}" + "".join(f"  {v:>{ver_w}}" for v in versions)
    sep = "-" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")  # noqa: T201
    for i, (label, count, *accs) in enumerate(rows):
        if i in {1, 3, 5}:
            print()  # noqa: T201
        acc_str = "".join(f"  {a:>{ver_w - 1}.1%} " for a in accs)
        print(f"{label:<{col_w}}  {count:>6}{acc_str}")  # noqa: T201
    print(sep)  # noqa: T201
