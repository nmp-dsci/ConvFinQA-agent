"""HTML report and terminal accuracy-table renderers for predictions CSVs."""

from __future__ import annotations

import html as _html
from pathlib import Path
from typing import Any

import pandas as pd

from convfinqa.pipeline.prompts_loader import GEPA_NAME
from convfinqa.reporting import render_cell, render_page, viewer_panel_html

_JSON_COLUMNS = {
    "triage_io",
    "preprocess_io",
    "retriever_io",
    "calculator_io",
    "pred_sub_questions",
}
_LONG_TEXT_COLUMNS = {"history_text", "question"}


def write_predictions_html(csv_path: Path, output_path: Path | None = None) -> Path:
    """Render an HTML report from a predictions CSV."""
    df = pd.read_csv(csv_path).fillna("")
    out_path = output_path or csv_path.with_suffix(".html")
    columns = list(df.columns)

    def _cell(col: str, value: object, *, row_id: str) -> str:
        return render_cell(
            col,
            value,
            row_id=row_id,
            json_columns=_JSON_COLUMNS,
            long_text_columns=_LONG_TEXT_COLUMNS,
        )

    correct_series = df["correct"] if "correct" in df.columns else None
    n_rows = len(df)
    n_correct = (
        int(correct_series.astype(str).str.lower().isin({"true", "1"}).sum())
        if correct_series is not None
        else 0
    )
    n_incorrect = n_rows - n_correct
    accuracy = n_correct / n_rows if n_rows else 0.0

    # Conversation-level (report_id) stats. A report counts as "correct" iff
    # every one of its turns is correct. The incorrect-report count here must
    # match the number of first-wrong cases the diagnostic harness consumes
    # for this predictions CSV (see diagnostic_results_<variant>.html).
    n_reports = 0
    n_reports_correct = 0
    n_reports_incorrect = 0
    reports_accuracy = 0.0
    if correct_series is not None and "report_id" in df.columns:
        ok_bool = correct_series.astype(str).str.lower().isin({"true", "1"})
        per_report_all_ok = ok_bool.groupby(df["report_id"]).all()
        n_reports = int(per_report_all_ok.shape[0])
        n_reports_correct = int(per_report_all_ok.sum())
        n_reports_incorrect = n_reports - n_reports_correct
        reports_accuracy = n_reports_correct / n_reports if n_reports else 0.0

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

    filter_script = """
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
    function attach() {
      ['only-wrong', 'tt-filter', 'ct-filter', 'search'].forEach(id => {
        document.getElementById(id).addEventListener('input', applyFilters);
        document.getElementById(id).addEventListener('change', applyFilters);
      });
      attachViewer();
    }
    document.addEventListener('DOMContentLoaded', attach);
    """

    summary_html = (
        f"<div><strong>file:</strong> {_html.escape(csv_path.name)}</div>"
        f"<div><strong>turns:</strong> {n_rows}</div>"
        f"<div><strong>turns-correct:</strong> {n_correct}</div>"
        f"<div><strong>turns-incorrect:</strong> {n_incorrect}</div>"
        f"<div><strong>accuracy-correct:</strong> {accuracy:.1%}</div>"
        f"<div><strong>reports:</strong> {n_reports}</div>"
        f"<div><strong>reports-correct:</strong> {n_reports_correct}</div>"
        f"<div><strong>reports-incorrect:</strong> {n_reports_incorrect}</div>"
        f"<div><strong>reports-accuracy:</strong> {reports_accuracy:.1%}</div>"
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

    title = f"Pydantic predictions — {GEPA_NAME}"
    report_body = f"""  <h1>Pydantic predictions — {_html.escape(GEPA_NAME)}</h1>
  <div class="summary">{summary_html}</div>
  {filters_html}
{viewer_panel_html()}
  <table>
    <thead><tr>{head_cells}</tr></thead>
    <tbody>{body}</tbody>
  </table>"""
    page = render_page(title=title, body=report_body, filter_script=filter_script)
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

    def _row(label: str, masks: dict[str, pd.Series]) -> tuple[Any, ...]:
        count = int(masks[versions[0]].sum())
        accs = []
        for v in versions:
            m = masks[v]
            total = int(m.sum())
            correct = int(dfs[v].loc[m, "_ok"].sum())
            accs.append(correct / total if total else 0.0)
        return (label, count, *accs)

    rows: list[tuple[Any, ...]] = []
    rows.append(_row("Overall", {v: pd.Series([True] * len(dfs[v])) for v in versions}))
    for tt in ["Number", "Program"]:
        rows.append(
            _row(
                f"turn_type={tt}",
                {
                    v: dfs[v]["gold_turn_type"].str.lower() == tt.lower()
                    for v in versions
                },
            )
        )
    for ct in ["Type I", "Type II"]:
        rows.append(
            _row(
                f"conv_type={ct}", {v: dfs[v]["gold_conv_type"] == ct for v in versions}
            )
        )
    max_turn = max(int(dfs[v]["turn_index"].max()) for v in versions)
    for ti in range(max_turn + 1):
        rows.append(
            _row(f"question={ti}", {v: dfs[v]["turn_index"] == ti for v in versions})
        )

    col_w = 22
    ver_w = 12
    header = f"{'Cut':<{col_w}}  {'Count':>6}" + "".join(
        f"  {v:>{ver_w}}" for v in versions
    )
    sep = "-" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")  # noqa: T201
    for i, (label, count, *accs) in enumerate(rows):
        if i in {1, 3, 5}:
            print()  # noqa: T201
        acc_str = "".join(f"  {a:>{ver_w - 1}.1%} " for a in accs)
        print(f"{label:<{col_w}}  {count:>6}{acc_str}")  # noqa: T201
    print(sep)  # noqa: T201
