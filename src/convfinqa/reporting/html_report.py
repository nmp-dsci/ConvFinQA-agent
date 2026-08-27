"""Reusable mechanics for dark-theme HTML reports with a sticky inspector panel.

Both the predictions report (``evaluation.reporting``) and the diagnostic report
(``diagnosis.results_html``) render a filterable table whose long/JSON cells pop
their content into a sticky viewer panel above the table. The theme, the viewer
markup, the viewer JS, and the viewable-cell logic are identical between them —
they live here so a fix lands in both reports at once.

Callers keep their own domain concerns: column classification, summary/pivot
blocks, filter controls, and the page-specific ``applyFilters`` JS.
"""

from __future__ import annotations

import html as _html
import json
from collections.abc import Callable
from typing import Any

# Strict superset of both reports' styles. Predictions pages simply don't emit
# the ``.subhead``/``.pivots``/``.fix-box``/``row-pending`` markup, so those
# rules are inert there.
REPORT_CSS = """
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
.subhead { display: flex; gap: 0.6em; align-items: center; flex-wrap: wrap;
  margin: 0 0 0.85em 0; font-size: 13px; }
.subhead .pill { background: var(--panel2); border-radius: 999px;
  padding: 4px 10px; color: var(--text-main); }
.subhead .pill strong { color: var(--text-muted); font-weight: 500;
  margin-right: 0.4em; }
.subhead .muted { color: var(--text-muted); font-style: italic; }
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
.pivots { display: flex; gap: 1em; margin-bottom: 0.75em; flex-wrap: wrap; }
.pivot { background: var(--panel); border: 1px solid var(--border); border-radius: 6px;
         padding: 8px 10px; }
.pivot h2 { font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em;
            color: var(--text-muted); font-weight: 500; margin: 0 0 6px 0; }
.pivot table { table-layout: auto; width: auto; }
.pivot th, .pivot td { padding: 3px 10px; font-size: 12px;
                       border: 1px solid var(--border); cursor: default;
                       max-width: none; white-space: nowrap; }
.pivot th { position: static; text-align: left; }
.pivot td.num { text-align: right; font-variant-numeric: tabular-nums; }
.pivot tr.total td { font-weight: 600; border-top: 2px solid var(--accent2);
                     background: var(--panel2); }
.pivot tr.subtotal td { color: var(--text-muted); background: var(--panel2); }
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 6px; }
::-webkit-scrollbar-track { background: transparent; }
"""

# Shared viewer behaviour. Callers add their own ``applyFilters`` and an
# ``attach`` that wires their filter controls and then calls ``attachViewer()``.
_VIEWER_JS = """
let _activeViewBtn = null;
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
function attachViewer() {
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
"""

# Sticky inspector panel that ``openViewer`` populates.
_VIEWER_PANEL_HTML = """  <div id="viewer-panel" class="viewer hidden">
    <div class="viewer-header">
      <span class="viewer-label">
        <strong id="viewer-col"></strong>
        <span id="viewer-row"></span>
      </span>
      <button class="viewer-close" id="viewer-close" type="button" aria-label="close">×</button>
    </div>
    <pre id="viewer-content"></pre>
  </div>"""


def viewer_js() -> str:
    """Return the shared viewer JavaScript (``openViewer``/``closeViewer``/``attachViewer``)."""
    return _VIEWER_JS


def viewer_panel_html() -> str:
    """Return the sticky inspector-panel markup."""
    return _VIEWER_PANEL_HTML


def viewable(col: str, label: str, body: str, row_id: str) -> str:
    """A ``view`` button plus the hidden ``<pre>`` body that the panel reads."""
    return (
        f'<button class="view-btn" type="button" data-col="{_html.escape(col)}" '
        f'data-row="{_html.escape(row_id)}">{_html.escape(label)}</button>'
        f'<pre class="hidden-content">{_html.escape(body)}</pre>'
    )


def render_cell(
    col: str,
    value: Any,
    *,
    row_id: str,
    json_columns: set[str],
    long_text_columns: set[str],
    empty_placeholder: bool = False,
    special: dict[str, Callable[[str], str]] | None = None,
) -> str:
    """Render one table cell.

    - ``json_columns`` are pretty-printed and made viewable.
    - ``long_text_columns`` longer than 200 chars are truncated and made viewable.
    - ``empty_placeholder`` renders a ``—`` for blank cells (diagnostic report).
    - ``special`` maps a column name to a custom renderer for its raw string.
    """
    s = "" if value is None else str(value)
    if empty_placeholder and s.strip() == "":
        return '<span class="placeholder">—</span>'
    if special and col in special and s.strip():
        return special[col](s)
    if col in json_columns and s.strip():
        try:
            parsed = json.loads(s)
            pretty = json.dumps(parsed, indent=2, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError):
            pretty = s
        return viewable(col, "view", pretty, row_id)
    if col in long_text_columns and len(s) > 200:
        return viewable(col, s[:120] + "…", s, row_id)
    return _html.escape(s)


def render_page(*, title: str, body: str, filter_script: str) -> str:
    """Wrap a report body in the shared dark-theme document shell.

    ``body`` is the full ``<body>`` content excluding the trailing ``<script>``;
    it should include ``viewer_panel_html()`` wherever the panel belongs.
    ``filter_script`` is the caller's page-specific JS (``applyFilters`` + an
    ``attach`` that wires filters and calls ``attachViewer()``).
    """
    return f"""<!doctype html>
<html lang="en" class="dark">
<head>
  <meta charset="utf-8">
  <meta name="color-scheme" content="dark">
  <title>{_html.escape(title)}</title>
  <style>{REPORT_CSS}</style>
</head>
<body>
{body}
  <script>{viewer_js()}
{filter_script}</script>
</body>
</html>
"""
