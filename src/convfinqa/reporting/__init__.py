"""Shared HTML-report mechanics for predictions and diagnostic renderers.

This package owns the *how* of rendering dark-theme HTML tables with a sticky
inspector panel: the CSS theme, the viewable-cell helpers, the inspector-panel
markup, and the viewer JavaScript. Callers (``evaluation.reporting`` and
``diagnosis.results_html``) own the *what*: which columns are JSON vs long-text,
which summary/pivot/filter blocks to emit, and their own filter JS.
"""

from convfinqa.reporting.html_report import (
    REPORT_CSS,
    render_cell,
    render_page,
    viewable,
    viewer_js,
    viewer_panel_html,
)

__all__ = [
    "REPORT_CSS",
    "render_cell",
    "render_page",
    "viewable",
    "viewer_js",
    "viewer_panel_html",
]
