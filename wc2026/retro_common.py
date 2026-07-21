"""Shared chrome for the post-tournament retrospective pages so the four read as one browsable
section instead of dead-end islands: a top nav bar + Open-Graph share tags. Used by the root
review.py / competition.py / report.py and by wc2026.share.comparison_page.

Links are root-relative (/outputs/… , /compare.html) so they resolve the same on the deployed
site and under a static server rooted at the repo. Colours reuse each page's Emerald Pitch tokens.
"""
from __future__ import annotations

SITE = "https://worldcup2026ml.pt"

# (key, root-relative href, label) — the four retrospective destinations, in reading order.
_PAGES = [
    ("review", "/outputs/review.html", "🏆 Review"),
    ("competition", "/outputs/competition.html", "📊 The numbers"),
    ("report", "/outputs/report.html", "🎯 Report card"),
    ("compare", "/compare.html", "🔮 Predicted vs reality"),
    ("journeys", "/outputs/journeys.html", "🧭 Team journeys"),
]

NAV_CSS = """
.rnav{display:flex;flex-wrap:wrap;align-items:center;gap:4px 6px;margin:0 0 22px;padding:0 0 12px;border-bottom:1px solid var(--line)}
.rnav .rnh{font-size:13px;font-weight:600;color:var(--muted);text-decoration:none;margin-right:6px}
.rnav .rnh:hover{color:var(--green)}
.rnav a{font-size:12.5px;font-weight:600;color:var(--muted);text-decoration:none;padding:6px 11px;border-radius:999px;white-space:nowrap}
.rnav a:hover{color:var(--text);background:var(--panel)}
.rnav a.on{color:#052018;background:var(--green)}
@media(max-width:600px){.rnav .rnh{width:100%;margin:0 0 4px}}
"""


def nav(current: str) -> str:
    """The shared top nav, with `current` (a page key) highlighted."""
    parts = ['<nav class="rnav"><a class="rnh" href="/index.html">← Home</a>']
    for k, href, label in _PAGES:
        cls = ' class="on"' if k == current else ""
        parts.append(f'<a href="{href}"{cls}>{label}</a>')
    return "".join(parts) + "</nav>"


def og(title: str, desc: str, url_path: str, image: str = "/outputs/share_card.png") -> str:
    """Open-Graph + Twitter card + theme-color meta for rich link previews."""
    return (
        f'<meta property="og:title" content="{title}">'
        f'<meta property="og:description" content="{desc}">'
        f'<meta property="og:image" content="{SITE}{image}">'
        '<meta property="og:image:width" content="1200"><meta property="og:image:height" content="630">'
        f'<meta property="og:url" content="{SITE}{url_path}"><meta property="og:type" content="website">'
        '<meta name="twitter:card" content="summary_large_image"><meta name="theme-color" content="#0a0e14">')
