"""Render docs/diagrams/*.mmd to light and dark SVGs for the README.

PyPI does not render Mermaid, so the README embeds these SVGs through a
``<picture>`` with absolute raw.githubusercontent.com URLs — the same pattern
as the capability radar. Each SVG carries its own background, so it looks the
same on GitHub light, GitHub dark and PyPI, and labels are plain SVG text
(no ``<foreignObject>``), which is what keeps them visible inside an ``<img>``.

Needs, once:
    pip install playwright && playwright install chromium
    npm install --no-save mermaid          # or pass --mermaid-js

Then:
    python scripts/render_diagrams.py

Already have a Chromium? Skip `playwright install` and pass
`--chromium /path/to/chrome`.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from playwright.sync_api import sync_playwright

REPO = Path(__file__).resolve().parent.parent
DIAGRAMS = REPO / "docs" / "diagrams"
PAD = 16

# Surfaces match docs/capability-radar-*.svg so the README reads as one set.
THEMES = {
    "light": {"theme": "default", "surface": "#fcfcfb"},
    "dark": {"theme": "dark", "surface": "#1a1a19"},
}

# Helvetica/Arial on macOS and Windows; Liberation Sans (metric-compatible
# with Arial) on Linux. Node boxes are sized from the rendering machine's font
# metrics, so the family must measure the same everywhere the SVG is viewed.
FONT = "Helvetica, Arial, sans-serif"

RENDER_JS = """
async ([src, theme]) => {
  mermaid.initialize({
    startOnLoad: false,
    theme,
    securityLevel: "strict",
    // Both are needed: nodes read the top-level flag, edges the flowchart one.
    htmlLabels: false,
    flowchart: { htmlLabels: false, padding: 12, nodeSpacing: 36, rankSpacing: 44 },
    themeVariables: { fontFamily: __FONT__, fontSize: "15px" },
  });
  const { svg } = await mermaid.render("d" + Math.random().toString(36).slice(2), src);
  return svg;
}
""".replace("__FONT__", json.dumps(FONT))


def alt_text(source: str) -> str:
    match = re.search(r"^%%\s*alt:\s*(.+)$", source, re.M)
    return match.group(1).strip() if match else ""


def finalize(svg: str, surface: str, alt: str) -> str:
    """Give the SVG fixed dimensions, a margin, a background, and alt text."""
    match = re.search(r'viewBox="([-\d.]+) ([-\d.]+) ([-\d.]+) ([-\d.]+)"', svg)
    if not match:
        raise SystemExit("mermaid output has no viewBox")
    x, y, w, h = (float(v) for v in match.groups())
    x, y, w, h = x - PAD, y - PAD, w + 2 * PAD, h + 2 * PAD

    head_end = svg.index(">") + 1
    head, body = svg[:head_end], svg[head_end:]
    # Mermaid sets its own role/aria attributes; replace rather than duplicate
    # them, since a duplicate attribute makes the file invalid XML and browsers
    # then refuse to show it inside <img>.
    head = re.sub(r'\s(width|height|style|role|aria-roledescription)="[^"]*"', "", head)
    head = re.sub(r'viewBox="[^"]*"', f'viewBox="{x:g} {y:g} {w:g} {h:g}"', head)
    alt_attr = html.escape(alt, quote=True)
    head = head[:-1] + f' width="{w:g}" height="{h:g}" role="img" aria-label="{alt_attr}">'

    backdrop = f'<rect x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" fill="{surface}"/>'
    title = f"<title>{html.escape(alt)}</title>" if alt else ""
    result = head + title + backdrop + body
    ET.fromstring(result)  # raises on malformed output rather than writing a broken image
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--mermaid-js",
        type=Path,
        default=REPO / "node_modules" / "mermaid" / "dist" / "mermaid.min.js",
        help="path to mermaid.min.js (default: ./node_modules/mermaid/dist/mermaid.min.js)",
    )
    parser.add_argument(
        "--chromium",
        type=Path,
        default=None,
        help="use this Chromium binary instead of Playwright's own download",
    )
    args = parser.parse_args()
    if not args.mermaid_js.is_file():
        print(f"{args.mermaid_js} not found; run `npm install --no-save mermaid`", file=sys.stderr)
        return 1

    sources = sorted(DIAGRAMS.glob("*.mmd"))
    if not sources:
        print(f"no .mmd files in {DIAGRAMS}", file=sys.stderr)
        return 1

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            executable_path=str(args.chromium) if args.chromium else None
        )
        page = browser.new_page()
        page.set_content("<!doctype html><html><body></body></html>")
        page.add_script_tag(path=str(args.mermaid_js))
        for source_path in sources:
            source = source_path.read_text(encoding="utf-8")
            alt = alt_text(source)
            for name, spec in THEMES.items():
                svg = page.evaluate(RENDER_JS, [source, spec["theme"]])
                out = DIAGRAMS / f"{source_path.stem}-{name}.svg"
                out.write_text(finalize(svg, spec["surface"], alt), encoding="utf-8")
                print(f"wrote {out.relative_to(REPO)}")
        browser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
