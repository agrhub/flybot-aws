"""Render the diagrams to PNG and docs/REPORT.md to docs/REPORT.pdf (through docs/REPORT.html) with Chrome headless.

    uv run --no-project python docs/diagrams/src/build_diagrams.py              # SVGs first
    uv run --no-project --with markdown==3.7 python docs/diagrams/src/render.py  # PNGs (1920 px wide) + HTML + PDF (A4)
    ... render.py --png | --pdf [--keep-html]                                    # one of the two

Needs a local Chrome or Edge (CHROME_PATH overrides the search). docs/REPORT.html is an intermediate step (it embeds
the SVG diagrams by relative path) and is removed after printing unless --keep-html is given.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DIAGRAMS = HERE.parent
DOCS = DIAGRAMS.parent

CANDIDATES = [
    os.environ.get("CHROME_PATH", ""),
    "C:/Program Files/Google/Chrome/Application/chrome.exe",
    "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
]

CSS = """
@page { size: A4; margin: 14mm 13mm; }
body { font-family: "Segoe UI", Arial, Helvetica, sans-serif; font-size: 9.4pt; line-height: 1.38; color: #0f172a; }
h1 { font-size: 19pt; margin: 0 0 6pt; }
h2 { font-size: 13pt; margin: 14pt 0 5pt; border-bottom: 1.2pt solid #cbd5e1; padding-bottom: 2pt; break-after: avoid; }
h3 { font-size: 10.5pt; margin: 10pt 0 4pt; break-after: avoid; }
p, li { margin: 3pt 0; }
ul, ol { padding-left: 16pt; margin: 3pt 0; }
table { border-collapse: collapse; width: 100%; margin: 5pt 0 7pt; font-size: 8.2pt; break-inside: auto; }
th, td { border: 0.6pt solid #cbd5e1; padding: 2.5pt 4pt; vertical-align: top; text-align: left; }
th { background: #f1f5f9; }
tr { break-inside: avoid; }
code { font-family: Consolas, "Courier New", monospace; font-size: 8.2pt; background: #f1f5f9; padding: 0 1.5pt; }
pre { background: #f8fafc; border: 0.6pt solid #cbd5e1; padding: 5pt 7pt; font-size: 7.6pt; line-height: 1.3;
      white-space: pre-wrap; break-inside: avoid; }
pre code { background: none; padding: 0; font-size: 7.6pt; }
img { width: 100%; border: 0.6pt solid #e2e8f0; margin: 4pt 0 0; break-inside: avoid; }
p:has(> img) { break-inside: avoid; margin: 6pt 0 0; }
p:has(+ pre), p:has(+ table), p:has(+ ul) { break-after: avoid; }
em { color: #475569; }
a { color: #1d4ed8; text-decoration: none; }
"""


def chrome() -> str:
    for c in CANDIDATES:
        if c and Path(c).exists():
            return c
    sys.exit("no Chrome or Edge found; set CHROME_PATH")


def run_chrome(args: list[str]) -> None:
    with tempfile.TemporaryDirectory() as profile:
        subprocess.run([chrome(), "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--user-data-dir={profile}",
                        *args], check=True, capture_output=True, timeout=180)


def pngs() -> None:
    for svg in sorted(DIAGRAMS.glob("*.svg")):
        out = svg.with_suffix(".png")
        w, h = re.search(r'viewBox="0 0 (\d+) (\d+)"', svg.read_text(encoding="utf-8")).groups()
        run_chrome([f"--screenshot={out}", f"--window-size={w},{h}", svg.as_uri()])
        print("wrote", out.relative_to(DOCS.parent).as_posix())


def report() -> None:
    import markdown

    md = (DOCS / "REPORT.md").read_text(encoding="utf-8")
    body = markdown.markdown(md, extensions=["tables", "fenced_code"])
    html = ('<!doctype html><html lang="en"><head><meta charset="utf-8"><title>FlyBot on AWS: technical report'
            f'</title><style>{CSS}</style></head><body>{body}</body></html>\n')
    (DOCS / "REPORT.html").write_text(html, encoding="utf-8", newline="\n")
    print("wrote docs/REPORT.html")
    out = DOCS / "REPORT.pdf"
    run_chrome([f"--print-to-pdf={out}", "--no-pdf-header-footer", (DOCS / "REPORT.html").as_uri()])
    print("wrote docs/REPORT.pdf", out.stat().st_size, "bytes")
    if "--keep-html" not in sys.argv:
        (DOCS / "REPORT.html").unlink()


def main() -> None:
    args = set(sys.argv[1:]) - {"--keep-html"}
    if not args or "--png" in args:
        pngs()
    if not args or "--pdf" in args:
        report()


if __name__ == "__main__":
    main()
