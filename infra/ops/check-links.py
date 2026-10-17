"""Check that the local links and cited source files in the docs exist.

    python infra/ops/check-links.py                   tracked (and new, not ignored) *.md / *.html; exit 1 on a missing reference
    python infra/ops/check-links.py --list-evidence   print the docs/evidence/ paths the docs cite (files and folders)

Checked: Markdown links `[text](path)` and HTML href/src that are not URLs or anchors, and backtick-quoted repository
paths that start with a known top folder (docs/, eval/, bench/, models/, infra/, lambda/, data/). Paths with
placeholders (`<config>`, `*`, `{a,b}`) are checked for their fixed folder part only. Git-ignored outputs
(models/yolo26/, data/replay/, ...) are reported as "ignored" and the references listed in EXTERNAL as "external";
neither fails the check.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOPS = ("docs/", "eval/", "bench/", "models/", "infra/", "lambda/", "data/", "firmware-esp32/")
MD_LINK = re.compile(r"\]\(([^)\s]+)\)")
TICKED = re.compile(r"`([^`\s]+)`")
HREF = re.compile(r"""(?:href|src)=["']([^"'#]+)["']""")
# Cited on purpose although they are not files of this repository
EXTERNAL = {
    "bench/logs/": "S3 prefix of the benchmark run (s3://<bucket>/bench/logs/)",
    "bench/summary_t41.json": "S3 key; the local copy is bench/results/summary_t41.json",
    "data/labels/": "local labeling workspace; labels are not published",
    "eval/aws_traces.py": "module of the earlier queue-based pipeline, not part of this repository",
}


def docs() -> list[Path]:
    """Tracked docs plus new, not ignored ones (e.g. evidence being added)."""
    out = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--cached", "--others", "--exclude-standard"],
                         capture_output=True, text=True, check=True).stdout
    return [ROOT / f for f in sorted(set(out.split("\n"))) if f.endswith((".md", ".html")) and (ROOT / f).is_file()]


def fixed_part(ref: str) -> str:
    """The part of a path before the first placeholder (<x>, *, {a,b}), cut back to a folder."""
    m = re.search(r"[<*{]", ref)
    if not m:
        return ref
    head = ref[: m.start()]
    return head[: head.rfind("/") + 1] if "/" in head else ""


def references(doc: Path, text: str) -> set[tuple[str, Path]]:
    """(as written, the file it names) for every local reference in a doc."""
    refs: set[tuple[str, Path]] = set()
    for rx in (MD_LINK, HREF):
        for r in rx.findall(text):
            r = r.split("#", 1)[0]
            if not r or re.match(r"^[a-z]+:", r) or r.startswith("/"):
                continue
            fixed = fixed_part(r)
            if fixed:
                refs.add((r, (doc.parent / fixed).resolve()))
    for r in TICKED.findall(text):
        r = r.rstrip(".,;:)")
        if r.startswith(TOPS) and "=" not in r and not r.startswith("docs/plan"):
            fixed = fixed_part(r)
            if fixed:
                refs.add((r, (ROOT / fixed).resolve()))
    return refs


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def is_ignored(r: str) -> bool:
    return any(subprocess.run(["git", "-C", str(ROOT), "check-ignore", "-q", x], check=False).returncode == 0
               for x in (r, r.rstrip("/") + "/x"))


def main(argv: list[str]) -> int:
    missing, ignored, external, evidence = set(), set(), set(), set()
    for doc in docs():
        text = doc.read_text(encoding="utf-8", errors="replace")
        for ref, target in references(doc, text):
            r = rel(target)
            if r.startswith("docs/evidence"):
                evidence.add(r + ("/" if target.is_dir() or ref.endswith("/") else ""))
            if target.exists():
                continue
            where = f"{rel(doc)}: {ref}"
            ext = next((k for k in EXTERNAL if r.startswith(k.rstrip("/")) or ref.startswith(k)), None)
            if ext:
                external.add(f"{where} ({EXTERNAL[ext]})")
            elif is_ignored(r):
                ignored.add(where)
            else:
                missing.add(where)
    if "--list-evidence" in argv:
        print("\n".join(sorted(evidence)))
        return 0
    for tag, lines in (("external", external), ("ignored ", ignored), ("MISSING ", missing)):
        for line in sorted(lines):
            print(tag, line)
    print(f"{len(missing)} missing, {len(ignored)} git-ignored (generated or private), {len(external)} external")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
