"""Stdlib-only helpers for infra/deploy.sh (JSON, secrets merge, hashes). Never prints a secret value.

    python infra/deploy_helpers.py settings deploy.env            shell assignments for the non-secret settings
    python infra/deploy_helpers.py merge-app --env deploy.env --current cur.json --out new.json [--contact EMAIL] [--gcp F]
    python infra/deploy_helpers.py gcp --file sa.json --current cur.json --out new.json
    python infra/deploy_helpers.py get deploy.env KEY --out file   one secret value to a file (exit 3 when empty)
    python infra/deploy_helpers.py ctx-hash                       hash of the vision Lambda build inputs
    python infra/deploy_helpers.py contract response.json --detector yolo26
    python infra/deploy_helpers.py latency bench/results/lambda_latency.json
    python infra/deploy_helpers.py ctx-zip out.zip                the vision Lambda build context for CodeBuild
    python infra/deploy_helpers.py pack-flybot --src ../ezviz-manager --out flybot.tgz [--exclude-media]   the release
    python infra/deploy_helpers.py media-list --src ../ezviz-manager                       videos kept out of it
    python infra/deploy_helpers.py gh-release r.json | gh-prune list.json --keep 5 | gh-new-release ...  GitHub API JSON

deploy.env format: KEY=VALUE per line, `#` comments. Values are taken literally (no $ expansion, nothing is executed);
quote a value with "..." or '...' when it contains `#` or leading/trailing spaces. CRLF files are fine.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import secrets
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Non-secret deploy settings (become shell variables in deploy.sh)
SETTINGS = {
    "STAGE", "AWS_PROFILE", "AWS_REGION", "DOMAIN_NAME", "ACME_EMAIL", "ALARM_EMAIL", "OWNER_CIDR", "DETECTOR",
    "DETECTOR_TILES", "DETECTOR_ROI", "PLANNER_MODE", "INSTANCE_TYPE", "KEY_NAME", "REPO_URL", "REPO_REF", "AMI_ID", "ANTHROPIC_SECRET_NAME",
    "ANTHROPIC_MODEL", "GCP_SERVICE_ACCOUNT_FILE", "DNS_WAIT_MIN", "MODELS_BUCKET", "DEMO_MAX_CAMERAS_PER_USER",
    "DEMO_MAX_SNAPSHOTS_PER_DAY", "DEMO_MAX_UNCLAIMED", "APP_MODE", "DEVICE_EVENTS_URL", "SKIP_COLDSTART",
    "FLYBOT_SRC", "VITE_GOOGLE_MAP_KEY", "S3_BUCKET_NAME", "S3_REGION", "S3_ENDPOINT", "NODE_VERSION", "CADDY_VERSION",
    "FLYBOT_RELEASE_STORE", "GITHUB_REPO", "FLYBOT_MEDIA", "FLYBOT_KEEP_RELEASES", "PAYMENTS_ENABLED",
}
# Payments (FlyBot docs/payments.md): the keys are secrets; the settings are stored the same way (FlyBot reads both
# from flybot-<stage>-app), so switching payments on or off never changes the stack or restarts the instance
PAYMENT_SECRETS = ["PAYOS_CLIENT_ID", "PAYOS_API_KEY", "PAYOS_CHECKSUM_KEY", "STRIPE_SECRET_KEY", "STRIPE_WEBHOOK_SECRET"]
PAYMENT_SETTINGS = ["PAYMENTS_ENABLED", "PAYMENTS_GRACE_DAYS", "PAYMENTS_ALLOW_LIVE", "PAYOS_API_URL", "PAYOS_LINK_TTL_MINUTES",
                    "STRIPE_API_URL", "STRIPE_WEBHOOK_TOLERANCE"]
# FlyBot application values stored in Secrets Manager flybot-<stage>-app (FLYBOT_SECRETS)
APP_KEYS = [
    "JWT_SECRET", "APP_KEY", "SECRET_KEY", "EZVIZ_DOMAIN",
    "SMTP_HOST", "SMTP_PORT", "SMTP_SECURE", "SMTP_USER", "SMTP_PASS", "MAIL_FROM", "CONTACT_EMAIL",
    "TELEGRAM_BOT_TOKEN", "TELEGRAM_BOT_USERNAME",
    "VERTEX_PROJECT_ID", "VERTEX_LOCATION", "VERTEX_MODEL",
    "VAPID_PUBLIC_KEY", "VAPID_PRIVATE_KEY", "VAPID_SUBJECT",
    "MONGO_URI", "S3_ACCESS_KEY", "S3_SECRET_KEY",
    *PAYMENT_SECRETS, *PAYMENT_SETTINGS,
]
# deploy.env decides these alone: empty there = removed from the secret (a local MongoDB / the instance role is used)
AUTHORITATIVE = {"MONGO_URI", "S3_ACCESS_KEY", "S3_SECRET_KEY", *PAYMENT_SECRETS, *PAYMENT_SETTINGS}
OTHER_SECRETS = {"ANTHROPIC_API_KEY", "DEVICE_EVENTS_SECRET", "GITHUB_TOKEN"}   # GITHUB_TOKEN: laptop only, never stored or sent
# files of the prebuilt FlyBot release (what the FlyBot Dockerfile's runtime stage copies)
FLYBOT_FILES = ["package.json", "package-lock.json", "client/package.json", "server/package.json"]
DEFAULTS = {"VERTEX_LOCATION": "global", "VERTEX_MODEL": "gemini-2.5-flash"}
KEY_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


def parse_env(path: str | Path) -> dict[str, str]:
    out: dict[str, str] = {}
    p = Path(path)
    if not p.is_file():
        return out
    for n, raw in enumerate(p.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, sep, val = line.partition("=")
        key = key.strip()
        if not sep or not KEY_RE.match(key):
            print(f"deploy.env line {n}: not KEY=VALUE, ignored", file=sys.stderr)
            continue
        val = val.strip()
        if val[:1] in ("'", '"'):
            q = val[0]
            end = val.find(q, 1)
            val = val[1:end] if end > 0 else val[1:]
        else:
            val = "" if val.startswith("#") else re.split(r"\s+#", val, maxsplit=1)[0].strip()
        out[key] = val
    return out


def cmd_settings(a) -> int:
    env = parse_env(a.file)
    known = SETTINGS | set(APP_KEYS) | OTHER_SECRETS
    for k in sorted(env):
        if k not in known:
            print(f"deploy.env: unknown key {k} (ignored)", file=sys.stderr)
    for k in sorted(SETTINGS & env.keys()):
        print(f"CFG_{k}={shlex.quote(env[k])}")
    # presence flags for the secrets, never their values
    for k in sorted((set(APP_KEYS) | OTHER_SECRETS) & env.keys()):
        if env[k]:
            print(f"HAS_{k}=1")
    return 0


def load_json(path: str | None) -> dict:
    if not path or not Path(path).is_file() or Path(path).stat().st_size == 0:
        return {}
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
    except ValueError:
        return {}
    return d if isinstance(d, dict) else {}


def canon(d: dict) -> str:
    return json.dumps(d, sort_keys=True, separators=(",", ":"))


def b64url(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def vapid_pair() -> tuple[str, str] | None:
    """(public, private) VAPID keys (P-256, base64url, the web-push format) through openssl, or None."""
    exe = shutil.which("openssl")
    if not exe:
        return None
    with tempfile.TemporaryDirectory() as td:
        pem = Path(td) / "k.pem"
        try:
            subprocess.run([exe, "ecparam", "-name", "prime256v1", "-genkey", "-noout", "-out", str(pem)],
                           check=True, capture_output=True, timeout=30)
            text = subprocess.run([exe, "ec", "-in", str(pem), "-noout", "-text"], check=True, capture_output=True,
                                  text=True, timeout=30).stdout
        except (subprocess.SubprocessError, OSError):
            return None
    m_priv = re.search(r"priv:\s*([0-9a-f:\s]+?)\s*pub:", text, re.DOTALL)
    m_pub = re.search(r"pub:\s*([0-9a-f:\s]+?)\s*ASN1 OID", text, re.DOTALL)
    if not (m_priv and m_pub):
        return None
    priv = bytes.fromhex(re.sub(r"[^0-9a-f]", "", m_priv.group(1)))[-32:].rjust(32, b"\0")
    pub = bytes.fromhex(re.sub(r"[^0-9a-f]", "", m_pub.group(1)))
    if len(pub) != 65 or pub[0] != 4:
        return None
    return b64url(pub), b64url(priv)


def cmd_merge_app(a) -> int:
    env = parse_env(a.env)
    cur = load_json(a.current)
    new = dict(cur)  # keys added by hand in the console are kept
    report = {"set": [], "kept": [], "generated": [], "default": [], "empty": []}
    for k in APP_KEYS:
        v = env.get(k, "")
        if v:
            new[k] = v
            report["set" if cur.get(k) != v else "kept"].append(k)
        elif k in AUTHORITATIVE:
            if new.pop(k, None):
                report["set"].append(k + " (removed)")
        elif cur.get(k):
            report["kept"].append(k)
        elif k in DEFAULTS:
            new[k] = DEFAULTS[k]
            report["default"].append(k)
        else:
            new.pop(k, None)
    if not new.get("JWT_SECRET"):
        new["JWT_SECRET"] = secrets.token_urlsafe(36)
        report["generated"].append("JWT_SECRET")
    if not new.get("VAPID_PUBLIC_KEY") or not new.get("VAPID_PRIVATE_KEY"):
        pair = vapid_pair()
        if pair:
            new["VAPID_PUBLIC_KEY"], new["VAPID_PRIVATE_KEY"] = pair
            report["generated"] += ["VAPID_PUBLIC_KEY", "VAPID_PRIVATE_KEY"]
        else:
            print("openssl not available: VAPID keys not generated (web push stays off)", file=sys.stderr)
    if not new.get("VAPID_SUBJECT") and a.contact:
        new["VAPID_SUBJECT"] = f"mailto:{a.contact}"
        report["default"].append("VAPID_SUBJECT")
    if not new.get("VERTEX_PROJECT_ID") and a.gcp:
        proj = load_json(a.gcp).get("project_id")
        if isinstance(proj, str) and proj:
            new["VERTEX_PROJECT_ID"] = proj
            report["default"].append("VERTEX_PROJECT_ID")
    for k in APP_KEYS:
        if not new.get(k):
            report["empty"].append(k)
    new = {k: v for k, v in new.items() if v not in (None, "")}
    out = Path(a.out)
    out.write_text(json.dumps(new, indent=1, sort_keys=True), encoding="utf-8")
    report["changed"] = canon(new) != canon(cur)
    report["set"] = [k for k in report["set"] if k not in report["generated"]]
    print(json.dumps(report))
    return 0


def cmd_gcp(a) -> int:
    cur = load_json(a.current)
    p = Path(a.file) if a.file else None
    if not p or not p.is_file():
        new = cur if cur else {"_comment": "No Google service account: the FlyBot chatbot (Vertex AI / Gemini) is off. "
                                           "Set GCP_SERVICE_ACCOUNT_FILE in deploy.env and run deploy.sh again."}
        status = "kept" if cur.get("private_key") else "placeholder"
    else:
        new = load_json(p)
        missing = [k for k in ("type", "project_id", "private_key", "client_email") if not new.get(k)]
        if missing or new.get("type") != "service_account":
            print(f"{p.name}: not a Google service-account key (missing {missing or ['type=service_account']})",
                  file=sys.stderr)
            return 4
        status = "file"
    Path(a.out).write_text(json.dumps(new), encoding="utf-8")
    print(json.dumps({"changed": canon(new) != canon(cur), "status": status, "project": new.get("project_id")}))
    return 0


def cmd_get(a) -> int:
    v = parse_env(a.file).get(a.key, "")
    if not v:
        return 3
    Path(a.out).write_text(v, encoding="utf-8")
    return 0


def ctx_files() -> list[Path]:
    v = ROOT / "lambda" / "vision"
    files = [v / "Dockerfile", v / "requirements-lambda.txt", v / ".dockerignore", ROOT / "models" / "fetch_models.py"]
    files += sorted(p for p in (v / "sentinel").rglob("*") if p.is_file() and "__pycache__" not in p.parts
                    and p.suffix != ".pyc")
    for d in ("yolov7-tiny", "yolo26"):
        files += sorted((ROOT / "models" / d).glob("*.onnx*"))
    return [f for f in files if f.is_file()]


def cmd_ctx_hash(a) -> int:
    h = hashlib.sha256()
    for f in ctx_files():
        rel = f.relative_to(ROOT).as_posix()
        data = f.read_bytes()
        if f.suffix not in (".onnx",):
            data = data.replace(b"\r\n", b"\n")  # same hash on a Windows (autocrlf) and a Linux checkout
        h.update(rel.encode() + b"\0" + hashlib.sha256(data).digest())
    print(h.hexdigest()[:16])
    return 0


def ctx_arcname(f: Path) -> str:
    rel = f.relative_to(ROOT).as_posix()
    return rel.removeprefix("lambda/vision/")


def cmd_ctx_zip(a) -> int:
    """Dockerfile at the root, sentinel/, requirements-lambda.txt, models/ (what lambda/vision/Dockerfile COPYs)."""
    import zipfile

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in ctx_files():
            z.write(f, ctx_arcname(f))
    print(f"{out.stat().st_size} bytes, {len(ctx_files())} files")
    return 0


def is_media_video(rel: str) -> bool:
    return rel.startswith("media/") and rel.lower().endswith(".mp4")


def release_files(src: Path, exclude_media: bool) -> list[tuple[Path, str]]:
    """(file, arcname) of a clean server/dist: FlyBot's copy-client step copies client/dist into server/dist without
    removing older hashed bundles, so every entry the client build owns is taken from client/dist (fresh) and only the
    server's own outputs (app.js, ...) from server/dist."""
    dist, client = src / "server" / "dist", src / "client" / "dist"
    client_top = {p.name for p in client.iterdir()} if client.is_dir() else set()
    out: list[tuple[Path, str]] = []
    for base, from_client in ((client, True), (dist, False)):
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(base).as_posix()
            if not from_client and rel.split("/", 1)[0] in client_top:
                continue  # stale copy of a client file: the fresh one comes from client/dist
            if exclude_media and is_media_video(rel):
                continue
            out.append((p, "server/dist/" + rel))
    return out


def cmd_pack_flybot(a) -> int:
    """A clean server/dist (the built client inside, stale bundles dropped, videos optional) + the package manifests,
    as a .tgz; prints the sha256."""
    import tarfile

    src = Path(a.src)
    dist = src / "server" / "dist"
    if not (dist / "app.js").is_file():
        print(f"{dist}/app.js missing: run npm run build in {src}", file=sys.stderr)
        return 4
    missing = [f for f in FLYBOT_FILES if not (src / f).is_file()]
    if missing:
        print(f"missing in {src}: {missing}", file=sys.stderr)
        return 4
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    files = release_files(src, a.exclude_media)
    with tarfile.open(out, "w:gz", compresslevel=6) as tar:
        for f in FLYBOT_FILES:
            tar.add(src / f, arcname=f)
        for p, arc in files:
            tar.add(p, arcname=arc)
        if a.meta:
            tar.add(a.meta, arcname="RELEASE.json")
    h = hashlib.sha256(out.read_bytes()).hexdigest()
    print(h)
    return 0


def cmd_media_list(a) -> int:
    """The videos kept out of the release: '<sha256> <size> <path relative to client/dist>' per line."""
    client = Path(a.src) / "client" / "dist"
    for p in sorted((client / "media").glob("*")) if (client / "media").is_dir() else []:
        rel = p.relative_to(client).as_posix()
        if p.is_file() and is_media_video(rel):
            print(hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_size, rel)
    return 0


def cmd_gh_release(a) -> int:
    """GitHub release JSON -> 'id <n>' then 'asset <name> <id> <state> <size>' lines (nothing for a 404 body)."""
    d = load_json(a.file)
    if not d.get("id"):
        return 3
    print("id", d["id"])
    for x in d.get("assets") or []:
        print("asset", x.get("name"), x.get("id"), x.get("state"), x.get("size"))
    return 0


def cmd_gh_prune(a) -> int:
    """Deploy builds to delete: pre-releases tagged flybot-* named '... (deploy build)', newest --keep kept (and the
    current tag always). Prints '<id> <tag>' per release to delete."""
    try:
        rels = json.loads(Path(a.file).read_text(encoding="utf-8"))
    except ValueError:
        return 3
    mine = [r for r in rels if isinstance(r, dict) and str(r.get("tag_name", "")).startswith("flybot-")
            and r.get("prerelease") and str(r.get("name", "")).endswith("(deploy build)")]
    mine.sort(key=lambda r: r.get("created_at") or "", reverse=True)
    for r in mine[a.keep:]:
        if r.get("tag_name") != a.current:
            print(r["id"], r["tag_name"])
    return 0


def cmd_gh_new_release(a) -> int:
    """The JSON body of POST /repos/{repo}/releases for a deploy build (a pre-release, so it is not a product release)."""
    body = (f"Deploy build of the FlyBot app, made by `./deploy.sh` for the AWS server (not a product release).\n\n"
            f"- FlyBot branch: `{a.branch}`\n- FlyBot commit: `{a.commit}`\n- Built: {a.built}\n"
            f"- Asset: `{a.asset}`, sha256 `{a.sha}`, {a.size_mb} MB\n\n"
            f"The server downloads the asset and checks the sha256 before installing it.")
    Path(a.out).write_text(json.dumps({"tag_name": a.tag, "name": f"FlyBot {a.commit} (deploy build)", "body": body,
                                       "draft": False, "prerelease": True, "make_latest": "false"}), encoding="utf-8")
    return 0


def cmd_contract(a) -> int:
    d = load_json(a.file)
    problems = []
    if d.get("errorMessage"):
        problems.append(f"FunctionError: {str(d.get('errorMessage'))[:200]}")
    if d.get("v") != 1:
        problems.append("v != 1")
    total = (d.get("counts") or {}).get("total")
    if not isinstance(total, int):
        problems.append("counts.total missing")
    model = (d.get("model") or {}).get("name") or ""
    want = {"yolo26": "yolo26", "yolov7": "yolov7"}.get(a.detector, "")
    if want and want not in model:
        problems.append(f"model.name {model!r} is not the {a.detector} detector")
    for k in ("traceId", "boxes", "quality", "trace"):
        if k not in d:
            problems.append(f"{k} missing")
    if problems:
        print("contract check FAILED: " + "; ".join(problems))
        return 1
    ms = (d.get("timings") or {}).get("totalMs")
    print(f"contract ok: model {model} ({(d.get('model') or {}).get('version')}), {total} flies, "
          f"{len(d.get('proposals') or [])} proposals, flags {(d.get('quality') or {}).get('flags')}, {ms} ms")
    return 0


def cmd_latency(a) -> int:
    d = load_json(a.file)
    if not d:
        print("not measured")
        return 1
    c, w = d.get("cold", {}), d.get("warm", {})
    print(f"cold init {c.get('init_ms')} ms + {c.get('duration_ms')} ms, warm p50 {w.get('duration_ms_p50')} ms "
          f"(max memory {max(c.get('max_memory_mb') or 0, w.get('max_memory_mb') or 0)} MB)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("settings"); s.add_argument("file"); s.set_defaults(fn=cmd_settings)
    s = sub.add_parser("merge-app"); s.add_argument("--env", required=True); s.add_argument("--current")
    s.add_argument("--out", required=True); s.add_argument("--contact"); s.add_argument("--gcp"); s.set_defaults(fn=cmd_merge_app)
    s = sub.add_parser("gcp"); s.add_argument("--file"); s.add_argument("--current"); s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_gcp)
    s = sub.add_parser("get"); s.add_argument("file"); s.add_argument("key"); s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_get)
    s = sub.add_parser("ctx-hash"); s.set_defaults(fn=cmd_ctx_hash)
    s = sub.add_parser("contract"); s.add_argument("file"); s.add_argument("--detector", default="")
    s.set_defaults(fn=cmd_contract)
    s = sub.add_parser("latency"); s.add_argument("file"); s.set_defaults(fn=cmd_latency)
    s = sub.add_parser("ctx-zip"); s.add_argument("out"); s.set_defaults(fn=cmd_ctx_zip)
    s = sub.add_parser("pack-flybot"); s.add_argument("--src", required=True); s.add_argument("--out", required=True)
    s.add_argument("--meta"); s.add_argument("--exclude-media", action="store_true"); s.set_defaults(fn=cmd_pack_flybot)
    s = sub.add_parser("media-list"); s.add_argument("--src", required=True); s.set_defaults(fn=cmd_media_list)
    s = sub.add_parser("gh-release"); s.add_argument("file"); s.set_defaults(fn=cmd_gh_release)
    s = sub.add_parser("gh-prune"); s.add_argument("file"); s.add_argument("--keep", type=int, default=5)
    s.add_argument("--current", default=""); s.set_defaults(fn=cmd_gh_prune)
    s = sub.add_parser("gh-new-release")
    for k in ("--tag", "--commit", "--branch", "--built", "--asset", "--sha", "--size-mb", "--out"):
        s.add_argument(k, required=True)
    s.set_defaults(fn=cmd_gh_new_release)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
