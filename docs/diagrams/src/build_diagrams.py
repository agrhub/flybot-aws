"""Build the report and video diagrams as hand-laid-out SVG (no external tools).

    uv run --no-project python docs/diagrams/src/build_diagrams.py     # -> docs/diagrams/*.svg
    python docs/diagrams/src/render.py --png                         # -> docs/diagrams/*.png (Chrome headless)

Every number drawn here is copied from a cited file (the comment next to it, or the trace JSON it reads);
nothing is computed except the layout.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
ROOT = OUT.parent.parent

FONT = "Segoe UI, Arial, Helvetica, sans-serif"
INK, MUTED, LINE, PANEL = "#0f172a", "#475569", "#cbd5e1", "#f8fafc"
RED = "#b91c1c"
BADGE = {  # marker -> (fill, label)
    "COOL": ("#0f766e", "COOL"),
    "AGENT": ("#6d28d9", "AGENT"),
    "MCP": ("#1d4ed8", "MCP"),
    "HUMAN": ("#c2410c", "HUMAN APPROVAL"),
    "IOT": ("#15803d", "IOT CORE"),
    "FW": ("#78350f", "FIRMWARE"),
    "V5": ("#a16207", "OPENCV 5 NEW"),
}


def esc(s: str) -> str:
    return html.escape(s, quote=True)


class Svg:
    def __init__(self, w: int, h: int, title: str):
        self.w, self.h = w, h
        self.parts: list[str] = []
        self.title = title

    def add(self, s: str) -> None:
        self.parts.append(s)

    def text(self, x, y, s, size=15, weight="normal", fill=INK, anchor="start", italic=False, mono=False):
        st = ' font-style="italic"' if italic else ""
        ff = ' font-family="Consolas, Menlo, monospace"' if mono else ""
        self.add(f'<text x="{x:.0f}" y="{y:.0f}" font-size="{size}" font-weight="{weight}" fill="{fill}" '
                 f'text-anchor="{anchor}"{st}{ff}>{esc(s)}</text>')

    def pill(self, x, y, key, anchor="end"):
        fill, label = BADGE[key]
        w = 8.0 * len(label) + 18
        x0 = x - w if anchor == "end" else x
        self.add(f'<rect x="{x0:.0f}" y="{y:.0f}" width="{w:.0f}" height="22" rx="11" fill="{fill}"/>')
        self.text(x0 + w / 2, y + 15.5, label, size=12, weight="bold", fill="#ffffff", anchor="middle")
        return w

    def box(self, x, y, w, lines, title=None, badges=(), title_fill=INK, fill="#ffffff", stroke=LINE,
            size=15, lh=19, dashed=False, h=None):
        n = len(lines)
        hh = h or (16 + (24 if title else 0) + lh * n + 6)
        dash = ' stroke-dasharray="6 4"' if dashed else ""
        self.add(f'<rect x="{x:.0f}" y="{y:.0f}" width="{w:.0f}" height="{hh:.0f}" rx="8" fill="{fill}" '
                 f'stroke="{stroke}" stroke-width="1.5"{dash}/>')
        ty = y + 26
        if title:
            self.text(x + 12, ty, title, size=17, weight="bold", fill=title_fill)
            ty += 22
        for ln in lines:
            self.text(x + 12, ty, ln, size=size, fill=INK)
            ty += lh
        bx = x + w - 8
        for b in badges:
            bx -= self.pill(bx, y - 11, b) + 6
        return hh

    def arrow(self, pts, color=MUTED, width=2.5, dashed=False, head=True):
        d = "M " + " L ".join(f"{px:.0f} {py:.0f}" for px, py in pts)
        dash = ' stroke-dasharray="7 5"' if dashed else ""
        mk = (' marker-end="url(#ahr)"' if color == RED else ' marker-end="url(#ah)"') if head else ""
        self.add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"{dash}{mk}/>')

    def render(self) -> str:
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" '
                f'height="{self.h}" font-family="{FONT}">\n<title>{esc(self.title)}</title>\n'
                '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
                f'orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="{MUTED}"/></marker>'
                '<marker id="ahr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
                f'orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="{RED}"/></marker></defs>\n'
                f'<rect width="{self.w}" height="{self.h}" fill="#ffffff"/>\n' + "\n".join(self.parts) + "\n</svg>\n")


def legend(s: Svg, y: int, keys, notes):
    x = 30
    for k, note in zip(keys, notes):
        w = s.pill(x, y, k, anchor="start")
        s.text(x + w + 8, y + 16, note, size=14, fill=MUTED)
        x += w + 8 + 7.0 * len(note) + 24


# ---------------------------------------------------------------------------------------------- architecture

def architecture() -> str:
    s = Svg(1920, 1080, "FlyBot on AWS: architecture")
    s.text(30, 40, "FlyBot on AWS: architecture", size=28, weight="bold")
    s.text(30, 68, "Input → OpenCV 5 → Vision/Perception layer → AI/Agent layer → AWS infrastructure → "
           "Decision/Action; a confirmed action produces the next photo", size=18, fill=MUTED)
    cols = ["1  INPUT", "2  OPENCV 5", "3  VISION / PERCEPTION", "4  AI / AGENT", "5  AWS INFRASTRUCTURE",
            "6  DECISION / ACTION"]
    subs = ["trap cameras, grower questions", "opencv-python-headless 5.0.0", "lambda/vision/sentinel/vision",
            "sentinel/agent, same Lambda", "ap-southeast-2, one SAM stack", "a person confirms, FlyBot acts"]
    X = [30 + i * 314 for i in range(6)]
    W = 280
    for i, (c, sub) in enumerate(zip(cols, subs)):
        s.add(f'<rect x="{X[i]}" y="84" width="{W}" height="56" rx="8" fill="#1e293b"/>')
        s.text(X[i] + 12, 108, c, size=18, weight="bold", fill="#ffffff")
        s.text(X[i] + 12, 130, sub, size=14, fill="#cbd5e1")
        s.add(f'<rect x="{X[i]}" y="148" width="{W}" height="787" rx="8" fill="{PANEL}" stroke="#e2e8f0"/>')
        if i < 5:
            s.arrow([(X[i] + W + 4, 112), (X[i] + 310, 112)], width=3)
            s.arrow([(X[i] + W + 4, 540), (X[i] + 310, 540)], width=3)

    def col(i, items, gap=14, y0=166):
        y = y0
        for it in items:
            kw = it[3] if len(it) > 3 else {}
            y += s.box(X[i] + 8, y, W - 16, it[1], title=it[0], badges=it[2] if len(it) > 2 else (), **kw) + gap
        return y

    col(0, [
        ("EZVIZ trap cameras x 2", ["lure traps BE4089629, BE4111159", "in an orchard, southern Vietnam;",
                                    "FlyBot captures on schedule", "through EZVIZ Open Platform"]),
        ("ESP32-S3 trap camera", ["firmware-esp32 v0.6.2 (public)", "Wi-Fi or 4G, GPS, solar + LiPo,",
                                  "deep sleep; HTTPS photo upload,", "MQTT to IoT Core, KVS master"], ("FW",)),
        ("Grower in FlyBot", ["web app, chatbot question about", "a photo (\"why did it jump?\"),",
                              "confirm cards, Telegram"]),
        ("Replay (no hardware)", ["24 real photos in data/samples,", "make replay-sample: the same",
                                  "handler, rule planner, offline"]),
    ])
    col(1, [
        ("dnn, new engine", ["readNet(..., ENGINE_NEW),", "YOLO26n ONNX [1, 300, 6],", "NMSBoxes; the engine used",
                             "is recorded in every answer"], ("V5",)),
        ("imgproc", ["HSV inRange, contours,", "approxPolyDP, warpPerspective,", "Laplacian, histograms,",
                     "adaptiveThreshold, watershed"]),
        ("features", ["SIFT + FLANN + findHomography", "(USAC_MAGSAC) per photo;", "ALIKED + LightGlueMatcher",
                      "on demand"], ("V5",)),
        ("video + objdetect", ["MOG2, phaseCorrelate,", "FaceDetectorYN (YuNet) blur"]),
        ("COOL on Graviton4", ["AMI, OpenCV 5.1.0-dev, c8g;", "benchmarked on 500 real pairs,", "not in the serving path"],
         ("COOL",)),
    ])
    col(2, [
        ("Fly count (op 5)", ["YOLO26n + box rules,", "test F1 0.9006 (OpenCV path)"], ("V5",)),
        ("Quality gate (op 2)", ["blur, glare, dark, IR, trap", "missing; flags + 0-100 score"]),
        ("Trap find (op 1)", ["lure tray floor or sticky card"]),
        ("Registration + diff (ops 3-4)", ["new vs persisted boxes,", "card moved / changed"]),
        ("Crowding · density (ops 6-7)", ["card saturation, 4 x 4 grid"]),
        ("Occlusion + privacy (op 8)", ["hand / animal in front, camera", "moved, faces blurred"]),
        ("Output", ["contract JSON (CONTRACTS.md §1)", "+ annotated evidence JPEG"]),
    ], gap=11)
    col(3, [
        ("Planner", ["detect mode: rule planner", "(policy.yaml), deterministic", "deep mode: Claude Haiku 5.5",
                     "(Anthropic API), rule fallback"], ("AGENT",)),
        ("14 typed tools", ["analyze_trap_photo, history,", "capture, set_image, schedule,", "review, follow-up, proposals;",
                            "device tools return proposals"], ("AGENT",)),
        ("Guardrails", ["run-state whitelist, budgets,", "quiet hours, battery, evidence-", "grounded proposals, untrusted-",
                        "text filter, kill switch"], ("AGENT",)),
        ("Trace per photo", ["decisions, tool calls, tokens,", "timings → stored by FlyBot"]),
        ("FlyBot chatbot", ["Gemini (Vertex AI) with tools", "analyze_photo_deep and", "propose_action"], ("AGENT",)),
        ("MCP", ["none in this entry: tools are", "called in-process"], (), {"dashed": True}),
    ], gap=13)
    col(4, [
        ("EC2 t4g.micro + Elastic IP", ["FlyBot (Node 22, Vue 3),", "MongoDB 7, Caddy TLS:", "flybot.agrhub.com"]),
        ("Lambda arm64, 2 GB, 120 s", ["container from ECR, built by", "CodeBuild; one Invoke per photo"]),
        ("AWS IoT Core", ["thing type camos-camera, one", "policy, shadow, RPC, presence;", "role alias for KVS keys"],
         ("IOT",)),
        ("Kinesis Video Streams", ["WebRTC channel per camera,", "15-min viewer credentials"]),
        ("S3", ["photos, evidence (90 days),", "releases, build context"]),
        ("Ops", ["Secrets Manager, CloudWatch", "alarms → SNS, SSM, IAM,", "CloudFormation (SAM)"]),
    ], gap=13)
    col(5, [
        ("Proposals → confirm cards", ["capture_now, set_isp (CamOS),", "set_interval, review_request,", "notify; in FlyBot chat",
                                       "+ Telegram; expire after 24 h"], ("HUMAN",)),
        ("Confirmed: camera action", ["EZVIZ capture API, or CamOS", "RPC snapshot.now / shadow", "(image settings, schedule)"],
         ("IOT",)),
        ("New photo", ["→ Lambda again (the loop)"]),
        ("Chatbot answer", ["deep-mode markdown + its", "proposals as cards"], ("HUMAN",)),
        ("Alerts", ["FlyBot thresholds and surges:", "Telegram, email, web push"]),
        ("Live view", ["KVS WebRTC to the browser,", "on request"]),
    ], gap=13)
    s.arrow([(X[5] + 140, 935), (X[5] + 140, 952), (X[0] + 140, 952), (X[0] + 140, 939)], color=RED, width=3)
    s.text(960, 978, "LOOP: a confirmed capture produces a new photo, which is perceived again",
           size=17, weight="bold", fill=RED, anchor="middle")
    legend(s, 996, ["COOL", "AGENT", "MCP", "HUMAN", "IOT", "FW", "V5"],
           ["Cloud-Optimized OpenCV", "agent component", "none in this entry", "a person decides",
            "AWS IoT Core", "our device firmware", "OpenCV 5 feature"])
    s.text(30, 1052, "Columns 2-4 run inside one Lambda invocation per photo. Numbers: models/MODEL_CARD.md, "
           "bench/REPORT_COOL.md, infra/template.yaml.", size=14, fill=MUTED)
    return s.render()


# ---------------------------------------------------------------------------------------------- agent workflow

def agent_workflow() -> str:
    s = Svg(1920, 1080, "FlyBot on AWS: agent workflow")
    s.text(30, 46, "Agent workflow: perception → decision → action → new visual input", size=30, weight="bold")
    s.text(30, 78, "One guarded run per photo inside the vision Lambda; FlyBot turns its proposals into confirm "
           "cards and runs only what a person confirms", size=18, fill=MUTED)
    Y = 120
    W = 300
    xs = [30, 400, 770, 1140, 1510]
    s.box(xs[0], Y, W, ["EZVIZ schedule (FlyBot cron),", "ESP32-S3 upload (device API),", "confirmed capture_now,",
                        "chatbot question (deep mode)"], title="Photo or question")
    s.box(xs[1], Y, W, ["photo + previous photo (S3 keys),", "camera, trap kind, platform,", "battery, 14 days of counts;",
                        "Lambda.Invoke, timeout 120 s"], title="FlyBot request")
    s.box(xs[2], Y, W, ["analyze_trap_photo → OpenCV 5:", "count, quality, trap, diff,", "crowding, occlusion, faces;",
                        "evidence JPEG to S3"], title="PERCEPTION", title_fill="#0e7490", badges=("V5",))
    s.box(xs[3], Y, W, ["detect: rule planner", "deep: Claude Haiku 5.5 with tool", "use; error, refusal or timeout",
                        "→ rule planner, traced"], title="DECISION", title_fill="#6d28d9", badges=("AGENT",))
    s.box(xs[4], Y, 380, ["run-state whitelist, quiet hours", "20:00-05:00, capture budget,", "battery < 40 %: half, < 20 %: none,",
                          "evidence refs, kill switch"], title="GUARDRAILS check", title_fill="#6d28d9", badges=("AGENT",))
    for i in range(4):
        s.arrow([(xs[i] + W + 4, Y + 70), (xs[i + 1] - 4, Y + 70)], width=3)
    s.arrow([(xs[4] + 120, Y + 132), (xs[4] + 120, Y + 168), (xs[3] + 200, Y + 168), (xs[3] + 200, Y + 136)],
            color=RED, dashed=True)
    s.text(xs[3] + 215, Y + 190, "blocked: the reason goes back to the planner and into the trace", size=15, fill=RED)
    s.arrow([(xs[4] + 300, Y + 132), (xs[4] + 300, 368)], width=3)
    s.text(xs[4] + 312, 330, "allowed", size=15, weight="bold", fill=MUTED)
    AY = 372
    s.box(1140, AY, 750, ["capture → capture_now · set_image → set_isp (CamOS only) · set_capture_schedule →",
                          "set_interval · request_human_review → review_request · surge / threshold → notify",
                          "each with reason, confidence, requiresApproval (policy.yaml) and the evidence it cites;",
                          "the Lambda itself never touches a camera"],
          title="ACTION = proposal in the response", title_fill="#c2410c")
    s.box(30, AY, 1060, ["{traceId, model, counts, boxes, quality, diff, density, proposals[], answer,",
                         " trace {planner, decisions[{rule, action, reasons, evidence}], toolCalls[{name, args, status,",
                         " blocked_reason, state_before/after, ms}], tokens, fallback, outcome}, evidence, timings}",
                         " → FlyBot stores it on the photo; the trace is shown with the photo"],
          title="RESPONSE + TRACE (every photo)", title_fill=INK)
    s.arrow([(1140, AY + 70), (1094, AY + 70)], width=3)
    LY = 560
    lanes = [
        ("Confirm card → camera", ["FlyBot chat card + Telegram,", "Confirm / Reject, 24 h expiry", "→ EZVIZ capture, CamOS RPC",
                                   "or IoT shadow (settings,", "schedule) → new photo"], ("HUMAN",)),
        ("Chatbot (deep mode)", ["Gemini calls analyze_photo_deep", "→ Lambda deep mode → answer", "(numbers and flags as text,",
                                 "never pixels) → propose_action", "→ confirm card"], ("HUMAN",)),
        ("Alerts and review", ["FlyBot's own rules on the stored", "counts: thresholds, surges;", "review_request puts the photo",
                               "in front of a person"], ()),
    ]
    lx = [30, 400, 770]
    for (t, lines, b), x in zip(lanes, lx):
        s.box(x, LY, 340, lines, title=t, badges=b)
    s.arrow([(1500, AY + 120), (1500, 535), (200, 535), (200, LY - 6)], width=2.5)
    s.arrow([(570, 535), (570, LY - 6)], width=2.5)
    s.arrow([(940, 535), (940, LY - 6)], width=2.5)
    s.arrow([(130, LY + 143), (130, LY + 172), (14, LY + 172), (14, Y + 60), (27, Y + 60)], color=RED, width=3)
    s.text(22, LY + 205, "LOOP: new visual input", size=16, weight="bold", fill=RED)
    s.box(1140, 560, 750, [
        "1  ir → not counted, wait;  battery < 20 % → no extra capture, no ISP change",
        "2  glare / dark / overexposed on a CamOS camera → set_isp, then a verification capture",
        "3  blurry, or glare on EZVIZ (no ISP control) → capture again (once per event)",
        "4  occluder (op 8) → not counted, one recapture;  trap missing → recapture, then review",
        "5  sticky card full or changed → review (card change)",
        "6  two counts apart by > max(3, 20 %) (when a second count is given) → review",
        "7  surge (>= 5 flies, +3, x 1.5 vs 3 days ago, from the request's history) → notify /",
        "    review; no treatment is ever applied by the system",
    ], title="Rule planner order (lambda/vision/sentinel/agent/policy.yaml)", size=14, lh=19)
    SY = 830
    states = ["INGESTED", "ANALYZED", "DECIDED", "ACTED", "WAITING_APPROVAL", "FOLLOWUP"]
    notes = ["read tools only", "analyze ok", "planner decision", "proposal accepted",
             "no more device actions", "expired / decided"]
    s.text(30, SY - 12, "Run states (guardrail whitelist)", size=18, weight="bold")
    sx = 30
    for i, (st, nt) in enumerate(zip(states, notes)):
        w = 270 if st != "WAITING_APPROVAL" else 300
        s.add(f'<rect x="{sx}" y="{SY}" width="{w}" height="66" rx="33" fill="#eef2ff" stroke="#6366f1" '
              f'stroke-width="1.5"/>')
        s.text(sx + w / 2, SY + 28, st, size=16, weight="bold", fill="#3730a3", anchor="middle")
        s.text(sx + w / 2, SY + 50, nt, size=14, fill=MUTED, anchor="middle")
        if i < 5:
            s.arrow([(sx + w + 3, SY + 33), (sx + w + 37, SY + 33)], width=2.5)
        sx += w + 40
    s.text(30, 945, "Human control: every proposal from the Lambda becomes a confirm card, also the ones policy.yaml "
           "marks requiresApproval=false; nothing runs before a person presses Confirm.", size=16)
    s.text(30, 972, "Evidence: docs/evidence/agent/trace_example.json (real photos, clean → no_op, glare → capture_now), "
           "eval/RESULTS.md section 4 (failure injection).", size=16)
    legend(s, 1010, ["AGENT", "HUMAN", "V5"], ["agent component", "a person decides", "OpenCV 5 feature"])
    return s.render()


# ---------------------------------------------------------------------------------------------- trace example

def trace_example() -> str:
    t = json.loads((ROOT / "docs" / "evidence" / "agent" / "trace_example.json").read_text(encoding="utf-8"))
    seq = t["sequence_BE4089629"]
    glare = t["responses"]["glare_capture_now"]
    s = Svg(1920, 830, "FlyBot on AWS: a real perception, decision and action trace")
    s.text(30, 46, "Trace: OpenCV 5 output changes the next action (real photos, camera BE4089629, 5 Oct 2026)",
           size=28, weight="bold")
    s.text(30, 78, "The vision Lambda's handler on 12 consecutive photos of one lure trap (make replay-sample, "
           "YOLO26n, rule planner); source docs/evidence/agent/trace_example.json", size=18, fill=MUTED)
    # timeline table
    y0 = 112
    cols = [("photo (UTC)", 30), ("OpenCV count", 200), ("FlyBot count", 330), ("quality score", 460),
            ("glare share", 600), ("flags", 730), ("decision (rule)", 900), ("proposal", 1240), ("registration", 1400)]
    s.add(f'<rect x="20" y="{y0}" width="1880" height="34" rx="6" fill="#1e293b"/>')
    for name, x in cols:
        s.text(x, y0 + 23, name, size=16, weight="bold", fill="#ffffff")
    y = y0 + 34
    for i, r in enumerate(seq):
        hot = bool(r["proposals"])
        fill = "#fff7ed" if hot else ("#ffffff" if i % 2 == 0 else PANEL)
        s.add(f'<rect x="20" y="{y}" width="1880" height="30" fill="{fill}"/>')
        o = r["opencv"]
        reg = "-" if o["registrationOk"] is None else (f"ok, {o['inliers']} inliers" if o["registrationOk"] else "failed")
        vals = [r["capturedAt"][11:19], str(o["count"]), str(r["productionCount"]), f"{o['qualityScore']:.2f}",
                f"{o['glareRatio']:.4f}", ", ".join(o["qualityIssues"]) or "none", " + ".join(r["decision"]),
                ", ".join(r["proposals"]) or "-", reg]
        for (name, x), v in zip(cols, vals):
            s.text(x, y + 21, v, size=15, weight="bold" if hot and name in ("flags", "proposal") else "normal",
                   fill="#c2410c" if hot and name in ("flags", "decision (rule)", "proposal") else INK)
        y += 30
    s.text(30, y + 26, "Orange rows: op 2 sees glare, so the planner asks for a new photo; the same camera minutes "
           "earlier and later gets no_op. FlyBot's count is its production model's, not a human count.",
           size=15, fill=MUTED)
    # detail of the first glare photo
    dy = y + 52
    tr = glare["trace"]
    q = glare["quality"]
    sc = q["scores"]
    p = glare["proposals"][0]
    w = 600
    s.box(30, dy, w - 20, [
        f"photo {seq[5]['photo'].split('/')[-1]}, 768 x 432",
        f"op 5 YOLO26n: {glare['counts']['total']} flies, DNN {glare['timings']['dnnMs']} ms (laptop)",
        f"op 2: issues {q['issues']} → flag {q['flags']}, score {q['score']}",
        f"   glare share {sc['glare_ratio']}, clipped highs {sc['clip_high']}, sharpness {sc['sharpness']}",
        f"op 1: lure floor found, covers {sc['card_coverage']} of the frame",
        f"ops 3-4: registered ({glare['diff']['inliers']} SIFT inliers), new 0, gone 0",
        f"op 8: person suspected {glare['privacy']['person_suspected']}",
    ], title="PERCEPTION (OpenCV 5)", title_fill="#0e7490", badges=("V5",), size=15)
    s.box(30 + w, dy, w - 20, [
        f"planner: {tr['planner']}, fallback {str(tr['fallback']).lower()}, LLM calls {tr['tokens']['llmCalls']}",
        f"1 {tr['decisions'][0]['rule']} → {tr['decisions'][0]['action']}",
        f"   \"{tr['decisions'][0]['reasons'][0]}\"",
        f"2 {tr['decisions'][1]['rule']} → {tr['decisions'][1]['action']}",
        "   the new photo arrives as a new event",
        f"event: photo_not_counted, why {tr['events'][0]['detail']['why']}",
        f"outcome: {tr['outcome']}",
    ], title="DECISION (agent)", title_fill="#6d28d9", badges=("AGENT",), size=15)
    calls = tr["toolCalls"]
    s.box(30 + 2 * w, dy, 1890 - (30 + 2 * w), [
        (f"tool 0 {calls[0]['name']}: {calls[0]['status']}, {calls[0]['state_before']} → {calls[0]['state_after']}, "
         f"{calls[0]['ms']} ms"),
        f"tool 1 {calls[1]['name']}({calls[1]['args']['purpose']}, {calls[1]['args']['resolution']}): {calls[1]['status']},",
        f"   {calls[1]['state_before']} → {calls[1]['state_after']}, transport {calls[1]['result']['transport']}",
        f"proposal {p['action']}: confidence {p['confidence']}, requiresApproval {str(p['requiresApproval']).lower()}",
        "FlyBot: a confirm card + Telegram; Confirm runs the",
        "EZVIZ capture → a new photo → this loop again",
        f"total {glare['timings']['totalMs']} ms on the laptop (Lambda warm p50 1,091 ms)",
    ], title="ACTION (guarded tool calls)", title_fill="#c2410c", badges=("HUMAN",), size=15)
    s.text(30, 782, "Not measured: whether these glare flags are right (no human look) and the confirm-to-new-photo "
           "time on AWS. The replay cannot take photos: the 07:17-07:35 photos are the camera's own.", size=15, fill=MUTED)
    s.text(30, 806, "Lambda warm p50: bench/results/lambda_latency.json. Small specks on the lid are missed by "
           "YOLO26n (models/MODEL_CARD.md), so FlyBot's production count is higher here.", size=15, fill=MUTED)
    return s.render()


def main() -> None:
    (OUT / "architecture.svg").write_text(architecture(), encoding="utf-8", newline="\n")
    (OUT / "agent_workflow.svg").write_text(agent_workflow(), encoding="utf-8", newline="\n")
    (OUT / "trace_example.svg").write_text(trace_example(), encoding="utf-8", newline="\n")
    print("wrote", ", ".join(p.name for p in sorted(OUT.glob("*.svg"))))


if __name__ == "__main__":
    main()
