# Contracts between the parts (v1, 16 Oct 2026)

Three teams build in parallel: FlyBot (../ezviz-manager, branch feat/aws), the vision Lambda (lambda/vision + infra), and the ESP32 firmware (../camos-esp32-trap). These contracts are the only coupling. Change them only by editing this file first and noting the change in docs/DECISIONS.md.

## 1. FlyBot -> vision Lambda (synchronous)
Transport: AWS SDK `Lambda.Invoke` (RequestResponse) with function name from env `VISION_LAMBDA_NAME` on AWS; for local development the same JSON over `POST http://localhost:9001/invoke` served by `uv run python -m sentinel.lambda_local` (no auth, loopback only). Timeout 120 s. Payload limit 6 MB, so photos travel by reference, never inline.

Request:
```json
{
  "v": 1,
  "mode": "detect",                       // "detect" (every photo) | "deep" (chatbot question)
  "photo": {"s3Bucket": "flybot-photos-...", "s3Key": "tenants/t1/cams/BE4089629/2026-10-16T08-00-00Z.jpg"},
                                          // or {"url": "https://..."} for local dev / EZVIZ CDN
  "prevPhoto": {"s3Bucket": "...", "s3Key": "..."},   // optional, previous photo of the same camera
  "camera": {"id": "BE4089629", "name": "Củ Chi", "trapKind": "lure", "platform": "ezviz",
             "location": {"lat": 10.97, "lng": 106.49}},        // trapKind: "sticky" | "lure"; platform: "ezviz" | "camos" | "kinesis"
  "capturedAt": "2026-10-16T08:00:00Z",
  "history": [{"capturedAt": "2026-10-15T08:00:00Z", "count": 12}],   // optional, up to 14 days
  "question": "Why did the count jump today?",   // deep mode only
  "locale": "en"                                 // "vi" | "en", for `answer` and `reason` strings
}
```
Response (HTTP 200 / no FunctionError):
```json
{
  "v": 1,
  "traceId": "tr_20261016_080012_ab12",
  "model": {"name": "yolo26n", "version": "2026-10-18", "engine": "opencv-5.0.0-dnn-new"},
  "counts": {"total": 14, "byClass": {"fruit_fly": 14}},
  "boxes": [{"x": 0.12, "y": 0.33, "w": 0.04, "h": 0.05, "score": 0.91, "cls": "fruit_fly"}],   // normalised to the photo
  "quality": {"ok": true, "flags": [], "blur": 0.12, "exposure": 0.48},     // flags: "blurry" | "dark" | "overexposed" | "occluded" | "camera_moved" | "trap_missing"
  "diff": {"newCount": 3, "goneCount": 1, "cardMoved": false},             // only when prevPhoto was given
  "density": {"gridRows": 4, "gridCols": 4, "cells": [[0, 1, 0, 2], ...]}, // optional
  "proposals": [
    {"id": "p1", "action": "capture_now", "params": {}, "reason": "Count doubled since yesterday", "confidence": 0.8, "requiresApproval": true}
  ],
  // action: "capture_now" | "set_interval" {"minutes": n} | "set_isp" {"brightness": n, "wdr": bool} | "review_request" {} | "notify" {"message": s}
  "answer": "markdown text, deep mode only",
  "trace": {"planner": "rule|llm", "steps": [{"tool": "detect", "ms": 410}], "tokens": {"in": 0, "cachedIn": 0, "out": 0}, "ms": 1320},
  "evidence": {"annotatedS3Key": "evidence/BE4089629/2026-10-16T08-00-00Z.annotated.jpg", "crops": []},
  "timings": {"totalMs": 1320, "dnnMs": 410}
}
```
Error: `{"v": 1, "error": {"code": "photo_not_found" | "bad_request" | "model_unavailable" | "internal", "message": "..."}}` with a Lambda FunctionError (or HTTP 4xx/5xx locally). FlyBot keeps the photo, marks the analysis failed, and retries once after 30 s.

FlyBot stores `counts`, `boxes`, `quality`, `diff`, `proposals`, `traceId`, `trace`, `model`, `evidence` on the snapshot document, shows proposals as confirm cards, and emits the `snapshot_analyzed` webhook unchanged.

### 1a. Implementation notes (vision Lambda, 16 Oct 2026; additive, section 1 unchanged)
- Errors: the Lambda raises, so `Invoke` returns `FunctionError: Unhandled` and the payload is Lambda's `{"errorMessage": "<the contract JSON above as a string>", "errorType": "PhotoNotFound|BadRequest|ModelUnavailable|Internal", ...}`: FlyBot reads `JSON.parse(payload.errorMessage).error.code`. Locally (`uv run python -m sentinel.lambda_local`) the same JSON is the HTTP body with status 400 / 404 / 503 / 500.
- Additive response fields (ignore when unknown): `quality.issues` (raw op 2 issue names beside the enumerated `flags`), `quality.score`, `quality.scores`; `diff.persistedCount|cardChangeSuspected|meaning|registrationOk|inliers|changeRatio|skipped`; `proposals[].detail|rule|evidence|toolCall|by`; `trace.decisions|events|toolCalls|reasoning|model|fallback|llmTurns|state|outcome`; `evidence.annotatedPath` (local runs); `crowd.saturation` (sticky); `privacy`; `photo{width,height}`; `opencv`; `mode`; `warnings[]`.
- `proposals[].params` per action: `capture_now {purpose, resolution}`, `set_isp {brightness?, contrast?, wdr: bool?, flicker?}`, `set_interval {minutes, from, to}`, `review_request {kind: review|card_change|treatment, ...}`, `notify {kind, at, message}`. `requiresApproval` and `confidence` are policy values (`lambda/vision/sentinel/agent/policy.yaml`), not measurements.
- A missing or unreadable `prevPhoto` is not an error: `diff` is null and `warnings` says so. `camera.trapKind` defaults to `lure`. Optional request extras the planner uses: `camera.battery` (percent), `camera.isp` ({brightness, contrast, wdr, flicker} of a CamOS camera), `camera.alertThreshold`, `photoId` (echoed as `review_request.params.snapshotId`). Unknown keys and malformed `history` rows are ignored, never a `bad_request`.
- `set_isp` is proposed only for `camera.platform == "camos"` (EZVIZ cameras expose no ISP through FlyBot, `kinesis` has no control channel); the other actions apply to every platform.
- `evidence.annotatedS3Key` is written to the bucket the photo came from (`photo.s3Bucket`), so FlyBot serves it from its own bucket (`GET /api/sentinel/ui/evidence?key=`); for a `photo.url` request it goes to the Lambda's `PHOTOS_BUCKET`. The response also carries `evidence.bucket`.

## 2. ESP32 camera -> FlyBot device API (HTTPS)
Base URL and device token come from pairing (protocol v1 section 12). Header on every call: `X-Device-Token: <token>`; `X-Device-Serial: <serial>`.
- `POST /api/device/snapshot` multipart/form-data: part `image` (image/jpeg, <= 2 MB), part `meta` (application/json): `{"v":1,"serial":"ESP32S3-0001","capturedAt":"2026-10-16T08:00:00Z","trigger":"schedule|rpc|boot","battery":{"mv":3900,"pct":72,"charging":true},"net":"wifi|4g","rssi":-61,"location":{"lat":..,"lng":..,"fixAt":"..."},"fw":"0.1.0"}`. Response `{"ok":true,"snapshotId":"...","nextWakeAt":"2026-10-16T09:00:00Z"}` (nextWakeAt optional hint). 413 when too large, 401 on bad token.
- `GET /api/device/time` -> `{"now":"2026-10-16T08:00:00Z","epoch":1792137600}` (fallback when NTP is blocked on 4G).
- `GET /api/device/ota` -> `{"fw":"0.2.0","url":"https://...bin","sha256":"..."}` or 204.
- `POST /api/device/pair` body `{"serial":"...","verifyCode":"...","provisionToken":"..."}` -> `{"ok":true,"deviceToken":"...","iot":{"endpoint":"a38g...-ats.iot.ap-southeast-2.amazonaws.com","thingName":"..."},"kvs":{"region":"ap-southeast-2","channelName":"cam-<serial>"}}` (the camera calls this once after the portal provisions it; the IoT certificate is installed through the portal or fleet provisioning, see firmware docs).

## 2a. Zero-touch self-registration (public firmware, v1 addition, 16 Oct 2026)
Goal: anyone flashes the public release (GitHub Releases or the browser flasher), the camera joins the demo at https://flybot.agrhub.com by itself, and the person claims it in FlyBot. No secret is compiled into the public image; FlyBot is the only party that talks to AWS IoT.

1. **Identity on the device.** First boot: serial = `ESP` + factory MAC (as today); the device generates a 6-character verify code (Crockford base32, from the hardware RNG) once and keeps it in NVS; the portal page, the serial console and the LED-free label QR (`CAMOS:1;SN:<serial>;VC:<code>;M:<model>;;`) show it. Factory reset keeps serial and code.
2. **Register** (no auth, HTTPS with the bundled public CA list): `POST /api/device/register` body `{"v":1,"serial":"ESPF85B1B92877C","verifyCode":"9WBYWY","model":"AIC-FREENOVE-S3CAM","fw":"0.3.0","csr":"<PEM PKCS#10, ECDSA P-256, generated on the device; the private key never leaves NVS>"}`.
   - FlyBot rate-limits (per IP and globally), checks the serial format and that the serial is not already registered with a different verify code (409 `serial_taken`), signs the CSR through AWS IoT (`CreateCertificateFromCsr`), creates thing `<serial>` of type `camos-camera`, attaches the device policy and the certificate, creates the KVS channel `cam-<serial>`, stores the camera as **unclaimed** with a hashed device token, and returns once:
   - `201 {"ok":true,"certificatePem":"...","deviceToken":"...","iot":{"endpoint":"a38gyfo6dhqxf4-ats.iot.ap-southeast-2.amazonaws.com","thingName":"ESPF85B1B92877C"},"kvs":{"region":"ap-southeast-2","channelName":"cam-ESPF85B1B92877C"},"claimUrl":"https://flybot.agrhub.com/claim?sn=ESPF85B1B92877C","claimed":false}`.
   - Re-register with the same serial + verify code before the first MQTT connect returns the same certificate (idempotent); after that 409 unless the owner resets the camera in FlyBot.
3. **Unclaimed mode** (device connected, no owner yet): the camera publishes state/telemetry and may upload at most 1 snapshot per hour (FlyBot answers `nextWakeAt` accordingly); photos of unclaimed cameras are analysed with the rule planner only (no LLM), kept 24 h, visible to nobody but the claim preview. Unclaimed cameras that are never claimed are deleted (thing, certificate, channel, photos) after 7 days.
4. **Claim** (signed-in FlyBot user): scan the label QR or type serial + verify code on `/claim` -> `POST /api/cameras/claim {"serial","verifyCode","farmId?","trapKind"}` -> camera moves to the user's tenant, shadow `claimed: true` + the user's capture interval; the device gets the full schedule at its next connect. Wrong code: 404 without saying whether the serial exists; 5 failures per serial per hour lock the serial for an hour.
5. **Demo limits** for self-registered cameras (env, defaults): `DEMO_MAX_CAMERAS_PER_USER=3`, `DEMO_MAX_SNAPSHOTS_PER_DAY=48` per camera, `DEMO_MAX_UNCLAIMED=200` total, LLM deep analysis allowed for claimed cameras only, live view 5 min per request.
6. **Time, timezone and IP are automatic** (no portal input needed; the portal shows the detected values with a manual override):
   - IP: DHCP on Wi-Fi, PPP on 4G (the device never asks for an address); the device reports `net.ip` (local) and FlyBot echoes the public address it saw.
   - Clock: SNTP (`pool.ntp.org`, `time.google.com`) -> modem network time (`AT+CCLK?` with NITZ, 4G) -> GPS time -> `GET /api/device/time`; the first source that answers wins, RTC kept across deep sleep.
   - Timezone: `GET /api/device/time` (works before registration, no auth) returns `{"now":"2026-10-16T08:00:00Z","epoch":1792137600,"timezone":"Asia/Ho_Chi_Minh","posix":"<+07>-7","tzSource":"farm|gps|ip|default","publicIp":"115.72.212.239"}`. FlyBot picks the timezone in this order: the claiming user's farm timezone, the camera's last GPS fix (`location` in the shadow, lookup via geo-tz), IP geolocation of the request (offline GeoLite-style database bundled with FlyBot, no external call), default `Asia/Ho_Chi_Minh`. `posix` is the POSIX TZ string the ESP-IDF `setenv("TZ")` needs (FlyBot holds the IANA -> POSIX table). The device re-reads it at every boot and when the shadow's `tz` changes.
7. **Owner path stays**: `tools/factory.*` (pre-provisioned certificate in NVS) and the existing `POST /api/device/pair` with a provisioning token keep working for the owner's own fleet.

## 3. ESP32 camera <-> FlyBot over AWS IoT Core (MQTT, protocol v1)
Topics, shadow and RPC exactly as docs/protocol-v1.md in camos-esp32-trap (CamOS v1). Additions for this project (v1 additions, never redefining existing fields):
- Shadow `desired`/`reported` keys: `capture.intervalMin` (int), `power.mode` ("auto"|"battery"|"powered"), `live.requested` (bool, FlyBot sets true to ask for a live session at the next wake), `kvs.channelName` (string), `location` (reported: `{lat,lng,fixAt}`), `net` (reported: "wifi"|"4g"), `battery` (reported: `{mv,pct,charging,solarMv}`).
- RPC methods: `snapshot.now`, `live.start` `{"minutes": 5}`, `live.stop`, `gps.fix`, `reboot`, `ota.apply`.
- LWT / `goodbye` carries `sleeping: true, wakeAt: <ms epoch>` so FlyBot can show "sleeping, next wake HH:MM".
- FlyBot talks to IoT Core with the AWS SDK (shadow and publish through the data plane; device->cloud messages through an MQTT-over-WebSocket client with SigV4 from the instance role, or IoT rules to an HTTPS endpoint if the client proves unreliable; decision recorded in DECISIONS.md).

## 4. Live view (KVS WebRTC)
FlyBot creates one KVS signaling channel per CamOS camera (`cam-<serial>`) and writes the name into the shadow. The camera joins as MASTER when `power.mode` is powered, or for `live.start`/`live.requested` minutes otherwise, and only encodes when a VIEWER is connected. FlyBot's `GET /api/cameras/:id/live` returns `{"platform":"kinesis","channelARN":"...","region":"...","signalingEndpoints":{...},"iceServers":[...],"clientId":"...","expiresAt":"..."}` for the browser (temporary credentials scoped to the viewer role, 15 min). EZVIZ cameras keep their existing live URL shape.
