# Responsible use

The rules FlyBot on AWS runs by, and their limits. Section 10 of the technical report ([REPORT.md](REPORT.md)) is the
short version. State: 20 October 2026. Sources are named in each item; "not measured" means nobody measured it.

## People in photos

- Trap cameras point into traps, but a person, a hand or a neighbour's house can enter the frame.
- Op 8 (`lambda/vision/sentinel/vision/motion.py`) flags large objects in front of the trap and suspected people.
  A photo with an occluder or a suspected person is **not counted** and gets at most one recapture proposal
  (`eval/RESULTS.md` section 4, case `occluder_hand`).
- The vision Lambda stores no picture. When op 8 suspects a person, it returns the face regions found by OpenCV's
  YuNet (`cv2.FaceDetectorYN`) as `privacy.regions`, or `privacy.blurAll` when the face model is missing
  (`docs/CONTRACTS.md` section 1a); FlyBot's Sentinel console blurs those regions on its own photo
  ([AnnotatedPhoto.vue](https://github.com/agrhub/flybot/blob/feat/aws/client/src/components/sentinel/AnnotatedPhoto.vue)). Other photo views in FlyBot show
  the photo as the camera took it. Face-detection accuracy on real people is **not measured**: no photo with a person
  exists in our data.
- **No photo is sent to an LLM.** The rule planner and the Claude Haiku 5.5 planner receive OpenCV's numbers and flags
  as text, never pixels; device names and on-screen text reach the LLM only quoted as untrusted text
  (`docs/DECISIONS.md`, `lambda/vision/sentinel/agent/guardrails.py`). FlyBot's chatbot (Gemini on Vertex AI) gets the
  Lambda's answer and counts through its `analyze_photo_deep` tool, not the photo.
- The camera that filmed a residential building instead of a trap (AC6527557) is excluded from every dataset,
  evaluation, replay and benchmark (`data/excluded_cameras.csv`). It is not in the bundled sample; the 24 sample
  photos were checked for people by eye and with op 8 and YuNet (`data/samples/README.md`).

## People decide

- The vision Lambda has **no actuator**. Every action it accepts (`capture_now`, `set_isp`, `set_interval`,
  `review_request`, `notify`) comes back to FlyBot as a proposal (`docs/CONTRACTS.md` section 1).
- FlyBot turns **every** proposal into a confirm card for the camera's owner, in the chat panel and on Telegram, even
  the ones `lambda/vision/sentinel/agent/policy.yaml` marks `requiresApproval: false`. Nothing runs before a person
  presses Confirm; an unanswered card expires after 24 hours and expiry never triggers an action
  ([sentinelService.js](https://github.com/agrhub/flybot/blob/feat/aws/server/src/services/sentinelService.js)).
- The chatbot's `propose_action` tool only shows a card ("nothing done yet"); the person confirms
  ([sentinelTools.js](https://github.com/agrhub/flybot/blob/feat/aws/server/src/agent/tools/sentinelTools.js)).
- **No autonomous spraying.** The system cannot apply a treatment. A surge raises a notification or a review; the
  product, the dose, the timing and compliance with local pesticide rules stay with the grower.
- A proposal must cite evidence from the same run, and an LLM proposal must be grounded in what the run observed;
  text burned into a photo or put in a device name cannot trigger one (failure injection, `eval/RESULTS.md` section 4).
- The kill switch (`SENTINEL_KILL_SWITCH=1` on the Lambda) keeps the analysis on and turns every proposal off
  (`lambda/vision/sentinel/agent/guardrails.py`).

## Guests and simulated data

- **Guest access** opens one shared, read-only account for two hours; the server refuses every write request from it
  except chatbot messages and sign-out (no settings, captures, sharing, deleting or approvals), each guest session has
  its own chatbot conversation, and turning the switch off ends every guest session at once
  ([guestAccess.js](https://github.com/agrhub/flybot/blob/feat/aws/server/src/services/guestAccess.js), [auth.js](https://github.com/agrhub/flybot/blob/feat/aws/server/src/middlewares/auth.js)).
- The **simulated farm network** shown to guests is labelled on every item: "SIM ·" trap names, "(simulated)" farm
  names, a banner on every page, a line in exported PDFs, and the number of flies drawn on each synthetic photo.
  Simulated data never enters emailed reports, alerts or webhooks, and one command removes all of it
  ([DEPLOY.md](https://github.com/agrhub/flybot/blob/feat/aws/DEPLOY.md)). Its photos are synthetic (`eval/synth`), stamped SIMULATED and contain no person.
- Synthetic results are never reported as field results: `eval/RESULTS.md` holds real photos only, and the synthetic
  checks are `eval/SYNTHETIC_RESULTS.md` and `eval/SYNTHETIC_AGENT_EVAL.md`, each headed "SIMULATED DATA".

## Cameras and power

- Guardrails inside each run: one recapture per event, at most 4 extra captures per camera per farm day, halved below
  40 % battery and none below 20 %, no device action in quiet hours (20:00-05:00 farm time)
  (`lambda/vision/sentinel/agent/policy.yaml`).
- Image-setting changes are proposed only for CamOS cameras (EZVIZ exposes none through FlyBot,
  `docs/CONTRACTS.md` section 1a).
- Self-registered ESP32-S3 cameras are limited: 3 cameras per user, 48 photos per camera per day, 200 unclaimed cameras
  in total; an unclaimed camera sends at most one photo an hour, its photos are kept 24 hours and it is deleted after
  7 days; LLM deep analysis is for claimed cameras only (`docs/CONTRACTS.md` section 2a).
- Each camera has its own certificate and can publish only on its own topics; it generates its key on the device and no
  secret is compiled into the public firmware (`docs/DECISIONS.md`, "Cameras and live view").
- Firmware updates: releases are checked against their `SHA256SUMS`; a camera downloads its image from a one-hour
  signed link; the battery guard holds an update below 40 % (or 3,700 mV) unless the camera is on mains or charging,
  and the camera also refuses on low battery; a camera that cannot connect with new firmware goes back to the old one
  ([firmwareService.js](https://github.com/agrhub/flybot/blob/feat/aws/server/src/services/firmwareService.js), [CHANGELOG.md](https://github.com/agrhub/flybot/blob/feat/aws/CHANGELOG.md) 1.6.0). An
  administrator can pause a camera: no photos, uploads or live view, and its owner sees why.
- The remote console log is off unless a camera's owner or an administrator turns it on, for at most an hour; the
  camera replaces the device token, Wi-Fi and access point passwords, SIM PIN, provisioning token and label code with
  `***` before sending ([PROTOCOL_ADDITIONS.md](https://github.com/agrhub/camos-esp32-trap/blob/main/docs/dev/PROTOCOL_ADDITIONS.md) section 11).
- Battery life, solar charging and 4G operation of the ESP32-S3 camera are **not measured**; its power budget is an
  estimate ([POWER_BUDGET.md](https://github.com/agrhub/camos-esp32-trap/blob/main/docs/dev/POWER_BUDGET.md)).

## Security

- Secrets live in Secrets Manager only; the server's environment file holds names and ARNs; the deploy scripts never
  print a secret value; `infra/ops/secrets-scan.sh` checks the history and every commit (README "Security and
  responsible use").
- Least-privilege IAM: the server invokes only its Lambda, reads only its buckets and manages only `camos-camera`
  things; browsers get 15-minute KVS viewer credentials for one channel; IMDSv2 is required; SSH is optional
  (`infra/template.yaml`, `docs/DECISIONS.md`).

## Data retention

| Data | Retention |
|---|---|
| Annotated evidence pictures | not written by the Lambda (only on request, e.g. the benchmark); the stack bucket's `evidence/` prefix expires after 90 days (`infra/template.yaml`) |
| Camera console logs / crash reports | 24 hours / 30 days ([cameraLogService.js](https://github.com/agrhub/flybot/blob/feat/aws/server/src/services/cameraLogService.js), [CameraCrash.js](https://github.com/agrhub/flybot/blob/feat/aws/server/src/models/CameraCrash.js)) |
| Guest sessions | 2 hours, never extended (see "Guests and simulated data") |
| Lambda build context in the artifacts bucket (`lambda/`) | 14 days (same file) |
| Lambda logs / CodeBuild logs (CloudWatch) | 30 days / 14 days (same file) |
| Photos of unclaimed cameras | 24 hours (`docs/CONTRACTS.md` section 2a) |
| Photos, traces and cards of claimed cameras (S3, MongoDB) | kept until the owner deletes the camera; FlyBot's plan-based retention is off by default; `./destroy.sh` removes the stack's bucket unless `--keep-photos` |

## Dataset and evaluation bias

- One farm, two EZVIZ lure-trap cameras, one month of photos, one region; the traps were empty from 7 October 2026.
- Sticky-card counting and tiled detection are measured only on synthetic cards; the synthetic flies come from the
  same dataset family as the training data, so field accuracy may be lower (`eval/SYNTHETIC_RESULTS.md`, "Limits").
- The public training set (camos/fruit-fly-wmcqi, CC BY 4.0) shows flies on white discs, not yellow sticky cards or
  lure-trap lids; YOLO26n misses the small flies on a full lure-trap view (`models/MODEL_CARD.md`).
- 0 human count labels and 0 expert verdicts (`eval/results/expert_review.json`, `eval/GROUND_TRUTH.md`): every
  accuracy number against FlyBot's production count is agreement between two models.
- Results do not transfer to other farms, trap types or pests without new labels. The model and the system are a
  monitoring aid, not a basis for pesticide decisions on their own.

## Licences and credits

This repository, the FlyBot app and the ESP32-S3 firmware are AGPL-3.0; the YOLO26n weights derive from Ultralytics
YOLO26 (AGPL-3.0); the dataset is CC BY 4.0, credit CamOS on Roboflow Universe (`LICENSE`, `NOTICE`,
`models/dataset.md`).
