# Deploy checklist: FlyBot on AWS

Everything runs from the `flybot-aws` repo root on the owner's laptop, in Git Bash (or any POSIX shell). Needed once:
AWS CLI v2, SAM CLI, `uv` (or any Python 3.9+), `git`, `curl`; `openssl` (ships with Git Bash) for the web-push keys.
Account 685527430355, region ap-southeast-2 only. No Docker and no SSH on the laptop.

## The four steps

```
git pull --ff-only                    # 0. the EC2 clones GitHub main: what you deploy is what is pushed
cp deploy.env.example deploy.env      # 1. once
#                                       2. fill deploy.env (below); it is git-ignored and the pre-commit hook refuses it
aws login --profile sentinel          # 3. browser login, 12 h
./deploy.sh                           # 4. ~35 min the first time, ~3 min when nothing changed
```

`make hooks` once installs the pre-commit check (`infra/ops/secrets-scan.sh --staged`).

### What to fill in deploy.env

| Setting | Needed for | If empty |
|---|---|---|
| `DOMAIN_NAME` | the web address (default `flybot.agrhub.com`) | default |
| `ACME_EMAIL` | Let's Encrypt expiry notices | `ALARM_EMAIL`, then admin@agrhub.com |
| `ALARM_EMAIL` | CloudWatch alarms | a topic with a confirmed email is reused (`sentinel-dev-alarms`) |
| `APP_KEY`, `SECRET_KEY` | the two EZVIZ traps | EZVIZ cameras cannot be added |
| `TELEGRAM_BOT_TOKEN`, `TELEGRAM_BOT_USERNAME` | Telegram alerts | off |
| `SMTP_HOST`, `SMTP_USER`, `SMTP_PASS` (+ `SMTP_PORT`, `MAIL_FROM`) | verification / reset / alert emails | emails go to the container log |
| `GCP_SERVICE_ACCOUNT_FILE` | the chatbot (Gemini on Vertex AI): path to the key JSON, outside the repo | chatbot off |
| `ANTHROPIC_API_KEY` | only if secret `sentinel-dev-anthropic` does not exist | deep mode uses the rule planner's template |
| `DETECTOR` | `yolo26` (default, needs `models/yolo26/`) or `yolov7` | yolo26, else yolov7 with a warning |

Generated once and kept in the secret: `JWT_SECRET`, the VAPID key pair. `OWNER_CIDR` defaults to the laptop's public IP.
Change a value later: edit deploy.env, run `./deploy.sh` again (only what changed is written; FlyBot restarts with it).

## What ./deploy.sh prints (in order)

1. `== preflight`: tool versions; `session ok: account 685527430355 (...)`; `region ap-southeast-2 allowed`; one
   `ok api ...` line per service (CloudFormation, IoT Core, Kinesis Video, Lambda, ECR, SSM, Secrets Manager, SNS, S3, IAM);
   `default VPC ...; t4g.small offered in ...`; `quotas: Elastic IPs n of 5 used; On-Demand vCPUs n of 16 used`;
   `GitHub main = local HEAD ...` (a WARN means you have unpushed changes the EC2 will not get).
   Session expired: the script prints `aws login --profile sentinel` and exits with code 2.
2. `== discover`: IoT endpoint `a38gyfo6dhqxf4-ats.iot.ap-southeast-2.amazonaws.com`, stack state, AMI, SSH source, alarm topic,
   photos bucket, secrets, detector `yolo26 fb4da5c05120`, `Lambda build inputs ctx-<hash>`.
3. `== models`: `uploaded models/yolo26/...` the first time, `up to date` afterwards; `anonymous GET ... -> 200`.
4. `== secrets`: key NAMES only (`set`, `generated`, `kept`, `default`, `empty`), then `flybot-dev-app written` or `unchanged`.
   `== repoaccess` (the FlyBot app `flybot/` is the private repo agrhub/flybot): the first time, **a box with a read-only deploy
   key**. Add it: github.com/agrhub/flybot > Settings > Deploy keys > Add deploy key, title `flybot-dev-ec2`, paste the key,
   leave "Allow write access" unchecked. The script waits (up to 30 min, polls every 20 s) until GitHub accepts it, then goes
   on; `--skip-wait` goes on at once (the EC2 retries for 30 min at boot). Later runs reuse the key (`flybot-dev-deploykey`).
   Instead of a deploy key: `FLYBOT_REPO_TOKEN` in deploy.env (fine-grained PAT, repository agrhub/flybot, Contents read-only).
5. `== stack`: SAM's changeset and `Successfully created/updated stack - flybot-dev` (~5 min), `stack ready: instance i-...`, and
   **a box with the DNS record**. Add it now, while the build runs:
   Namecheap > Domain List > agrhub.com > Manage > Advanced DNS > Add New Record: **Type A Record, Host `flybot`, Value
   `<Elastic IP>`, TTL Automatic**. (`./deploy.sh --show-ip` prints the box again at any time.)
6. `== image`: `in ECR already` or the SSM build (`waiting for the SSM agent`, then dots, 10-15 min the first time: it waits for
   the host's first boot to finish building FlyBot), `image <repo>@sha256:...`.
7. `== lambda`: SAM again, `Lambda flybot-vision-dev on ctx-<hash>`.
8. `== bootstrap`: `first boot: waiting for 'bootstrap done'` (new instance) or `re-running the bootstrap`; then
   `FlyBot containers up: caddy=running flybot=running mongo=running`.
9. `== health`: `DNS flybot.agrhub.com: 1.1.1.1=none 8.8.8.8=none, want <ip> (n/30 min)` every 30 s until the record is visible,
   then `https://flybot.agrhub.com/health -> ...` until Caddy has the certificate, then
   `valid certificate, /health 200, / 200, /api/device/time 401`.
10. `== smoke`: `contract ok: model yolo26n (2026-10-16), N flies, ...`.
11. `== coldstart`: `cold init ... ms + ... ms, warm p50 ... ms` (also in `bench/results/lambda_latency.json`).
12. `== summary`: URL, Elastic IP, instance, Lambda, detector + sha, IoT endpoint, KVS role, bucket, secrets, alarms, latency,
    cost (~24 USD/month, `infra/COST.md`), and **what is left for a human** (saved to `infra/.build/deploy-dev-summary.txt`).

If the DNS record was not visible within 30 minutes the run still finishes (summary says `DNS PENDING`); add the record and run
`./deploy.sh --only health`. Any failed step can be re-run alone: `./deploy.sh --only <step>`. Preview without changes:
`./deploy.sh --dry-run`. A first create that failed leaves the stack in ROLLBACK_COMPLETE: `./deploy.sh --yes` deletes and recreates
it (the photos bucket is retained).

## Left for a human, and why

| What | Why the script cannot |
|---|---|
| `aws login --profile sentinel` | browser sign-in of the owner (IAM Identity Center); no stored keys by design |
| The DNS A record at Namecheap | agrhub.com is not in Route 53; the owner keeps the registrar access |
| Add the deploy key on github.com/agrhub/flybot (first deploy only) | only a repo admin can grant access to the private app repo |
| Register the first account at `https://flybot.agrhub.com/auth/login` | the first account becomes the admin: it must be the owner, before the URL is shared |
| Click the SNS confirmation email (only for a new `ALARM_EMAIL`) | AWS requires the recipient to confirm |
| Email verification of that first account | the link arrives by email (SMTP set) or is in the log: `make host CMD='cd /opt/flybot/flybot-aws && docker compose logs --tail 200 flybot \| grep -A3 "mail not configured"'` |

## After the deploy

- Judges: `docs/flybot-aws-runbook.md` in the flybot repo, "How a judge logs in" (`emailVerified`, `sentinelViewer`, sharing).
- CamOS cameras: the public firmware registers itself (CONTRACTS 2a) and the owner claims it at `/claim`; the owner's own fleet
  can still be pre-provisioned with `make provision-camera SERIAL=... [CERT=...]`.
- Logs: `make host CMD='cd /opt/flybot/flybot-aws && docker compose logs --tail 100 flybot'`; Lambda: CloudWatch
  `/aws/lambda/flybot-vision-dev`.
- Remove: `./destroy.sh --yes` (then delete the `flybot-dev-ec2` deploy key on GitHub) (add `--keep-photos` to keep the photos for a later deploy, `--keep-secrets` to keep the secrets;
  secrets are otherwise recoverable for 7 days). The model bucket and `sentinel-dev` are never touched.
