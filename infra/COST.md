# Cost estimate: FlyBot on AWS (ap-southeast-2)

Two views: the demo stack running for a month, and the marginal cost per 1,000 analyzed photos. Unit prices are
the on-demand list prices for ap-southeast-2 as the author knows them (links in the table); they were **not
re-read on 16 Oct 2026** (the AWS session was expired), so verify them before quoting: the Price List API works
from this account (`aws pricing get-products --region us-east-1 --service-code AmazonEC2 ...`, used for the COOL
bench) and the real bill is in Billing > Cost Explorer. Usage figures are assumptions,
not measurements, unless a source file is named. The account has a USD 100 Free Plan credit; the Free Tier
allowances listed at the end reduce the real bill during the competition.

## Monthly, demo scale (2 EZVIZ cameras every 30 min in daylight + 2 ESP32 traps hourly ~= 1,500 photos/month)

| Item | Assumption | Unit price (Sydney) | USD / month | Source |
|---|---|---|---|---|
| EC2 t4g.small (FlyBot + MongoDB + Caddy) | 730 h | 0.0211 /h | 15.40 | https://aws.amazon.com/ec2/pricing/on-demand/ (ap-southeast-2, Linux) |
| EBS gp3 30 GB | 30 GB-month | 0.096 /GB-month | 2.88 | https://aws.amazon.com/ebs/pricing/ |
| Elastic IP | attached to a running instance | 0.005 /h (public IPv4) | 3.65 | https://aws.amazon.com/vpc/pricing/ (public IPv4 charge) |
| Lambda arm64 2048 MB | 1,500 invocations x 4 s x 2 GB = 12,000 GB-s (4 s assumed from the dev-laptop p50 of 3.8 s on the sample replay, `data/replay/sample/summary.json`; the Lambda itself is not measured until `make cold-start`) | 0.0000133334 /GB-s + 0.20 per 1M requests | 0.16 | https://aws.amazon.com/lambda/pricing/ (arm64) |
| Lambda cold starts | 1 per idle gap; assumed 100/month x 6 s | included above | ~0.16 | same |
| S3 photos + evidence | 1,500 photos x 0.3 MB + evidence 0.2 MB = 0.75 GB; 90-day lifecycle on evidence/ | 0.025 /GB-month, PUT 0.0055 /1k, GET 0.00044 /1k | 0.05 | https://aws.amazon.com/s3/pricing/ |
| ECR private | 1 image ~0.4 GB compressed (target <= 600 MB uncompressed) | 0.10 /GB-month | 0.04 | https://aws.amazon.com/ecr/pricing/ |
| IoT Core | 2 things, 1 connection each 24x7 (powered) = 2 x 43,200 min; ~3,000 messages/day shadow+telemetry; no rules by default (the 5 HTTPS rules exist only with DeviceEventsWebhookUrl) | 0.096 /1M connection-minutes; 1.20 /1M messages (first 1B); shadow 1.50 /1M ops | 0.03 + 0.11 + 0.14 = 0.28 | https://aws.amazon.com/iot-core/pricing/ |
| Kinesis Video Streams WebRTC | 2 channels, 20 live sessions x 5 min = 100 signaling minutes; TURN not needed when NAT allows direct | 0.03 /channel-month... + 2.25 per 1M signaling messages; 0.12 /1k TURN streaming minutes | 0.07 + ~0.01 | https://aws.amazon.com/kinesis/video-streams/pricing/ |
| Secrets Manager | 3 secrets (sentinel-dev-anthropic for the Lambda; flybot-<stage>-app and flybot-<stage>-gcp created by deploy.sh, read once per FlyBot start; +1 if flybot-<stage>-anthropic is needed) | 0.40 /secret-month + 0.05 /10k API calls | 1.20 | https://aws.amazon.com/secrets-manager/pricing/ |
| CloudWatch | 3 alarms, logs ~0.3 GB ingested | 0.10 /alarm, 0.67 /GB ingested (Sydney) | 0.30 + 0.20 | https://aws.amazon.com/cloudwatch/pricing/ |
| SNS email | < 1,000 notifications | free tier | 0.00 | https://aws.amazon.com/sns/pricing/ |
| Anthropic API (deep mode + LLM planner) | assumed 100 deep questions/month x (9k cached input + 2k input + 600 output tokens) on claude-haiku-5-5 | see https://www.anthropic.com/pricing (not an AWS cost; Bedrock blocked on this account) | not estimated here: run `uv run python -m sentinel.lambda_local` with the key and read `trace.tokens` | |
| **Total AWS** | | | **~ 24 USD / month** (EC2 + EIP + EBS are 92 %) | |

Free Tier (first 12 months / always free) that applies: Lambda 1M requests + 400k GB-s/month, S3 5 GB (12 months),
IoT Core 2.25M connection-minutes + 500k messages (12 months), CloudWatch 10 alarms + 5 GB logs, ECR 500 MB
(12 months). With them the month above is mostly the EC2 line.

## Per 1,000 photos (marginal, detect mode, 4 s per photo assumed on arm64 2048 MB)

| Item | USD per 1,000 photos |
|---|---|
| Lambda compute 1,000 x 4 s x 2 GB = 8,000 GB-s x 0.0000133334 | 0.107 |
| Lambda requests | 0.0002 |
| S3 PUT (photo by FlyBot + evidence by the Lambda) 2,000 x 0.0055/1k, GET 2,000 x 0.00044/1k, storage 0.5 GB for a month | 0.011 + 0.001 + 0.013 |
| Data transfer (EC2 -> S3 -> Lambda inside the region) | 0 |
| **Total** | **~ 0.13 USD** (0.00013 per photo) |

At 4 s per photo the 2048 MB function is CPU-bound by the DNN forward + SIFT registration; halving the memory would
roughly double the time at the same GB-s (not measured). The per-photo cost is dominated by compute: a 10x larger
farm (15,000 photos/month) adds ~2 USD/month.

Reference for the bench: the Roboflow hosted inference and the COOL c8g.xlarge (0.2076 USD/h,
bench/REPORT_COOL.md) columns use the same "per 1,000 photos" unit.
