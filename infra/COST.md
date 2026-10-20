# Cost: FlyBot on AWS (ap-southeast-2, Sydney)

Two views: the demo deployment running for a month, and the marginal cost per 1,000 analyzed photos. Prices are
on-demand list prices for ap-southeast-2. The EC2 t4g.micro and CodeBuild prices were read from the AWS Price List API
on 2026-10-17 (`infra/prices.json`); the other unit prices are the published list prices (links in the table) and
should be checked before quoting. Usage figures are assumptions unless a source file is named. The real bill is in
Billing > Cost Explorer; the stack's resources carry the tags `project=flybot-aws` and `stage=<stage>`.

## Monthly, demo scale

Assumption: 2 EZVIZ cameras every 30 min in daylight + 2 ESP32 traps hourly, about 1,500 photos per month.

| Item | Assumption | Unit price (Sydney) | USD / month | Source |
|---|---|---|---|---|
| EC2 t4g.micro (FlyBot + Caddy + MongoDB, no Docker) | 730 h | 0.0106 /h | 7.74 | `infra/prices.json` |
| EBS gp3 30 GB (system, MongoDB data, two FlyBot releases) | 30 GB-month | 0.096 /GB-month | 2.88 | https://aws.amazon.com/ebs/pricing/ |
| Public IPv4 (the Elastic IP) | attached, 730 h | 0.005 /h | 3.65 | https://aws.amazon.com/vpc/pricing/ |
| Lambda arm64 2048 MB | 1,500 invocations x 1.09 s x 2 GB = 3,270 GB-s, plus ~100 cold starts x 4 s | 0.0000133334 /GB-s + 0.20 /1M requests | 0.06 | warm p50 1,087 ms, cold billed 3,965 ms: `bench/results/lambda_latency.json` |
| CodeBuild ARM small (Lambda image) | 2 builds/month x 10 min | 0.0045 /min | 0.09 | `infra/prices.json` |
| S3 photos + evidence + releases | 0.75 GB photos/evidence (90-day lifecycle on evidence/), ~60 MB FlyBot releases | 0.025 /GB-month + requests | 0.05 | https://aws.amazon.com/s3/pricing/ |
| ECR private | 1-5 images, ~0.4 GB each compressed | 0.10 /GB-month | 0.04-0.20 | https://aws.amazon.com/ecr/pricing/ |
| IoT Core | 2 things connected 24x7, ~3,000 messages/day | connection minutes + messages + shadow ops | 0.28 | https://aws.amazon.com/iot-core/pricing/ |
| Kinesis Video Streams WebRTC | 2 channels, ~20 live sessions x 5 min | per channel-month + signaling | 0.08 | https://aws.amazon.com/kinesis/video-streams/pricing/ |
| Secrets Manager | 3-4 secrets (app, gcp, Anthropic) | 0.40 /secret-month | 1.20-1.60 | https://aws.amazon.com/secrets-manager/pricing/ |
| CloudWatch | 3 alarms, ~0.3 GB logs | 0.10 /alarm, 0.67 /GB ingested | 0.50 | https://aws.amazon.com/cloudwatch/pricing/ |
| SNS email | < 1,000 notifications | free tier | 0.00 | https://aws.amazon.com/sns/pricing/ |
| **Total AWS** | | | **~ 17 USD / month** (EC2 + public IPv4 + EBS are 85 %) | |

Not included: the Anthropic API (deep-mode answers and the optional LLM planner; billed by Anthropic per token,
`trace.tokens` in each Lambda response shows the usage), Google Vertex AI for the chatbot, and an external
MongoDB / S3 service if you configure one in deploy.env. Free Tier allowances (Lambda 1M requests + 400k GB-s per
month, CloudWatch 10 alarms + 5 GB logs, and the 12-month offers for S3, IoT Core and ECR) reduce the bill further.

The previous host layout (t4g.small with Docker, FlyBot built on the server) was ~24 USD/month: the prebuilt release
on a t4g.micro saves ~7.7 USD/month.

## Per 1,000 photos (marginal, detect mode)

| Item | USD per 1,000 photos |
|---|---|
| Lambda compute: 1,000 x 0.98 s x 2 GB = 1,960 GB-s x 0.0000133334 | 0.026 |
| Lambda requests | 0.0002 |
| S3: 1,000 PUT (the photo; the Lambda stores no picture) x 0.0055/1k, 2,000 GET x 0.00044/1k, 0.5 GB stored for a month | 0.0055 + 0.001 + 0.013 |
| Data transfer inside the region (EC2 -> S3 -> Lambda) | 0 |
| **Total** | **~ 0.046 USD** (0.000046 per photo) |

Source of the duration: warm p50 of 5 calls with a sample photo pair, 2048 MB arm64 (`bench/results/lambda_latency.json`,
measured by `./deploy.sh`'s coldstart step). The function is CPU-bound by the DNN forward and the SIFT registration;
a 10x larger farm (15,000 photos per month) adds about 0.7 USD per month. The COOL / Graviton / x86 comparison of the
same pipeline on EC2 uses the same unit: `bench/REPORT_COOL.md`.
