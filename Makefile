# FlyBot on AWS: one entry point for the judge and the owner. Python lives in lambda/vision (uv), infra in infra/.
#   make setup test lint            Python env, tests, ruff (lambda/vision)
#   make models                     fetch the detector ONNX (SHA-256 checked) into models/
#   make lambda-local               POST http://localhost:9001/invoke (CONTRACTS.md section 1) for FlyBot local dev
#   make replay-sample              24 bundled trap photos through the handler -> data/replay/sample/
#   make build-lambda               build + push the arm64 image (on the FlyBot EC2 via SSM, see infra/build-image.sh)
#   make deploy / make destroy      ./deploy.sh / ./destroy.sh, settings in deploy.env (infra/DEPLOY_CHECKLIST.md)
#   make hooks                      git pre-commit hook: infra/ops/secrets-scan.sh --staged
#   make host CMD='...'             run a command on the FlyBot EC2 through SSM (logs, compose, bootstrap re-run)
#   make bench                      bench matrix (bench/run_matrix.sh)
.PHONY: setup test test-fast lint lock-lambda models lambda-local replay-sample build-lambda deploy destroy provision-camera bench cold-start host hooks
PROFILE ?= sentinel
REGION ?= ap-southeast-2
STAGE ?= dev
UV ?= uv
VISION := lambda/vision

setup:
	cd $(VISION) && $(UV) sync --group dev

test:
	cd $(VISION) && $(UV) run pytest -q

# handler + detector + agent tests only (the full suite runs the OpenCV ops on fixtures, ~5 min)
test-fast:
	cd $(VISION) && $(UV) run pytest -q tests/test_handler.py tests/test_detect_yolo26.py tests/test_agent_guardrails.py tests/test_agent_llm.py

lint:
	cd $(VISION) && $(UV) run ruff check .

# requirements-lambda.txt from uv.lock (no dev group, no boto3: the Lambda base image ships it)
lock-lambda:
	cd $(VISION) && $(UV) lock && $(UV) export --no-dev --no-hashes --no-emit-project --format requirements-txt \
	  | grep -v "^[[:space:]]*#" | grep -vi "^boto3\|^botocore\|^s3transfer\|^jmespath\|^python-dateutil\|^six\|^urllib3" \
	  | sed 's/^opencv-python==/opencv-python-headless==/' > requirements-lambda.body \
	  && (head -3 requirements-lambda.txt; cat requirements-lambda.body) > requirements-lambda.txt && rm requirements-lambda.body

# Detector weights (git-ignored): YOLOv7-tiny baseline; ARGS="--only detector" / "--check". YOLO26: models/COLAB_RUNBOOK.md
models:
	cd $(VISION) && $(UV) run python ../../models/fetch_models.py --root ../.. $(ARGS)

# Local stand-in for the Lambda (loopback only). ENV_FILE=.env loads ANTHROPIC_API_KEY etc.
lambda-local:
	cd $(VISION) && $(UV) run $(if $(ENV_FILE),--env-file ../../$(ENV_FILE),) python -m sentinel.lambda_local

# Offline check of the whole contract on the bundled sample (rule planner, no AWS, local evidence JPEGs)
replay-sample:
	cd $(VISION) && EVIDENCE_DIR=../../data/replay/sample/evidence $(UV) run python ../../eval/replay_sample.py \
	  --manifest ../../data/samples/samples_manifest.csv --out ../../data/replay/sample $(ARGS)

# Cold start + warm latency of the deployed function (needs the stack): writes bench/results/lambda_latency.json
cold-start:
	STAGE=$(STAGE) PROFILE=$(PROFILE) REGION=$(REGION) bash infra/measure-lambda.sh

build-lambda:
	STAGE=$(STAGE) PROFILE=$(PROFILE) REGION=$(REGION) bash infra/build-image.sh

# Everything from deploy.env (./deploy.sh): make deploy ARGS='--dry-run' / ARGS='--only health'
deploy:
	bash infra/deploy.sh $(ARGS)

# make destroy YES=1 [ARGS='--keep-photos']
destroy:
	bash infra/destroy.sh $(if $(filter 1,$(YES)),--yes,) $(ARGS)

provision-camera:
	STAGE=$(STAGE) PROFILE=$(PROFILE) REGION=$(REGION) bash infra/provision-camera.sh $(SERIAL) $(CERT)

bench:
	bash bench/run_matrix.sh --only b,c && $(UV) run --project $(VISION) python bench/summarize.py

# One command on the FlyBot host through SSM: make host CMD='tail -20 /var/log/flybot-bootstrap.log'
host:
	STAGE=$(STAGE) PROFILE=$(PROFILE) REGION=$(REGION) bash infra/host.sh '$(CMD)'

# Refuse commits that carry deploy.env, keys or secret-looking values
hooks:
	printf '#!/bin/sh\nexec bash infra/ops/secrets-scan.sh --staged\n' > .git/hooks/pre-commit && chmod +x .git/hooks/pre-commit
