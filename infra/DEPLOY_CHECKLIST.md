# Deploy checklist

Run from the `flybot-aws` folder. macOS / Linux / Git Bash: `./deploy.sh`; Windows cmd or PowerShell: `deploy.bat`
(same options). Needed once: AWS CLI v2, SAM CLI, uv (or Python 3.9+), Node.js 22.12+ with npm, git, curl, and a
FlyBot checkout next to this folder (`FLYBOT_SRC`, default `../ezviz-manager`).

## 1. First deploy

```
cp deploy.env.example deploy.env      # fill it: see the table in README.md ("deploy.env")
aws login --profile sentinel
./deploy.sh --dry-run                 # optional: real checks, read-only, prints the plan
./deploy.sh
```

What you see, in order: `== preflight` (one `ok` line per check), `== discover`, `== models`, `== secrets` (key names
only), `== stack` (SAM changeset, ~5 min) and **a box with the DNS record**, `== flybot` (npm build here, upload),
`== image` (CodeBuild, 6-12 min the first time), `== lambda`, `== bootstrap` (`host: flybot=active caddy=active
mongod=active release=<commit>`), `== health`, `== smoke` (`contract ok: model yolo26n ...`), `== coldstart`,
`== summary`.

Then, by hand:

1. **DNS**: add the record from the box at your DNS provider. agrhub.com on Namecheap: Domain List > agrhub.com >
   Manage > Advanced DNS > Add New Record, Type A Record, Host `flybot`, Value the Elastic IP, TTL Automatic.
   `./deploy.sh --show-ip` prints it again; `./deploy.sh --only health` checks DNS and HTTPS later. Caddy gets the
   certificate by itself once the name resolves.
2. **First account**: register at `https://<DOMAIN_NAME>/auth/login`; the first account becomes the admin, so do it
   before sharing the URL. Without SMTP, the verification link is in the server log:
   `make host CMD='journalctl -u flybot -n 200 --no-pager | grep -A3 "mail not configured"'`.
3. **Alarm email**: only for a new `ALARM_EMAIL`, click the confirmation link AWS SNS sent.
4. **External MongoDB** (`MONGO_URI`): allow the Elastic IP in the database's IP access list (MongoDB Atlas: Network
   Access).

## 2. Every later change

Edit `deploy.env` (or commit a FlyBot change in `FLYBOT_SRC`), then `./deploy.sh`. Only what changed is redone: a new
FlyBot commit is built and installed (the previous release is kept), new secrets restart FlyBot, a Lambda code change
rebuilds the image. One step only: `./deploy.sh --only <step>` (`models secrets stack flybot image lambda bootstrap
health smoke coldstart summary`).

- Back to the previous FlyBot release: `./deploy.sh --rollback`.
- A test build from uncommitted FlyBot changes: `./deploy.sh --only flybot --allow-dirty && ./deploy.sh --only bootstrap`.
- A command on the server without SSH: `make host CMD='systemctl status flybot --no-pager'` (or `bash infra/host.sh '...'`).
- Logs: `make host CMD='journalctl -u flybot -n 100 --no-pager'`, `/var/log/flybot-bootstrap.log`; Lambda and builds in
  CloudWatch (`/aws/lambda/flybot-vision-<stage>`, `/aws/codebuild/flybot-vision-build-<stage>`).

## 3. Moving a stack from the Docker layout to the prebuilt release

A stack deployed before October 17, 2026 runs FlyBot, MongoDB and Caddy in Docker on a t4g.small. One `./deploy.sh`
moves it:

1. the stack update changes the instance type to t4g.micro and the user-data; CloudFormation stops and starts the
   instance (the Elastic IP, the disk and its data stay; FlyBot is down from here until step 3, about 15-20 minutes);
2. the FlyBot release is built on your computer and uploaded, the Lambda image is built by CodeBuild;
3. the bootstrap step runs the new user-data: it dumps the MongoDB container (`mongodump` to
   `/var/backups/flybot/compose-mongo.archive`), stops and disables Docker (its volumes are kept), installs Node 22,
   Caddy and MongoDB 7, restores the dump into the local MongoDB, installs the release and starts it under systemd.
   Caddy requests a new certificate.

Check afterwards: the summary shows `flybot=active caddy=active mongod=active`, and your data is there after logging in.
When everything is fine, free the Docker volumes: `make host CMD='docker volume prune -af; dnf remove -y docker'`.
To keep the t4g.small, set `INSTANCE_TYPE=t4g.small` in `deploy.env`.

## 4. Remove

`./destroy.sh --yes` deletes the stack, the stack's photo and artifacts buckets and the `flybot-<stage>-*` secrets (7-day
recovery window). `--keep-photos` keeps the photo bucket for a later deploy, `--keep-secrets` the secrets. A bucket of
your own (`S3_BUCKET_NAME`), the public model bucket and IoT things/certificates of registered cameras are never touched.
