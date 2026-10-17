#!/bin/bash
# The host bootstrap, published by ./deploy.sh to s3://<artifacts>/host/bootstrap.sh and run by the EC2 user-data stub
# (first boot) and by ./deploy.sh through SSM (every deploy). Changing this file never restarts the instance: only the
# stub is in the CloudFormation user-data. The stack's values come from /etc/flybot/stack.env, written by the stub.
# FlyBot host bootstrap (Amazon Linux 2023 arm64; a t4g.micro is enough: nothing is built here). No Docker, no git:
# Node 22 (nodejs.org, SHASUMS256 checked), Caddy (GitHub release, sha512 checked), MongoDB 7 when LocalMongo=true
# (repo.mongodb.org), then the FlyBot release named by s3://<artifacts>/flybot/current.json (sha256 checked; the
# previous release is kept for ./deploy.sh --rollback) under systemd. Idempotent: deploy.sh re-runs the current
# version of this script through SSM after every deploy (cloud-init runs it once). Log: /var/log/flybot-bootstrap.log
set -euo pipefail
exec > >(tee -a /var/log/flybot-bootstrap.log) 2>&1
set -a; . /etc/flybot/stack.env; set +a
FLYBOT_MEDIA_DIR=/var/lib/flybot-media
echo "bootstrap start $(date -u +%FT%TZ)"
export AWS_DEFAULT_REGION=${REGION} PATH=/usr/local/bin:$PATH HOME=/root
FORCE=${FORCE_RESTART:-0}
if [ ! -f /swapfile ]; then
  fallocate -l 2G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
  echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi
dnf install -y -q tar xz python3 >/dev/null
mkdir -p /opt/flybot/releases /etc/flybot /etc/caddy /var/backups/flybot

# 1. Migration from the docker compose host of the first deploys: dump its MongoDB, stop and disable Docker
if systemctl list-unit-files docker.service >/dev/null 2>&1 && systemctl is-enabled -q docker.service 2>/dev/null; then
  systemctl start docker || true
  M=$(docker ps -a --format '{{.Names}}' | grep -m1 -- '-mongo-' || true)
  if [ -n "$M" ] && [ ! -s /var/backups/flybot/compose-mongo.archive ]; then
    docker start "$M" >/dev/null && sleep 8
    docker exec "$M" mongodump --quiet --archive --db flybot > /var/backups/flybot/compose-mongo.archive \
      && echo "compose MongoDB dumped" || rm -f /var/backups/flybot/compose-mongo.archive
  fi
  docker ps -q | xargs -r docker stop >/dev/null || true
  systemctl disable --now docker.service docker.socket || true
  echo "Docker stopped and disabled (its volumes are kept; dnf remove -y docker to drop it)"
fi

# 2. Node.js 22
NODE_V="${NODE_VERSION}"
if [ -z "$NODE_V" ]; then
  NODE_V=$(curl -fsSL https://nodejs.org/dist/latest-v22.x/SHASUMS256.txt | grep -o 'node-v22\.[0-9.]*-linux-arm64\.tar\.xz' | head -1 | sed 's/^node-v//; s/-linux-arm64\.tar\.xz$//')
fi
if [ "$(node --version 2>/dev/null || true)" != "v$NODE_V" ]; then
  f=node-v$NODE_V-linux-arm64.tar.xz
  curl -fsSL -o /tmp/$f https://nodejs.org/dist/v$NODE_V/$f
  (cd /tmp && curl -fsSL https://nodejs.org/dist/v$NODE_V/SHASUMS256.txt | grep " $f\$" | sha256sum -c -)
  rm -rf /usr/local/lib/nodejs && mkdir -p /usr/local/lib/nodejs
  tar -xJf /tmp/$f -C /usr/local/lib/nodejs --strip-components=1 && rm -f /tmp/$f
  ln -sf /usr/local/lib/nodejs/bin/node /usr/local/bin/node
  ln -sf /usr/local/lib/nodejs/bin/npm /usr/local/bin/npm
fi
echo "node $(node --version)"

# 3. Caddy (TLS for ${DOMAIN_NAME}); certificates in /var/lib/caddy
CV="${CADDY_VERSION}"
if [ "$(caddy version 2>/dev/null | cut -d' ' -f1 || true)" != "v$CV" ]; then
  f=caddy_"$CV"_linux_arm64.tar.gz; base=https://github.com/caddyserver/caddy/releases/download/v$CV
  curl -fsSL -o /tmp/$f $base/$f
  (cd /tmp && curl -fsSL $base/caddy_"$CV"_checksums.txt | grep " $f\$" | sha512sum -c -)
  tar -xzf /tmp/$f -C /usr/local/bin caddy && rm -f /tmp/$f
fi
id caddy >/dev/null 2>&1 || useradd --system --home-dir /var/lib/caddy --create-home --shell /sbin/nologin caddy
cat > /etc/caddy/Caddyfile.new <<EOF
{
  email ${ACME_EMAIL}
}
${DOMAIN_NAME} {
  encode gzip zstd
  # The home page / manual videos are kept out of the release tarball and served from a local copy (synced from S3)
  # when it exists; otherwise the request goes to FlyBot like everything else (FLYBOT_MEDIA=bundled).
  @media {
    path_regexp ^/media/[^/]+\.mp4$
    file {
      root ${FLYBOT_MEDIA_DIR}
    }
  }
  handle @media {
    root * ${FLYBOT_MEDIA_DIR}
    file_server
  }
  handle {
    request_body {
      max_size 20MB
    }
    reverse_proxy 127.0.0.1:8080 {
      flush_interval -1
      transport http {
        read_timeout 180s
      }
    }
  }
  header {
    Strict-Transport-Security "max-age=31536000; includeSubDomains"
    -Server
  }
}
EOF
CADDY_CHANGED=0; cmp -s /etc/caddy/Caddyfile.new /etc/caddy/Caddyfile || CADDY_CHANGED=1
mv /etc/caddy/Caddyfile.new /etc/caddy/Caddyfile
cat > /etc/systemd/system/caddy.service <<'EOF'
[Unit]
Description=Caddy (TLS reverse proxy for FlyBot)
After=network-online.target
Wants=network-online.target
[Service]
User=caddy
Group=caddy
Environment=XDG_DATA_HOME=/var/lib XDG_CONFIG_HOME=/var/lib/caddy/.config
ExecStart=/usr/local/bin/caddy run --config /etc/caddy/Caddyfile
ExecReload=/usr/local/bin/caddy reload --config /etc/caddy/Caddyfile --force
AmbientCapabilities=CAP_NET_BIND_SERVICE
CapabilityBoundingSet=CAP_NET_BIND_SERVICE
NoNewPrivileges=true
LimitNOFILE=1048576
Restart=always
RestartSec=5
[Install]
WantedBy=multi-user.target
EOF

# 4. MongoDB 7 on this instance (LocalMongo=${LOCAL_MONGO}); otherwise MONGO_URI comes from the app secret
MONGO_LINE=""
if [ "${LOCAL_MONGO}" = true ]; then
  if ! rpm -q mongodb-org-server >/dev/null 2>&1; then
    cat > /etc/yum.repos.d/mongodb-org-7.0.repo <<'EOF'
[mongodb-org-7.0]
name=MongoDB 7.0
baseurl=https://repo.mongodb.org/yum/amazon/2023/mongodb-org/7.0/aarch64/
gpgcheck=1
enabled=1
gpgkey=https://pgp.mongodb.com/server-7.0.asc
EOF
    dnf install -y -q mongodb-org-server mongodb-database-tools >/dev/null
  fi
  cat > /etc/mongod.conf <<'EOF'
storage:
  dbPath: /var/lib/mongo
  wiredTiger:
    engineConfig:
      cacheSizeGB: 0.25
systemLog:
  destination: file
  logAppend: true
  path: /var/log/mongodb/mongod.log
net:
  port: 27017
  bindIp: 127.0.0.1
processManagement:
  timeZoneInfo: /usr/share/zoneinfo
EOF
  systemctl enable -q mongod && systemctl start mongod
  for _ in $(seq 1 30); do (echo > /dev/tcp/127.0.0.1/27017) 2>/dev/null && break; sleep 2; done
  if [ -s /var/backups/flybot/compose-mongo.archive ] && [ ! -f /var/backups/flybot/.restored ]; then
    mongorestore --quiet --archive=/var/backups/flybot/compose-mongo.archive && touch /var/backups/flybot/.restored \
      && echo "compose MongoDB restored into the local mongod"
  fi
  MONGO_LINE="MONGO_URI=mongodb://127.0.0.1:27017/flybot"
else
  systemctl disable --now mongod 2>/dev/null || true
fi

# 5. FlyBot environment (secrets stay in Secrets Manager flybot-${STAGE}-app: FLYBOT_SECRETS, read at start-up)
id flybot >/dev/null 2>&1 || useradd --system --home-dir /opt/flybot --shell /sbin/nologin flybot
# Cameras fetch KVS master keys from the IoT credential provider with the role alias of the stack (live view)
IOT_CREDENTIALS_ENDPOINT=$(aws iot describe-endpoint --endpoint-type iot:CredentialProvider --query endpointAddress --output text 2>/dev/null || true)
umask 077
cat > /etc/flybot/env.new <<EOF
NODE_ENV=production
PORT=8080
NODE_OPTIONS=--max-old-space-size=512
TRUST_PROXY=1
COOKIE_SECURE=true
APP_BASE_URL=https://${DOMAIN_NAME}
DEVICE_API_BASE_URL=https://${DOMAIN_NAME}
ENABLE_CRON=true
AWS_REGION=${REGION}
S3_BUCKET_NAME=${FLYBOT_BUCKET}
S3_REGION=${S3_REGION}
S3_ENDPOINT=${S3_ENDPOINT}
PHOTOS_BY_REFERENCE=${PHOTOS_BY_REFERENCE}
AI_PROVIDER=opencv-lambda
VISION_LAMBDA_NAME=flybot-vision-${STAGE}
IOT_ENDPOINT=${IOT_ENDPOINT}
IOT_EVENTS=${IOT_EVENTS}
IOT_RULE_SECRET=${IOT_RULE_SECRET}
IOT_THING_TYPE=camos-camera
IOT_POLICY_NAME=${IOT_POLICY_NAME}
KVS_VIEWER_ROLE_ARN=${KVS_VIEWER_ROLE_ARN}
IOT_CREDENTIALS_ENDPOINT=${IOT_CREDENTIALS_ENDPOINT}
IOT_ROLE_ALIAS=flybot-camera-kvs-${STAGE}
FLYBOT_SECRETS=flybot-${STAGE}-app
APP_MODE=${APP_MODE}
DEMO_MAX_CAMERAS_PER_USER=${DEMO_MAX_CAMERAS_PER_USER}
DEMO_MAX_SNAPSHOTS_PER_DAY=${DEMO_MAX_SNAPSHOTS_PER_DAY}
DEMO_MAX_UNCLAIMED=${DEMO_MAX_UNCLAIMED}
EOF
[ -n "$MONGO_LINE" ] && echo "$MONGO_LINE" >> /etc/flybot/env.new
if aws secretsmanager get-secret-value --secret-id "flybot-${STAGE}-gcp" --query SecretString --output text > /etc/flybot/gcp.tmp 2>/dev/null \
   && grep -q '"private_key"' /etc/flybot/gcp.tmp; then
  install -m 600 -o flybot -g flybot /etc/flybot/gcp.tmp /etc/flybot/gcp-service-account.json
  echo "GOOGLE_APPLICATION_CREDENTIALS=/etc/flybot/gcp-service-account.json" >> /etc/flybot/env.new
fi
rm -f /etc/flybot/gcp.tmp
ENV_CHANGED=0; cmp -s /etc/flybot/env.new /etc/flybot/env || ENV_CHANGED=1
mv /etc/flybot/env.new /etc/flybot/env
umask 022

# 6. The FlyBot release (GitHub Release or S3, sha256 from the pointer; switch atomically, keep the previous one)
RELEASED=0
if aws s3 cp s3://${ARTIFACTS_BUCKET}/flybot/current.json /opt/flybot/current.json --only-show-errors 2>/dev/null; then
  SHA=$(python3 -c 'import json;print(json.load(open("/opt/flybot/current.json"))["sha256"])')
  URL=$(python3 -c 'import json;print(json.load(open("/opt/flybot/current.json")).get("url",""))')
  KEY=$(python3 -c 'import json;print(json.load(open("/opt/flybot/current.json")).get("key",""))')
  R=/opt/flybot/releases/$(echo "$SHA" | cut -c1-16)
  if [ ! -f "$R/.ok" ]; then
    rm -rf "$R.tmp" && mkdir -p "$R.tmp"
    if [ -n "$URL" ]; then curl -fsSL --retry 3 -o /tmp/flybot.tgz "$URL"     # GitHub Release asset, public
    else aws s3 cp "s3://${ARTIFACTS_BUCKET}/$KEY" /tmp/flybot.tgz --only-show-errors; fi
    echo "$SHA  /tmp/flybot.tgz" | sha256sum -c -
    tar -xzf /tmp/flybot.tgz -C "$R.tmp" && rm -f /tmp/flybot.tgz
    (cd "$R.tmp" && npm ci --omit=dev --workspace=server --no-audit --no-fund --loglevel=error)
    cp /opt/flybot/current.json "$R.tmp/.pointer.json" && touch "$R.tmp/.ok" && mv "$R.tmp" "$R"
  fi
  CUR=$(readlink -f /opt/flybot/current 2>/dev/null || true)
  if [ "$CUR" != "$R" ]; then
    [ -n "$CUR" ] && [ -d "$CUR" ] && ln -sfn "$CUR" /opt/flybot/previous
    ln -sfn "$R" /opt/flybot/current.new && mv -T /opt/flybot/current.new /opt/flybot/current
    RELEASED=1
  fi
  PREV=$(readlink -f /opt/flybot/previous 2>/dev/null || true)
  for d in /opt/flybot/releases/*; do
    [ "$d" = "$R" ] || [ "$d" = "$PREV" ] || rm -rf "$d"
  done
else
  echo "no FlyBot release published yet (./deploy.sh uploads it)"
fi
# 7. Videos kept out of the release: s3://<artifacts>/media/ -> FLYBOT_MEDIA_DIR/media/ (read by Caddy)
mkdir -p "$FLYBOT_MEDIA_DIR/media" && chmod 755 "$FLYBOT_MEDIA_DIR" "$FLYBOT_MEDIA_DIR/media"
aws s3 sync "s3://${ARTIFACTS_BUCKET}/media/" "$FLYBOT_MEDIA_DIR/media/" --only-show-errors --exact-timestamps \
  || echo "media sync failed: the home page videos may be missing"
chmod -R a+rX "$FLYBOT_MEDIA_DIR"
cat > /etc/systemd/system/flybot.service <<'EOF'
[Unit]
Description=FlyBot (prebuilt release)
After=network-online.target mongod.service
Wants=network-online.target
[Service]
User=flybot
Group=flybot
WorkingDirectory=/opt/flybot/current
EnvironmentFile=/etc/flybot/env
ExecStart=/usr/local/bin/node server/dist/app.js
NoNewPrivileges=true
Restart=always
RestartSec=5
[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable -q caddy flybot
if [ "$CADDY_CHANGED" = 1 ]; then systemctl restart caddy; else systemctl start caddy; fi
if [ -e /opt/flybot/current ]; then
  if [ "$RELEASED" = 1 ] || [ "$ENV_CHANGED" = 1 ] || [ "$FORCE" != 0 ]; then systemctl restart flybot; else systemctl start flybot; fi
  for _ in $(seq 1 45); do curl -sf -o /dev/null http://127.0.0.1:8080/health && break; sleep 2; done
  echo "release $(python3 -c 'import json;d=json.load(open("/opt/flybot/current/.pointer.json"));print(d.get("commit"), d.get("branch"), d["sha256"][:12])')"
fi
echo "bootstrap done $(date -u +%FT%TZ)"
