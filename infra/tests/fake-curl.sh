#!/usr/bin/env bash
# Stand-in for curl's GitHub calls in test_deploy_dry_run.sh (deploy.sh uses $CURL only for GitHub).
# Logs "<METHOD> <URL> auth=<yes|no>" to $FAKE_LOG (never the token). Repository agrhub/flybot-aws is public; no release
# exists yet, except flybot-<FAKE_EXISTING_TAG>. GitHub write calls answer 201/204 so a non-dry run could be simulated.
method=GET out="" fmt="" url="" auth=no head=0 next=""
for a in "$@"; do
  case "$next" in
    o) out="$a"; next=""; continue ;;
    w) fmt="$a"; next=""; continue ;;
    X) method="$a"; next=""; continue ;;
    H) case "$a" in @*) grep -q '^Authorization: Bearer ' "${a#@}" 2>/dev/null && auth=yes ;; esac; next=""; continue ;;
    skip) next=""; continue ;;
  esac
  case "$a" in
    -o) next=o ;; -w) next=w ;; -X) next=X ;; -H) next=H ;;
    --data-binary|--max-time|--retry) next=skip ;;
    -sIL|-I) head=1 ;;
    http*) url="$a" ;;
  esac
done
[ "$head" = 1 ] && method=HEAD
printf 'curl %s %s auth=%s\n' "$method" "$url" "$auth" >> "${FAKE_LOG:-/dev/null}"
code=404 body='{"message":"Not Found"}'
case "$method $url" in
  "GET https://api.github.com/repos/agrhub/flybot-aws") code=200; body='{"full_name":"agrhub/flybot-aws","private":false}' ;;
  "GET https://api.github.com/repos/agrhub/flybot-aws/releases?per_page=100")
    code=200
    body='[{"id":11,"tag_name":"flybot-aaaaaaaaaaa1","name":"FlyBot aaaaaaaaaaa1 (deploy build)","prerelease":true,"created_at":"2026-10-01T00:00:00Z"},
           {"id":12,"tag_name":"flybot-aaaaaaaaaaa2","name":"FlyBot aaaaaaaaaaa2 (deploy build)","prerelease":true,"created_at":"2026-10-02T00:00:00Z"},
           {"id":13,"tag_name":"v1.0.0","name":"FlyBot on AWS 1.0","prerelease":false,"created_at":"2026-10-03T00:00:00Z"}]' ;;
  "GET https://api.github.com/repos/agrhub/flybot-aws/releases/tags/flybot-${FAKE_EXISTING_TAG:-none}")
    code=200; body='{"id":42,"assets":[{"name":"flybot-'"${FAKE_EXISTING_TAG:-none}"'.tgz","id":1,"state":"uploaded","size":17000000},{"name":"flybot-'"${FAKE_EXISTING_TAG:-none}"'.tgz.sha256","id":2,"state":"uploaded","size":90}]}' ;;
  POST\ *) code=201; body='{"id":42,"assets":[]}' ;;
  DELETE\ *) code=204; body='' ;;
  HEAD\ *) code=200; body='' ;;
  "GET http://minio.example.com:8080/") code=403; body='<Error><Code>AccessDenied</Code></Error>' ;;
esac
case "$url" in *.tgz.sha256) [ "$method" = GET ] && { printf '%064d  asset\n' 0; exit 0; } ;; esac
[ -n "$out" ] && printf '%s' "$body" > "$out"
[ -n "$fmt" ] && printf '%s' "$code"
exit 0
