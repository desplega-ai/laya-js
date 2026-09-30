#!/usr/bin/env bash
# Install the laya auto-deploy on the swarm host (root):  ssh swarm 'bash -s' < install-autodeploy.sh
# Adds /opt/laya/deploy.sh, laya-deploy.{service,timer} and a comment block in /opt/laya/run.sh.
# Touches nothing else: not Caddy, not other containers, not the running laya container itself
# (the first deploy runs from the timer about a minute after this finishes). Idempotent.
# Flags: --no-wait  do not wait for and print the first timer-triggered run.
set -euo pipefail
DIR=${LAYA_DIR:-/opt/laya}
die() { echo "install-autodeploy: $*" >&2; exit 1; }
for c in docker curl tar flock systemctl; do command -v "$c" >/dev/null || die "missing command: $c"; done
[ -x "$DIR/run.sh" ] || die "$DIR/run.sh not found or not executable"
docker inspect laya >/dev/null 2>&1 || die "no container named laya"
grep -q 'laya-server:current' "$DIR/run.sh" || die "$DIR/run.sh does not start laya-server:current, so deploy.sh's swap contract is not met. Nothing was changed."

install -d "$DIR/state"
cat > "$DIR/deploy.sh" <<'LAYA_EOF_DEPLOY'
#!/usr/bin/env bash
# /opt/laya/deploy.sh: pull-based deploy of laya-server. Run by laya-deploy.timer every ~5 min.
#
# Every tick: find the newest stable v* tag of desplega-ai/laya-js. If it is newer than the image
# behind laya-server:current, and its npm release exists, then build that tag, swap the `laya`
# container onto it through run.sh, check it, and roll back to the previous image if it fails.
# Nothing here needs a credential: the repo is public, the tag is read over HTTPS, and no host
# key or token is stored in GitHub.
#
#   deploy.sh                      poll and deploy if a newer release exists (what the timer runs)
#   deploy.sh --tag v0.1.1         deploy that tag (a downgrade or a re-deploy needs --force)
#   deploy.sh --force              ignore "already current" and "previously failed" markers
#
# Exit 0: nothing to do, deployed, or a transient poll failure (next tick retries).
# Exit 1: a deploy failed (the unit shows `failed`; see `journalctl -u laya-deploy`).
set -uo pipefail

REPO=${LAYA_REPO:-desplega-ai/laya-js}
NPM_PKG=${LAYA_NPM_PKG:-@desplega.ai/laya-server}
DIR=${LAYA_DIR:-/opt/laya}
IMAGE=${LAYA_IMAGE:-laya-server}
CONTAINER=${LAYA_CONTAINER:-laya}
RUN_SCRIPT=${LAYA_RUN_SCRIPT:-$DIR/run.sh}
EXPECT_MODELS=${LAYA_EXPECT_MODELS:-"multilingual english typed-decisions"}
HEALTH_TIMEOUT=${LAYA_HEALTH_TIMEOUT:-900}   # seconds; the first start also fetches non-baked models
LOAD_PER_MODEL=${LAYA_LOAD_PER_MODEL:-20}
MAX_BUILD_ATTEMPTS=3
STATE=$DIR/state

TARGET="" FORCE=${LAYA_FORCE:-0}
while [ $# -gt 0 ]; do
  case $1 in
    --tag) TARGET=${2:-}; shift 2 ;;
    --force) FORCE=1; shift ;;
    *) echo "usage: deploy.sh [--tag vX.Y.Z] [--force]" >&2; exit 2 ;;
  esac
done

log() { printf '%s laya-deploy: %s\n' "$(date -u +%FT%TZ)" "$*"; }

mkdir -p "$STATE"
exec 9>"$DIR/.deploy.lock"
flock -n 9 || { log "another deploy is running, skipping"; exit 0; }

# --- 1. what is the newest release, and what runs now ----------------------------------------------
if [ -z "$TARGET" ]; then
  refs=$(curl -fsS --max-time 30 "https://api.github.com/repos/$REPO/git/matching-refs/tags/v") \
    || { log "cannot list tags (network or GitHub rate limit), will retry next tick"; exit 0; }
  TARGET=$(printf '%s' "$refs" | grep -oE 'refs/tags/v[0-9]+\.[0-9]+\.[0-9]+"' | sed -E 's#refs/tags/(.*)"#\1#' | sort -V | tail -n1)
  [ -n "$TARGET" ] || { log "no stable v* tag found"; exit 0; }
fi
case $TARGET in v[0-9]*.[0-9]*.[0-9]*) ;; *) log "refusing odd tag '$TARGET'"; exit 2 ;; esac

label() { docker image inspect -f '{{index .Config.Labels "org.opencontainers.image.version"}}' "$1" 2>/dev/null || true; }
CURRENT_TAG=$(label "$IMAGE:current")
CURRENT_ID=$(docker image inspect -f '{{.Id}}' "$IMAGE:current" 2>/dev/null || true)

if [ "$FORCE" != 1 ]; then
  if [ "$CURRENT_TAG" = "$TARGET" ]; then log "up to date ($TARGET)"; exit 0; fi
  if [ -n "$CURRENT_TAG" ] && [ "$(printf '%s\n%s\n' "$CURRENT_TAG" "$TARGET" | sort -V | tail -n1)" != "$TARGET" ]; then
    log "newest tag $TARGET is older than running $CURRENT_TAG, skipping (use --force to downgrade)"; exit 0
  fi
  if [ -f "$STATE/failed-$TARGET" ]; then
    log "skipping $TARGET: $(cat "$STATE/failed-$TARGET" | head -n1). Fix forward with a new tag, or run: deploy.sh --tag $TARGET --force"; exit 0
  fi
  # A release counts once the publish workflow finished, so a tag whose CI or publish failed never deploys.
  if ! curl -fsS -o /dev/null --max-time 20 "https://registry.npmjs.org/${NPM_PKG/\//%2f}/${TARGET#v}"; then
    log "$TARGET is tagged but $NPM_PKG@${TARGET#v} is not on npm yet, will retry next tick"; exit 0
  fi
fi

log "deploying $TARGET (running: ${CURRENT_TAG:-unlabelled} ${CURRENT_ID:0:19})"

fail_build() { # reason: count build-time failures; give up after MAX_BUILD_ATTEMPTS
  local n=1; [ -f "$STATE/build-attempts-$TARGET" ] && n=$(( $(cat "$STATE/build-attempts-$TARGET") + 1 ))
  echo "$n" > "$STATE/build-attempts-$TARGET"
  if [ "$n" -ge "$MAX_BUILD_ATTEMPTS" ]; then echo "build failed $n times: $1" > "$STATE/failed-$TARGET"; fi
  log "BUILD FAILED ($n/$MAX_BUILD_ATTEMPTS): $1. The running container was not touched."
  exit 1
}

# --- 2. build (does not touch the running container) -----------------------------------------------
SRC=$DIR/src-$TARGET
rm -rf "$SRC"; mkdir -p "$SRC"
trap 'rm -rf "$SRC"' EXIT
curl -fsSL --max-time 300 "https://codeload.github.com/$REPO/tar.gz/refs/tags/$TARGET" | tar -xz --strip-components=1 -C "$SRC" \
  || fail_build "cannot fetch source tarball for $TARGET"
[ -f "$SRC/packages/laya-server/Dockerfile" ] || fail_build "tarball has no packages/laya-server/Dockerfile"
DOCKER_BUILDKIT=1 docker build -f "$SRC/packages/laya-server/Dockerfile" \
  --label "org.opencontainers.image.version=$TARGET" --label "org.opencontainers.image.source=https://github.com/$REPO" \
  -t "$IMAGE:$TARGET" "$SRC" || fail_build "docker build"
NEW_ID=$(docker image inspect -f '{{.Id}}' "$IMAGE:$TARGET")
log "built $IMAGE:$TARGET ${NEW_ID:0:19}"

# --- 3. smoke program, run inside the container (its own env has the API key; no host port needed) --
smoke_js() { cat <<'JS'
// argv: <health|smoke> <loadPerModel> <model>...
import fs from "node:fs";
const [mode, load, ...models] = process.argv.slice(2);
const base = `http://127.0.0.1:${process.env.LAYA_PORT || 8000}`;
const auth = process.env.LAYA_API_KEY ? { authorization: `Bearer ${process.env.LAYA_API_KEY}` } : {};
const q = { route: { type: "choice", instructions: "Which team should handle this?",
  criteria: { billing: "payment, refund or invoice", engineering: "bug or crash", sales: "pricing or plans" } } };
const call = async (model, state) => {
  const t = performance.now();
  const r = await fetch(`${base}/v1/systemone`, { method: "POST", headers: { "content-type": "application/json", ...auth },
    body: JSON.stringify({ state, questions: q, model }) });
  const j = await r.json().catch(() => ({}));
  return { ok: r.status === 200, status: r.status, ms: performance.now() - t, model: j?.routing?.model, ans: j?.answers?.route };
};
const die = (m) => { console.log("FAIL " + m); process.exit(1); };
const h = await fetch(`${base}/health`).then(async (r) => ({ s: r.status, j: await r.json() })).catch(() => null);
if (!h || h.s !== 200 || h.j.status !== "ok") die(`/health ${h ? h.s + " " + h.j.status : "unreachable"}`);
const missing = models.filter((m) => !h.j.loaded.includes(m));
if (missing.length) die(`not loaded: ${missing.join(",")} (loaded: ${h.j.loaded.join(",")})`);
if (mode === "health") { console.log(`ok loaded=${h.j.loaded.join(",")}`); process.exit(0); }
const p = (a, f) => a.sort((x, y) => x - y)[Math.min(a.length - 1, Math.floor(f * a.length))];
for (const m of models) {
  const one = await call(m, "Please refund my last order, it arrived broken.");
  const sum = Object.values(one.ans?.probabilities ?? {}).reduce((a, b) => a + b, 0);
  if (!one.ok || one.model !== m || Math.abs(sum - 1) > 0.01) die(`${m}: status=${one.status} routed=${one.model} probsum=${sum.toFixed(3)}`);
  const ms = []; let bad = 0, next = 0;
  await Promise.all(Array.from({ length: 4 }, async () => { while (next < +load) { next++;
    const c = await call(m, `Customer message ${next}: the checkout page crashes and I was charged twice.`);
    if (!c.ok || c.model !== m) bad++; else ms.push(c.ms); } }));
  console.log(`${m}: choice=${one.ans.choice} p=${one.ans.probabilities[one.ans.choice]?.toFixed(3)} 1st=${one.ms.toFixed(0)}ms load n=${ms.length} bad=${bad} p50=${p(ms, .5)?.toFixed(0)}ms p95=${p(ms, .95)?.toFixed(0)}ms`);
  if (bad) die(`${m}: ${bad} failed requests under load`);
}
const st = fs.readFileSync("/proc/1/status", "utf8");
const hwm = +(st.match(/VmHWM:\s+(\d+)/)?.[1] ?? 0) / 1024;
let cg = null; for (const f of ["/sys/fs/cgroup/memory.peak", "/sys/fs/cgroup/memory/memory.max_usage_in_bytes"]) {
  try { cg = +fs.readFileSync(f, "utf8") / 1048576; break; } catch {} }
console.log(`peak_rss_mib=${hwm.toFixed(0)} cgroup_peak_mib=${cg === null ? "n/a" : cg.toFixed(0)}`);
JS
}
probe() { # mode -> runs inside the live container
  smoke_js | docker exec -i "$CONTAINER" node --input-type=module - "$1" "$LOAD_PER_MODEL" $EXPECT_MODELS 2>&1
}

wait_ready() { # -> 0 when /health is ok with all expected models, 1 on timeout or a dead container
  local end=$(( SECONDS + HEALTH_TIMEOUT )) out=""
  while [ $SECONDS -lt $end ]; do
    if [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null)" != true ]; then log "container is not running"; return 1; fi
    out=$(probe health) && { log "ready: $out"; return 0; }
    sleep 5
  done
  log "not ready after ${HEALTH_TIMEOUT}s: $out"; return 1
}

swap() { # $1 = image id or tag that becomes laya-server:current
  docker tag "$1" "$IMAGE:current" && "$RUN_SCRIPT"
}

rollback() { # $1 = reason
  log "ROLLBACK ($1)"
  echo "$1 ($(date -u +%FT%TZ))" > "$STATE/failed-$TARGET"
  if [ -z "$CURRENT_ID" ]; then log "no previous image to roll back to; leaving $TARGET in place"; exit 1; fi
  swap "$CURRENT_ID" || log "run.sh failed during rollback"
  [ -n "$OLD_PREV_ID" ] && docker tag "$OLD_PREV_ID" "$IMAGE:prev"   # rotation as it was before this attempt
  docker rmi "$IMAGE:$TARGET" >/dev/null 2>&1 || true                 # the failed build
  if wait_ready; then log "rolled back to ${CURRENT_TAG:-previous image} ${CURRENT_ID:0:19}; the endpoint is healthy"
  else log "!!! ROLLBACK ALSO UNHEALTHY: manual attention needed on $CONTAINER"; fi
  exit 1
}

# --- 4. swap ---------------------------------------------------------------------------------------
OLD_PREV_ID=$(docker image inspect -f '{{.Id}}' "$IMAGE:prev" 2>/dev/null || true)
[ -n "$CURRENT_ID" ] && docker tag "$CURRENT_ID" "$IMAGE:prev"
log "swapping $CONTAINER onto $TARGET (rollback image: ${CURRENT_ID:0:19})"
swap "$IMAGE:$TARGET" || rollback "run.sh failed"

# --- 5. verify: ready, one classify call and a short load per model, peak memory --------------------
wait_ready || rollback "container exited or was not healthy within ${HEALTH_TIMEOUT}s"
report=$(probe smoke) || { log "smoke output: $report"; rollback "classify smoke failed"; }
printf '%s\n' "$report" | while IFS= read -r l; do log "smoke: $l"; done

# --- 6. done: record it, drop the image that fell out of rotation (only ever laya-server images) ----
echo "$TARGET $NEW_ID $(date -u +%FT%TZ)" > "$STATE/deployed"
rm -f "$STATE/build-attempts-$TARGET" "$STATE/failed-$TARGET"
if [ -n "$OLD_PREV_ID" ] && [ "$OLD_PREV_ID" != "$NEW_ID" ] && [ "$OLD_PREV_ID" != "$CURRENT_ID" ]; then
  docker rmi "$OLD_PREV_ID" >/dev/null 2>&1 && log "removed retired image ${OLD_PREV_ID:0:19}"
fi
for t in $(docker images "$IMAGE" --format '{{.Tag}}' | grep -E '^v[0-9]'); do
  id=$(docker image inspect -f '{{.Id}}' "$IMAGE:$t")
  [ "$id" = "$NEW_ID" ] || [ "$id" = "$CURRENT_ID" ] || docker rmi "$IMAGE:$t" >/dev/null 2>&1
done
log "DEPLOYED $TARGET ${NEW_ID:0:19} (previous: ${CURRENT_TAG:-unlabelled} ${CURRENT_ID:0:19} kept as $IMAGE:prev)"
LAYA_EOF_DEPLOY
chmod 0755 "$DIR/deploy.sh"
cat > /etc/systemd/system/laya-deploy.service <<'LAYA_EOF_SERVICE'
[Unit]
Description=Deploy the newest laya-js release to the laya container
Documentation=file:///opt/laya/run.sh
Wants=network-online.target
After=network-online.target docker.service

[Service]
Type=oneshot
ExecStart=/opt/laya/deploy.sh
# A build plus a first model fetch plus a health wait; the timer never stacks runs (see the timer).
TimeoutStartSec=45min
Nice=10
LAYA_EOF_SERVICE
cat > /etc/systemd/system/laya-deploy.timer <<'LAYA_EOF_TIMER'
[Unit]
Description=Poll for a new laya-js release every 5 minutes

[Timer]
# First poll 1 min after the timer starts (this is what makes the install itself the first run),
# then 5 min after each run ends. OnUnitInactiveSec waits for a running deploy to finish.
OnActiveSec=1min
OnBootSec=3min
OnUnitInactiveSec=5min
RandomizedDelaySec=20
AccuracySec=10s

[Install]
WantedBy=timers.target
LAYA_EOF_TIMER
if ! grep -q '^# AUTO-DEPLOY (installed' "$DIR/run.sh"; then
  cp -p "$DIR/run.sh" "$DIR/run.sh.bak-$(date -u +%Y%m%dT%H%M%SZ)"
  cat >> "$DIR/run.sh" <<'LAYA_EOF_COMMENT'

# ---------------------------------------------------------------------------------------------
# AUTO-DEPLOY (installed 2026-09-30; laya-js releases deploy themselves)
#
# What runs: systemd timer laya-deploy.timer starts laya-deploy.service (Type=oneshot) about 1 min
# after it is enabled, then 5 min after each run ends (RandomizedDelaySec=20). The service runs
# /opt/laya/deploy.sh. Pull-based on purpose: the host reaches out to public GitHub and npm, so no
# SSH key or host credential exists in GitHub (the repo is public) and nothing can push to this box.
#
# Every tick, deploy.sh:
#   1. lists the newest stable v* tag of desplega-ai/laya-js (GitHub API, no token) and compares it
#      with the version label of laya-server:current. Equal or older: exit 0, nothing happens.
#   2. requires @desplega.ai/laya-server@<version> to exist on npm, so a tag whose CI or publish
#      failed never deploys. Not there yet: retry next tick.
#   3. downloads the tag tarball, builds laya-server:<tag> (same Dockerfile, same context rule as the
#      old recipe: docker build -f packages/laya-server/Dockerfile <source dir>). The live container
#      is not touched while building. Fails 3 times: the tag is marked failed.
#   4. tags the running image laya-server:prev (the rollback target), points laya-server:current at
#      the new image and runs THIS script to recreate the container. Contract: this script (re)creates
#      container `laya` from laya-server:current with the production flags below. Keep it that way.
#   5. waits up to 15 min for /health = ok with multilingual, english and typed-decisions loaded, then
#      does one classify call per model plus a 20-request load per model (inside the container, with
#      its own LAYA_API_KEY) and logs p50/p95 and peak RSS (VmHWM) and cgroup peak.
#   6. any failure in 4-5: laya-server:current goes back to the previous image, this script runs
#      again, /health is re-checked, and laya-server:<tag> is marked failed (no retry loop; a bad
#      release causes one short restart, not one per tick). The unit ends `failed`.
#   7. success: state/deployed records tag, image id and time; the image that fell out of the
#      prev/current rotation is removed. Only laya-server images are ever touched.
#
# Downtime: the swap is remove-and-recreate (run.sh), so /health is down for the model load, about
# 30-90 s per release; the long part (the build) happens before it. Caddy is never touched.
#
# Operate it:
#   systemctl list-timers laya-deploy.timer          next poll
#   journalctl -u laya-deploy.service -n 100         what the last runs did
#   /opt/laya/deploy.sh --tag v0.1.1 --force         deploy or re-deploy a tag by hand (also downgrades)
#   rm /opt/laya/state/failed-v0.1.2                 allow a tag that was marked failed to retry
#   systemctl disable --now laya-deploy.timer        stop auto-deploys
#   docker tag laya-server:prev laya-server:current && /opt/laya/run.sh    manual rollback
# Reference copy of deploy.sh and the units: desplega-ai/laya-js, deploy/selfhost/.
# The old recipe (git archive of main to /opt/laya/src, then docker build) is superseded: the
# deployed source is now always a release tag, in a temp dir removed after the build.
# ---------------------------------------------------------------------------------------------
LAYA_EOF_COMMENT
fi
systemd-analyze verify /etc/systemd/system/laya-deploy.service /etc/systemd/system/laya-deploy.timer
systemctl daemon-reload
systemctl enable --now laya-deploy.timer
echo "installed. timer: $(systemctl list-timers laya-deploy.timer --no-pager | sed -n 2p)"
[ "${1:-}" = "--no-wait" ] && exit 0

echo "waiting for the first timer-triggered run (about 1 min, then the build)..."
for _ in $(seq 1 240); do
  started=$(systemctl show laya-deploy.service -p ExecMainStartTimestamp --value)
  state=$(systemctl is-active laya-deploy.service || true)
  [ -n "$started" ] && [ "$state" != active ] && [ "$state" != activating ] && break
  sleep 15
done
echo "---- journalctl -u laya-deploy.service ----"
journalctl -u laya-deploy.service --no-pager -o short-iso | tail -n 60
echo "---- result ----"
systemctl show laya-deploy.service -p Result -p ExecMainStatus
cat "$DIR/state/deployed" 2>/dev/null || true
docker ps --filter name=^/laya$ --format 'container: {{.Names}} {{.Image}} {{.Status}}'
