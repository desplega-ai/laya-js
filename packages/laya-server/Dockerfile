# syntax=docker/dockerfile:1
#
# laya-server with the multilingual fp32 bundle baked in (plan, Phase 9).
#
#   DOCKER_BUILDKIT=1 docker build -f packages/laya-server/Dockerfile -t laya-server .
#
# The build context is the repo root. The artifact store is public, so the build needs no token.
# An optional HF read token (rate limits, or a private mirror) goes in as a BuildKit secret:
#   --secret id=hf_token,env=HF_TOKEN
# It is mounted only in the `models` stage: never an ARG or ENV, so it is absent from every layer
# and from the image history. No torch in any stage; only onnxruntime-node's CPU binaries for the
# target arch ship.

ARG NODE_IMAGE=node:22-slim
ARG BUN_IMAGE=oven/bun:1.4.0

FROM ${BUN_IMAGE} AS bun

# ---- build: full install (TypeScript) and `tsc -b` of both packages ----------------------------
FROM ${NODE_IMAGE} AS build
COPY --from=bun /usr/local/bin/bun /usr/local/bin/bun
# onnxruntime-node's postinstall would fetch CUDA binaries on linux-x64; the image is CPU-only.
ENV ONNXRUNTIME_NODE_INSTALL=skip
WORKDIR /app
COPY package.json bun.lock tsconfig.json tsconfig.base.json ./
COPY packages/laya/package.json packages/laya/tsconfig.json packages/laya/
COPY packages/laya-server/package.json packages/laya-server/tsconfig.json packages/laya-server/
RUN bun install --frozen-lockfile
COPY packages/laya/src packages/laya/src
COPY packages/laya-server/src packages/laya-server/src
RUN bun run --filter @desplega.ai/laya-server build

# ---- deps: production dependencies only, pruned to this platform's ORT binaries -----------------
FROM ${NODE_IMAGE} AS deps
COPY --from=bun /usr/local/bin/bun /usr/local/bin/bun
ENV ONNXRUNTIME_NODE_INSTALL=skip
ARG TARGETARCH
WORKDIR /app
COPY package.json bun.lock ./
COPY packages/laya/package.json packages/laya/
COPY packages/laya-server/package.json packages/laya-server/
RUN bun install --frozen-lockfile --production \
 && arch="$([ "${TARGETARCH:-amd64}" = "arm64" ] && echo arm64 || echo x64)" \
 && for bin in node_modules/.bun/onnxruntime-node@*/node_modules/onnxruntime-node/bin/napi-v*; do \
      find "$bin" -mindepth 1 -maxdepth 1 ! -name linux -exec rm -rf {} + ; \
      find "$bin/linux" -mindepth 1 -maxdepth 1 ! -name "$arch" -exec rm -rf {} + ; \
      test -d "$bin/linux/$arch" || { echo "onnxruntime-node has no linux/$arch binary" >&2; exit 1; } ; \
    done

# ---- models: fetch multilingual/fp32 at the pinned revision, verify SHA-256 ---------------------
FROM build AS models
# fetch-models reads HF_TOKEN from its environment only; the secret file never lands in a layer.
RUN --mount=type=secret,id=hf_token,required=false \
    if [ -s /run/secrets/hf_token ]; then HF_TOKEN="$(cat /run/secrets/hf_token)"; export HF_TOKEN; fi; \
    node packages/laya-server/dist/fetch-models.js --dest /models multilingual

# ---- runtime ------------------------------------------------------------------------------------
FROM ${NODE_IMAGE}
ENV NODE_ENV=production \
    LAYA_HOST=0.0.0.0 \
    LAYA_PORT=8000 \
    LAYA_MODELS=multilingual \
    LAYA_PRECISION=fp32 \
    LAYA_MODEL_DIR=/models \
    LAYA_CACHE_DIR=/cache
WORKDIR /app
COPY LICENSE NOTICE ./
COPY --from=deps /app/node_modules ./node_modules
COPY --from=deps /app/packages/laya/node_modules packages/laya/node_modules
COPY --from=deps /app/packages/laya-server/node_modules packages/laya-server/node_modules
COPY packages/laya/package.json packages/laya/
COPY packages/laya-server/package.json packages/laya-server/
COPY --from=build /app/packages/laya/dist packages/laya/dist
COPY --from=build /app/packages/laya-server/dist packages/laya-server/dist
COPY --from=models /models /models
# /cache receives checkpoints that are listed in LAYA_MODELS but not baked; k8s mounts a volume here.
RUN mkdir -p /cache && chown 1000:1000 /cache
# The `node` user of the base image, numeric so k8s `runAsNonRoot` can verify it.
USER 1000:1000
WORKDIR /app/packages/laya-server
EXPOSE 8000
# node:22-slim has no curl; Node's fetch is enough. start-period covers the model load.
HEALTHCHECK --interval=15s --timeout=5s --start-period=300s --retries=3 \
  CMD ["node", "-e", "fetch('http://127.0.0.1:'+(process.env.LAYA_PORT||8000)+'/health').then(r=>process.exit(r.status===200?0:1),()=>process.exit(1))"]
CMD ["node", "dist/main.js"]
