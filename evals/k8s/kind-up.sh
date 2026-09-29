#!/usr/bin/env bash
# Phase 10 k8s target: a one-node kind cluster on the reference host with the eval image loaded.
#   evals/k8s/kind-up.sh base|all-checkpoints   (prints the Service URL on the last line)
set -euo pipefail
overlay=${1:-base}
here=$(cd "$(dirname "$0")" && pwd)
kind get clusters | grep -qx laya || kind create cluster --name laya --wait 120s
kind load docker-image laya-server:eval --name laya
kubectl get secret laya-api >/dev/null 2>&1 || kubectl create secret generic laya-api --from-literal=api-key="${LAYA_API_KEY:?set LAYA_API_KEY}"
kubectl delete deployment laya-server --ignore-not-found --wait=true
kubectl apply -k "$here/$overlay"
kubectl rollout status deployment/laya-server --timeout=900s
ip=$(docker inspect -f '{{.NetworkSettings.Networks.kind.IPAddress}}' laya-control-plane)
echo "http://$ip:30080"
