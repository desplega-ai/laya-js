"""Turn a split export (export_split.py --out-dir) into an uploadable fp32 bundle.

    uv run python finalize_bundle.py --bundle-dir <export out-dir> \
        --checkpoint multilingual --repo convaiinnovations/laya-multilingual --revision <sha>

1. Inlines external tensor data, so the bundle is exactly the files the vendored
   loader fetches (encoder.onnx, head.onnx, tokenizer.json, rl_agent_config.json).
   Every checkpoint is under protobuf's 2 GB limit, so inlining always works.
2. Checks both graphs with onnx.checker.
3. Writes manifest.json: checkpoint, HF repo and revision, upstream SHA, precision,
   opset, per-file bytes and SHA-256, export tool versions.

fp32 only. INT8 was deferred on 2026-09-28 (see the plan's Decisions); the
quantize step lives on the `phase-2-export-wip` branch for the later spike.
"""
import argparse
import hashlib
import importlib.metadata as md
import json
import os
import sys

import onnx

UPSTREAM_SHA = "9d955671415fc19f069b9cc998928075c1f255ec"
BUNDLE_FILES = ("encoder.onnx", "head.onnx", "tokenizer.json", "rl_agent_config.json")


def inline_external_data(path):
    """Rewrite `path` with every tensor inline and delete its external data files."""
    header = onnx.load(path, load_external_data=False)
    locations = {
        entry.value
        for init in header.graph.initializer
        if init.data_location == onnx.TensorProto.EXTERNAL
        for entry in init.external_data
        if entry.key == "location"
    }
    if not locations:
        return
    model = onnx.load(path, load_external_data=True)
    onnx.save_model(model, path, save_as_external_data=False)
    for loc in locations:
        os.remove(os.path.join(os.path.dirname(path), loc))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tool_versions():
    out = {"python": sys.version.split()[0]}
    for pkg in ("torch", "transformers", "onnx", "onnxruntime", "onnxscript", "laya"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = None
    return out


def write_manifest(bundle_dir, checkpoint, repo, revision, opset):
    files = {}
    for name in BUNDLE_FILES:
        p = os.path.join(bundle_dir, name)
        files[name] = {"bytes": os.path.getsize(p), "sha256": sha256(p)}
    manifest = {
        "checkpoint": checkpoint,
        "hfRepo": repo,
        "hfRevision": revision,
        "upstreamSha": UPSTREAM_SHA,
        "precision": "fp32",
        "encoderPrecision": "fp32",
        "headPrecision": "fp32",
        "opset": opset,
        "files": files,
        "tools": tool_versions(),
    }
    with open(os.path.join(bundle_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle-dir", required=True)
    p.add_argument("--checkpoint", required=True, choices=["english", "multilingual", "typed-decisions"])
    p.add_argument("--repo", required=True, help="Source HF repo of the checkpoint.")
    p.add_argument("--revision", required=True, help="Pinned HF commit the export used.")
    p.add_argument("--opset", type=int, default=18)
    args = p.parse_args(argv)

    for name in BUNDLE_FILES:
        if not os.path.isfile(os.path.join(args.bundle_dir, name)):
            raise SystemExit("missing %s in %s" % (name, args.bundle_dir))
    for name in ("encoder.onnx", "head.onnx"):
        path = os.path.join(args.bundle_dir, name)
        inline_external_data(path)
        onnx.checker.check_model(path)
    extra = sorted(set(os.listdir(args.bundle_dir)) - set(BUNDLE_FILES) - {"manifest.json"})
    if extra:
        raise SystemExit("unexpected files in %s: %s" % (args.bundle_dir, ", ".join(extra)))
    manifest = write_manifest(args.bundle_dir, args.checkpoint, args.repo, args.revision, args.opset)
    for name, f in manifest["files"].items():
        print("%-22s %12d  %s" % (name, f["bytes"], f["sha256"]))


if __name__ == "__main__":
    main()
