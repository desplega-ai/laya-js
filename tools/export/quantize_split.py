"""Build the fp32 and INT8 bundles from a split export (encoder.onnx + head.onnx).

    uv run python quantize_split.py --fp32-dir <export out-dir> --int8-dir <dir> \
        --checkpoint multilingual --repo convaiinnovations/laya-multilingual --revision <sha>

1. Inlines external tensor data, so each bundle is exactly the files the vendored
   loader fetches (encoder.onnx, head.onnx, tokenizer.json, rl_agent_config.json).
2. Quantizes encoder.onnx with upstream's recipe (scripts/export_onnx.py:8-40 at
   9d955671): quantize_dynamic, MatMul only, QInt8, per_channel=True.
   head.onnx stays fp32 (plan, Decisions: "Head precision").
3. Checks the INT8 encoder with onnx.checker, requires identical I/O names and a
   size that proves every MatMul weight was quantized, then writes manifest.json
   into both bundles.

The plan's "INT8 encoder at most 35% of fp32" bar assumed every weight shrinks.
Upstream's recipe quantizes MatMul only, so the token embedding (a Gather) stays
fp32. That caps the ratio near 35% for ModernBERT-large (50k vocab) and near 73%
for mmBERT-base (256k vocab x 768, about 64% of its weights). The check below
therefore bounds the size by the MatMul share instead of a fixed 35%.
"""
import argparse
import hashlib
import importlib.metadata as md
import json
import os
import shutil
import sys

import onnx

UPSTREAM_SHA = "9d955671415fc19f069b9cc998928075c1f255ec"
BUNDLE_FILES = ("encoder.onnx", "head.onnx", "tokenizer.json", "rl_agent_config.json")
# int8 bytes <= (fp32 bytes - 3/4 of MatMul weight bytes) * slack, for scales and zero points.
SIZE_SLACK = 1.02


def quantize_model(model_path, output_path):
    """Upstream's quantize_model (scripts/export_onnx.py:8-40), unchanged in effect."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    model = onnx.load(model_path)
    # Upstream: the exporter's value_info shapes disagree with the quantizer's own
    # shape inference; they are informational only, so drop them.
    del model.graph.value_info[:]
    quantize_dynamic(
        model_input=model,
        model_output=output_path,
        op_types_to_quantize=["MatMul"],
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    return output_path


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


def matmul_weight_bytes(path):
    """Bytes of float initializers that feed a MatMul (what quantize_dynamic shrinks)."""
    model = onnx.load(path, load_external_data=False)
    inits = {i.name: i for i in model.graph.initializer}
    names = {x for n in model.graph.node if n.op_type == "MatMul" for x in n.input if x in inits}
    total = 0
    for name in names:
        t = inits[name]
        if t.data_type == onnx.TensorProto.FLOAT:
            n = 1
            for d in t.dims:
                n *= d
            total += 4 * n
    return total


def io_names(path):
    model = onnx.load(path, load_external_data=False)
    return [i.name for i in model.graph.input], [o.name for o in model.graph.output]


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


def write_manifest(bundle_dir, checkpoint, repo, revision, precision, opset):
    files = {}
    for name in BUNDLE_FILES:
        p = os.path.join(bundle_dir, name)
        files[name] = {"bytes": os.path.getsize(p), "sha256": sha256(p)}
    manifest = {
        "checkpoint": checkpoint,
        "hfRepo": repo,
        "hfRevision": revision,
        "upstreamSha": UPSTREAM_SHA,
        "precision": precision,
        "encoderPrecision": precision,
        "headPrecision": "fp32",
        "opset": opset,
        "files": files,
        "tools": tool_versions(),
    }
    if precision == "int8":
        manifest["quantization"] = {
            "method": "onnxruntime.quantization.quantize_dynamic",
            "opTypes": ["MatMul"],
            "weightType": "QInt8",
            "perChannel": True,
            "appliedTo": ["encoder.onnx"],
        }
    with open(os.path.join(bundle_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fp32-dir", required=True)
    p.add_argument("--int8-dir", required=True)
    p.add_argument("--checkpoint", required=True, choices=["english", "multilingual", "typed-decisions"])
    p.add_argument("--repo", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--opset", type=int, default=18)
    args = p.parse_args(argv)

    for name in ("encoder.onnx", "head.onnx"):
        inline_external_data(os.path.join(args.fp32_dir, name))
    extra = sorted(set(os.listdir(args.fp32_dir)) - set(BUNDLE_FILES) - {"manifest.json"})
    if extra:
        raise SystemExit("unexpected files in fp32 bundle: %s" % extra)

    os.makedirs(args.int8_dir, exist_ok=True)
    fp32_enc = os.path.join(args.fp32_dir, "encoder.onnx")
    int8_enc = os.path.join(args.int8_dir, "encoder.onnx")
    quantize_model(fp32_enc, int8_enc)
    for name in ("head.onnx", "tokenizer.json", "rl_agent_config.json"):
        shutil.copyfile(os.path.join(args.fp32_dir, name), os.path.join(args.int8_dir, name))

    onnx.checker.check_model(int8_enc)
    if io_names(int8_enc) != io_names(fp32_enc):
        raise SystemExit("INT8 encoder I/O names differ: %s vs %s" % (io_names(int8_enc), io_names(fp32_enc)))
    fp32_bytes, int8_bytes = os.path.getsize(fp32_enc), os.path.getsize(int8_enc)
    mm = matmul_weight_bytes(fp32_enc)
    bound = (fp32_bytes - 0.75 * mm) * SIZE_SLACK
    ratio = int8_bytes / fp32_bytes
    if mm == 0 or int8_bytes > bound:
        raise SystemExit("INT8 encoder %d bytes exceeds the MatMul-quantized bound %d (MatMul weights %d bytes)" % (
            int8_bytes, bound, mm))

    for d, prec in ((args.fp32_dir, "fp32"), (args.int8_dir, "int8")):
        write_manifest(d, args.checkpoint, args.repo, args.revision, prec, args.opset)
    print("ok: %s int8 encoder %.1f%% of fp32 (%d -> %d bytes; MatMul weights %.1f%% of fp32)" % (
        args.checkpoint, ratio * 100, fp32_bytes, int8_bytes, 100 * mm / fp32_bytes))


if __name__ == "__main__":
    main()
