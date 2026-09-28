"""Writes tiny.onnx: y = x + 1 over a float32 [1, 2] tensor.

Used by the Bun smoke test to prove onnxruntime-node can create a session and
run under Bun. Regenerate with the export env:
    uv run --project tools/export python packages/laya/test/fixtures/make_tiny_onnx.py
"""
import os

import onnx
from onnx import TensorProto, helper

x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])
y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])
one = helper.make_tensor("one", TensorProto.FLOAT, [1], [1.0])
graph = helper.make_graph([helper.make_node("Add", ["x", "one"], ["y"])], "tiny", [x], [y], [one])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], producer_name="laya-js")
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tiny.onnx"))
