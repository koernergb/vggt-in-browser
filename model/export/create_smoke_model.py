#!/usr/bin/env python3
"""Generate the tiny deterministic ONNX graph used to prove ORT Web routing."""

from pathlib import Path

import onnx
from onnx import TensorProto, helper

output = Path("apps/web/public/models/smoke-add.onnx")
output.parent.mkdir(parents=True, exist_ok=True)
graph = helper.make_graph(
    [helper.make_node("Add", ["input", "offset"], ["output"])],
    "smoke-add",
    [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])],
    [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])],
    [helper.make_tensor("offset", TensorProto.FLOAT, [1, 4], [2, 3, 4, 5])],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
model.ir_version = 9
onnx.checker.check_model(model)
onnx.save(model, output)
print(f"Wrote {output}")
