#!/usr/bin/env python3
"""Create an experimental INT8 weight-quantized ONNX candidate."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--per-channel", action="store_true")
    parser.add_argument(
        "--include-prefix",
        action="append",
        help="Quantize only MatMul/Gemm nodes whose names start with this prefix.",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nodes_to_quantize = None
    if args.include_prefix:
        model = onnx.load(args.input, load_external_data=False)
        nodes_to_quantize = [
            node.name
            for node in model.graph.node
            if node.op_type in {"MatMul", "Gemm"}
            and any(node.name.startswith(prefix) for prefix in args.include_prefix)
        ]
        if not nodes_to_quantize:
            raise RuntimeError("No quantizable nodes matched --include-prefix")
        print(f"Selected {len(nodes_to_quantize)} MatMul/Gemm nodes")
    quantize_dynamic(
        model_input=args.input,
        model_output=args.output,
        op_types_to_quantize=["MatMul", "Gemm"],
        per_channel=args.per_channel,
        weight_type=QuantType.QInt8,
        nodes_to_quantize=nodes_to_quantize,
        use_external_data_format=True,
    )
    print(f"Quantized {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
