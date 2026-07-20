"""Command-line export of generic beam-search helper graphs."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .discovery import discover_main_graph_contract
from .exporter import export_beam_search_graphs


def _csv(value, cast=str):
    if value is None:
        return None
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Export model-agnostic ONNX beam-search helper graphs."
    )
    parser.add_argument("--main-model", type=Path, required=True)
    parser.add_argument("--output-folder", type=Path, required=True)
    parser.add_argument("--state-count", type=int, default=None)
    parser.add_argument("--logits-output", default=None)
    parser.add_argument(
        "--state-inputs",
        default=None,
        help="Comma-separated state input names when they cannot be inferred.",
    )
    parser.add_argument(
        "--state-outputs",
        default=None,
        help="Comma-separated state output names; defaults to outputs before logits.",
    )
    parser.add_argument(
        "--sequence-axes",
        default=None,
        help="Comma-separated sequence axes when symbolic axes cannot be inferred.",
    )
    parser.add_argument("--beam-size", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--opset", type=int, default=None)
    parser.add_argument(
        "--token-dtype", choices=("int32", "int64"), default="int32"
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    contract = discover_main_graph_contract(
        args.main_model,
        state_count=args.state_count,
        logits_output_name=args.logits_output,
        state_input_names=_csv(args.state_inputs),
        state_output_names=_csv(args.state_outputs),
        sequence_axes=_csv(args.sequence_axes, int),
    )
    token_dtype = torch.int32 if args.token_dtype == "int32" else torch.int64
    exported = export_beam_search_graphs(
        contract.state_specs,
        contract.vocab_size,
        args.output_folder,
        beam_size=args.beam_size,
        top_k=args.top_k,
        opset=args.opset or contract.opset,
        token_dtype=token_dtype,
        logits_dtype=contract.logits_dtype,
    )
    print(
        f"Discovered {len(contract.state_specs)} state tensors, "
        f"vocab_size={contract.vocab_size}, opset={args.opset or contract.opset}."
    )
    for role, path in exported.items():
        print(f"{role}: {path}")


if __name__ == "__main__":
    main()