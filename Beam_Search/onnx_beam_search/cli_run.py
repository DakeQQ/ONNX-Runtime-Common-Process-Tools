"""Run generic merged beam graphs from raw token IDs."""

from __future__ import annotations

import argparse
import json

import numpy as np

from .manifest import load_beam_search_manifest
from .runtime import BeamSearchRunner, OrtProviderConfig


def _token_ids(value: str) -> list[int]:
    value = value.strip()
    if value.startswith("["):
        parsed = json.loads(value)
        if not isinstance(parsed, list):
            raise argparse.ArgumentTypeError("Token IDs JSON must be a list.")
        return [int(token) for token in parsed]
    return [int(token.strip()) for token in value.split(",") if token.strip()]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run model-agnostic ONNX beam search from raw token IDs."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--input-ids",
        required=True,
        type=_token_ids,
        help="Comma-separated IDs or a JSON list, for example 1,42,7.",
    )
    parser.add_argument("--beam-size", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument(
        "--stop-token-ids",
        type=_token_ids,
        default=None,
        help="Optional override for manifest stop IDs.",
    )
    parser.add_argument("--provider", default="CPUExecutionProvider")
    parser.add_argument(
        "--device-type", choices=("cpu", "cuda", "dml"), default=None
    )
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--ort-log", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    manifest = load_beam_search_manifest(args.manifest)
    runner = BeamSearchRunner(
        manifest.model,
        OrtProviderConfig(
            providers=(args.provider,),
            device_type=args.device_type,
            device_id=args.device_id,
            threads=args.threads,
            log=args.ort_log,
        ),
    )
    input_ids = np.asarray([args.input_ids], dtype=np.int64)
    stop_ids = (
        tuple(args.stop_token_ids)
        if args.stop_token_ids is not None
        else manifest.stop_token_ids
    )
    result = runner.generate(
        input_ids,
        beam_size=args.beam_size,
        top_k=args.top_k,
        stop_token_ids=stop_ids,
        max_new_tokens=args.max_new_tokens,
        prefill_inputs=manifest.resolved_prefill_inputs(input_ids.shape[1]),
        decode_inputs=manifest.decode_inputs,
    )
    print(json.dumps({
        "token_ids": list(result.token_ids),
        "prompt_tokens": result.prompt_tokens,
        "generated_tokens": result.generated_tokens,
        "prefill_seconds": result.prefill_seconds,
        "decode_seconds": result.decode_seconds,
        "prefill_tokens_per_second": result.prefill_tokens_per_second,
        "decode_tokens_per_second": result.decode_tokens_per_second,
    }, indent=2))


if __name__ == "__main__":
    main()