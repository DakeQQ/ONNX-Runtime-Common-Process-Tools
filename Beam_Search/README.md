# ONNX Beam Search Toolkit

A portable, model-agnostic toolkit for exporting beam-search ONNX operators and
running autoregressive prefill/decode graph pairs with ONNX Runtime I/O binding.

The folder is self-contained and can be moved into another repository. It does
not contain model adapters, tokenizer code, chat templates, model filenames, or
paths into its current parent repository.

## Contents

| Path | Purpose |
| --- | --- |
| `onnx_beam_search/exporter.py` | Beam operators and four generic ONNX exports. |
| `onnx_beam_search/discovery.py` | Discover recurrent-state contracts from an ONNX Main graph. |
| `onnx_beam_search/runtime.py` | Tokenizer-free prefill/decode runner with I/O-binding ping-pong. |
| `onnx_beam_search/manifest.py` | Portable JSON runtime configuration. |
| `onnx_beam_search/cli_export.py` | `onnx-beam-export` command. |
| `onnx_beam_search/cli_run.py` | `onnx-beam-run` command. |
| `examples/generic_export.py` | Source-tree export launcher. |
| `examples/generic_llm_demo.py` | Source-tree raw-token inference launcher. |
| `examples/beam_manifest.example.json` | Generic runtime manifest template. |
| `tests/test_toolkit.py` | Synthetic ONNX export and runtime tests. |

## Install

```bash
python -m pip install -e .
```

This installs:

- `onnx-beam-export`: inspect a Main graph and export beam helper graphs.
- `onnx-beam-run`: run compatible merged graphs from raw token IDs.

The only runtime dependencies are NumPy, ONNX, ONNX Runtime, and PyTorch.

## Beam Algorithm

Initial step:

1. Normalize one logits row with `logits - logsumexp(logits)`.
2. Select the best `beam_size` tokens.
3. Expand every opaque state tensor from batch one to `beam_size` rows.
4. Start one saved-token row and one cumulative score per beam.

Decode step:

1. Select `top_k` candidates independently from each beam.
2. Add each beam's cumulative log probability.
3. Select the global best `beam_size` values from the flattened candidates.
4. Recover parent beam rows with integer division by `top_k`.
5. Gather every state tensor and saved-token row from the same parents.

The toolkit does not interpret state contents. State can include floating-point
KV data, integer KV data, quantization scales and biases, recurrent tensors,
convolution state, or any other tensor whose batch axis is zero.

## Export Helpers

The default discovery rule treats the final Main output as logits and all
preceding outputs as recurrent state. A state output named `out_x` is paired
with `in_x`. Each state must have exactly one symbolic non-batch sequence axis.
Paired state inputs and outputs must have matching dtypes, ranks, sequence axes,
and compatible static dimensions. Logits must have shape `[batch, vocab]` with
a static vocabulary dimension.

```bash
onnx-beam-export \
  --main-model ./models/Main.onnx \
  --output-folder ./models/beam_helpers \
  --beam-size 3 \
  --top-k 3
```

Without installing the package:

```bash
python examples/generic_export.py \
  --main-model ./models/Main.onnx \
  --output-folder ./models/beam_helpers
```

For a graph that does not follow the naming or symbolic-axis convention:

```bash
onnx-beam-export \
  --main-model ./models/Main.onnx \
  --output-folder ./models/beam_helpers \
  --logits-output token_logits \
  --state-inputs past_key,past_value \
  --state-outputs present_key,present_value \
  --sequence-axes 3,2
```

The exporter creates:

| File | Contract |
| --- | --- |
| `BeamSearch_First.onnx` | Expand first-step state and select initial beams. |
| `BeamSearch_Next.onnx` | Select global candidates and gather parent state. |
| `BeamSearch_GatherFirst.onnx` | Select state row zero when leaving beam mode. |
| `BeamSearch_ConcatFirst.onnx` | Join a one-row prefix to beam row zero and broadcast it. |

## Python Export API

```python
from onnx_beam_search import (
    discover_main_graph_contract,
    export_beam_search_graphs,
)

contract = discover_main_graph_contract("models/Main.onnx")
graphs = export_beam_search_graphs(
    contract.state_specs,
    contract.vocab_size,
    "models/beam_helpers",
    beam_size=3,
    top_k=3,
    opset=contract.opset,
    logits_dtype=contract.logits_dtype,
)
```

Construct `BeamStateSpec` objects directly when state cannot be inferred from
the graph interface.

## Model Integration Contract

The four helpers are independent ONNX graphs. Integrate them into a model's own
prefill and decode pipeline using that model's graph-composition process. This
boundary is intentional: rotary encoding, attention masks, cache layout,
penalties, and graph naming vary between architectures.

`BeamSearchRunner` consumes one merged prefill graph and one merged decode graph
with this interface:

1. The leading `state_count` inputs and outputs are recurrent state in matching order.
2. Prefill accepts one state row and emits `beam_size` state rows.
3. Decode accepts selected token IDs, prior scores, saved IDs, and sequence length.
4. Both graphs emit exactly five values after their state outputs:

```text
saved_ids, cumulative_scores, next_token_ids, best_token_id, sequence_length
```

Names of non-state inputs are configured with `BeamGraphIO`. State is matched
positionally, so no model-specific prefixes are required. Runner construction
checks recurrent-state and beam-control dtypes, ranks, and static dimensions
across the prefill-to-decode and decode-to-decode boundaries.

## Raw-Token Runtime

The runtime deliberately accepts and returns token IDs. Tokenization, prompt
formatting, stop-token selection, and text decoding belong to the caller.

Copy `examples/beam_manifest.example.json` and configure:

- Prefill and decode model paths.
- Optional shared-initializer model path.
- State count and maximum sequence length.
- Non-state input names.
- Fixed prefill/decode inputs.
- Stop token IDs.

Manifest paths are resolved relative to the JSON file. The placeholder
`$prompt_length` is replaced recursively inside `prefill_inputs`. Manifest
field types are validated without coercing strings, numbers, booleans, or
arrays into another type.

```bash
onnx-beam-run \
  --manifest ./beam_manifest.json \
  --input-ids 1,42,7 \
  --beam-size 3 \
  --top-k 3 \
  --max-new-tokens 128
```

Without installing the package:

```bash
python examples/generic_llm_demo.py \
  --manifest ./beam_manifest.json \
  --input-ids '[1, 42, 7]' \
  --beam-size 3 \
  --top-k 3
```

The command prints generated token IDs and phase timings as JSON.

## Python Runtime API

```python
import numpy as np

from onnx_beam_search import (
    BeamGraphIO,
    BeamModelSpec,
    BeamSearchRunner,
)

model = BeamModelSpec(
    prefill_model="models/PrefillBeam.onnx",
    decode_model="models/DecodeBeam.onnx",
    shared_initializers=None,
    state_count=64,
    max_sequence_length=4096,
    io=BeamGraphIO(),
)
runner = BeamSearchRunner(model)
result = runner.generate(
    np.array([[1, 42, 7]], dtype=np.int64),
    beam_size=3,
    top_k=3,
    stop_token_ids=(2,),
    max_new_tokens=128,
    prefill_inputs={
        "ids_len": np.array([3]),
        "history_len": np.array([0]),
        "cache_len": np.array([0]),
    },
)
print(result.token_ids)
```

Pass `initial_states` when empty state cannot be represented by a zero-length
symbolic axis. Pass model-specific fixed controls through `prefill_inputs` and
`decode_inputs`. The NumPy-backed runtime rejects BF16 graph inputs during
runner construction because ONNX Runtime cannot construct BF16 OrtValues from
NumPy arrays; use a NumPy-compatible input dtype for merged runtime graphs.

## Runtime Mechanics

- Shared initializers can be memory-mapped and injected with `add_initializer`.
- Growing state and saved-ID outputs are rebound on every step.
- Prior outputs are bound as inputs before dynamic outputs are rebound.
- Two I/O bindings ping-pong across decode steps.
- Fixed-size token, score, and length controls reuse peer buffers after warm-up.
- Final output is read from the best beam's saved-ID row.

Configure execution providers with `OrtProviderConfig`. CPU, CUDA, DirectML,
and CPU-backed execution providers are supported; provider options remain under
caller control.

## Integrating Another Model

No toolkit source change should be necessary:

1. Discover or construct the model's `BeamStateSpec` values.
2. Export the four helpers with the desired filenames and dtypes.
3. Compose first-step and next-step helpers into the model's graph pipeline.
4. Describe the merged graph names with `BeamGraphIO` and `BeamModelSpec`.
5. Supply model-specific scalar inputs and stop IDs from application code.
6. Keep tokenizer and prompt logic outside this folder.

## Test

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  python -m unittest -v tests.test_toolkit
```

The tests use synthetic ONNX graphs. They validate contract discovery, all four
helper exports, ONNX checking, raw-token generation, stop tokens, manifests,
dynamic state growth, I/O-binding behavior, malformed interface rejection, and
strict manifest typing without any external model.