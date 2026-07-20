"""Model-independent tests for the standalone ONNX beam-search toolkit."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import torch
from onnx import TensorProto, helper

from onnx_beam_search import (
    BeamModelSpec,
    BeamSearchRunner,
    FirstBeamSearch,
    NextBeamSearch,
    discover_main_graph_contract,
    export_beam_search_graphs,
    load_beam_search_manifest,
)


class _SyntheticPrefill(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.beam = FirstBeamSearch(1, torch.int64)
        self.register_buffer(
            "logits", torch.tensor([[0.0, 1.0, 10.0, 3.0, 2.0, -1.0]])
        )

    def forward(self, state, input_ids, prompt_length, save_id_in, beam_size):
        state = torch.cat([state, input_ids.float().unsqueeze(1)], dim=-1)
        logits = self.logits.expand(input_ids.shape[0], -1)
        outputs = self.beam(state, logits, save_id_in, beam_size)
        return (*outputs, prompt_length)


class _SyntheticDecode(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.beam = NextBeamSearch(1, torch.int64)
        self.register_buffer(
            "logits", torch.tensor([[0.0, 1.0, 2.0, 3.0, 10.0, -1.0]])
        )

    def forward(
        self,
        state,
        kv_seq_len,
        input_ids,
        previous_prob,
        save_id_in,
        beam_size,
        topK,
    ):
        state = torch.cat([state, input_ids.float().unsqueeze(1)], dim=-1)
        logits = self.logits.expand(input_ids.shape[0], -1)
        outputs = self.beam(
            state,
            logits,
            save_id_in,
            previous_prob,
            beam_size,
            topK,
        )
        return (*outputs, kv_seq_len + 1)


class StandaloneBeamSearchTest(unittest.TestCase):

    def setUp(self):
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.folder = Path(self._temporary_directory.name)

    def tearDown(self):
        self._temporary_directory.cleanup()

    def _write_synthetic_main(self):
        graph = helper.make_graph(
            [
                helper.make_node("Identity", ["in_key"], ["out_key"]),
                helper.make_node("Identity", ["in_value"], ["out_value"]),
                helper.make_node("Identity", ["logits_in"], ["logits"]),
            ],
            "GenericMain",
            [
                helper.make_tensor_value_info(
                    "in_key", TensorProto.UINT8, ["batch", 2, 4, "history_len"]
                ),
                helper.make_tensor_value_info(
                    "in_value", TensorProto.FLOAT16, ["batch", 2, "history_len", 4]
                ),
                helper.make_tensor_value_info(
                    "logits_in", TensorProto.FLOAT, ["batch", 8]
                ),
            ],
            [
                helper.make_tensor_value_info(
                    "out_key", TensorProto.UINT8, ["batch", 2, 4, "kv_seq_len"]
                ),
                helper.make_tensor_value_info(
                    "out_value", TensorProto.FLOAT16, ["batch", 2, "kv_seq_len", 4]
                ),
                helper.make_tensor_value_info(
                    "logits", TensorProto.FLOAT, ["batch", 8]
                ),
            ],
        )
        model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", 20)],
            producer_name="StandaloneBeamSearchTest",
        )
        path = self.folder / "GenericMain.onnx"
        onnx.save(model, path)
        return path

    def _write_interface_model(
        self,
        name,
        *,
        state_input_type=TensorProto.FLOAT,
        state_output_type=TensorProto.FLOAT,
        state_input_shape=("batch", 2, "history_len"),
        state_output_shape=("batch", 2, "kv_seq_len"),
        logits_shape=("batch", 8),
    ):
        state_node = (
            helper.make_node("Identity", ["in_state"], ["out_state"])
            if state_input_type == state_output_type
            else helper.make_node(
                "Cast",
                ["in_state"],
                ["out_state"],
                to=state_output_type,
            )
        )
        graph = helper.make_graph(
            [
                state_node,
                helper.make_node("Identity", ["logits_in"], ["logits"]),
            ],
            name,
            [
                helper.make_tensor_value_info(
                    "in_state", state_input_type, state_input_shape
                ),
                helper.make_tensor_value_info(
                    "logits_in", TensorProto.FLOAT, logits_shape
                ),
            ],
            [
                helper.make_tensor_value_info(
                    "out_state", state_output_type, state_output_shape
                ),
                helper.make_tensor_value_info(
                    "logits", TensorProto.FLOAT, logits_shape
                ),
            ],
        )
        model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", 20)],
            producer_name="StandaloneBeamSearchTest",
        )
        path = self.folder / f"{name}.onnx"
        onnx.save(model, path)
        return path

    @staticmethod
    def _export_synthetic_runtime(prefill_path, decode_path):
        state = torch.zeros((1, 1, 1), dtype=torch.float32)
        prompt_ids = torch.tensor([[5, 6]], dtype=torch.int64)
        prompt_length = torch.tensor([2], dtype=torch.int64)
        save_ids = torch.zeros((2, 0), dtype=torch.int64)
        beam_size = torch.tensor([2], dtype=torch.int64)
        output_names = [
            "out_state",
            "save_id_out",
            "scores_out",
            "next_tokens_out",
            "best_token_out",
            "kv_seq_len_out",
        ]
        torch.onnx.export(
            _SyntheticPrefill().eval(),
            (state, prompt_ids, prompt_length, save_ids, beam_size),
            str(prefill_path),
            input_names=[
                "in_state",
                "input_ids",
                "prompt_length",
                "save_id_in",
                "beam_size",
            ],
            output_names=output_names,
            dynamic_axes={
                "in_state": {0: "prefill_batch", 2: "history_len"},
                "input_ids": {1: "prompt_len"},
                "save_id_in": {0: "beam_size", 1: "saved_len"},
                "out_state": {0: "beam_size", 2: "kv_seq_len"},
                "save_id_out": {0: "beam_size", 1: "saved_len_out"},
                "scores_out": {0: "beam_size"},
                "next_tokens_out": {0: "beam_size"},
            },
            opset_version=20,
            dynamo=False,
        )

        beam_state = torch.zeros((2, 1, 3), dtype=torch.float32)
        decode_ids = torch.tensor([[2], [3]], dtype=torch.int64)
        previous_scores = torch.zeros((2, 1), dtype=torch.float32)
        saved = torch.tensor([[2], [3]], dtype=torch.int64)
        top_k = torch.tensor([2], dtype=torch.int32)
        torch.onnx.export(
            _SyntheticDecode().eval(),
            (
                beam_state,
                prompt_length,
                decode_ids,
                previous_scores,
                saved,
                beam_size,
                top_k,
            ),
            str(decode_path),
            input_names=[
                "in_state",
                "kv_seq_len",
                "input_ids",
                "previous_prob",
                "save_id_in",
                "beam_size",
                "topK",
            ],
            output_names=output_names,
            dynamic_axes={
                "in_state": {0: "beam_size", 2: "history_len"},
                "input_ids": {0: "beam_size"},
                "previous_prob": {0: "beam_size"},
                "save_id_in": {0: "beam_size", 1: "saved_len"},
                "out_state": {0: "beam_size", 2: "kv_seq_len"},
                "save_id_out": {0: "beam_size", 1: "saved_len_out"},
                "scores_out": {0: "beam_size"},
                "next_tokens_out": {0: "beam_size"},
            },
            opset_version=20,
            dynamo=False,
        )

    def test_discover_and_export_without_model_metadata(self):
        main_path = self._write_synthetic_main()
        contract = discover_main_graph_contract(main_path)
        self.assertEqual(contract.vocab_size, 8)
        self.assertEqual(contract.opset, 20)
        self.assertEqual([spec.sequence_axis for spec in contract.state_specs], [3, 2])

        exported = export_beam_search_graphs(
            contract.state_specs,
            contract.vocab_size,
            self.folder / "helpers",
            beam_size=2,
            top_k=2,
            opset=contract.opset,
            token_dtype=torch.int64,
            logits_dtype=contract.logits_dtype,
        )
        self.assertEqual(
            set(exported),
            {"first_beam", "next_beam", "gather_first_beam", "concat_first_beam"},
        )
        for path in exported.values():
            onnx.checker.check_model(onnx.load(str(path), load_external_data=False))

        first = onnxruntime.InferenceSession(
            str(exported["first_beam"]), providers=["CPUExecutionProvider"]
        )
        key = np.zeros((1, 2, 4, 3), dtype=np.uint8)
        value = np.zeros((1, 2, 3, 4), dtype=np.float16)
        outputs = dict(zip(
            [value.name for value in first.get_outputs()],
            first.run(None, {
                "in_key": key,
                "in_value": value,
                "logits": np.array(
                    [[0.0, 8.0, 7.0, 1.0, 0.0, -1.0, -2.0, -3.0]],
                    dtype=np.float32,
                ),
                "save_id_in": np.zeros((2, 0), dtype=np.int64),
                "beam_size": np.array([2], dtype=np.int64),
            }),
        ))
        np.testing.assert_array_equal(
            outputs["top_beam_indices"], np.array([[1], [2]], dtype=np.int64)
        )
        self.assertEqual(outputs["out_key"].shape[0], 2)

    def test_discovery_rejects_incompatible_interfaces(self):
        cases = (
            (
                "StateDtypeMismatch",
                {"state_output_type": TensorProto.INT32},
                "different dtypes",
            ),
            (
                "StateRankMismatch",
                {"state_output_shape": ("batch", 2, 1, "kv_seq_len")},
                "different ranks",
            ),
            (
                "StateStaticDimensionMismatch",
                {"state_output_shape": ("batch", 3, "kv_seq_len")},
                "incompatible static dimensions",
            ),
            (
                "StateSequenceAxisMismatch",
                {"state_input_shape": ("batch", "history_len", 2)},
                "do not include output sequence axis",
            ),
            (
                "RankThreeLogits",
                {"logits_shape": ("batch", 1, 8)},
                "must have rank 2",
            ),
        )
        for name, arguments, message in cases:
            with self.subTest(name=name):
                path = self._write_interface_model(name, **arguments)
                with self.assertRaisesRegex(ValueError, message):
                    discover_main_graph_contract(path)

    def test_runtime_generates_raw_token_ids_without_a_tokenizer(self):
        prefill_path = self.folder / "prefill.onnx"
        decode_path = self.folder / "decode.onnx"
        self._export_synthetic_runtime(prefill_path, decode_path)
        manifest_path = self.folder / "beam.json"
        manifest_path.write_text(json.dumps({
            "model": {
                "prefill_model": prefill_path.name,
                "decode_model": decode_path.name,
                "state_count": 1,
                "max_sequence_length": 16,
            },
            "prefill_inputs": {"prompt_length": ["$prompt_length"]},
            "decode_inputs": {},
            "stop_token_ids": [4],
        }))
        manifest = load_beam_search_manifest(manifest_path)
        self.assertEqual(
            manifest.resolved_prefill_inputs(2), {"prompt_length": [2]}
        )
        runner = BeamSearchRunner(manifest.model)
        result = runner.generate(
            np.array([[5, 6]], dtype=np.int64),
            beam_size=2,
            top_k=2,
            max_new_tokens=3,
            prefill_inputs=manifest.resolved_prefill_inputs(2),
        )
        self.assertEqual(result.token_ids, (2, 4, 4))
        self.assertEqual(result.prompt_tokens, 2)
        self.assertEqual(result.generated_tokens, 3)

        stopped = runner.generate(
            np.array([[5, 6]], dtype=np.int64),
            beam_size=2,
            top_k=2,
            stop_token_ids=manifest.stop_token_ids,
            max_new_tokens=3,
            prefill_inputs=manifest.resolved_prefill_inputs(2),
        )
        self.assertEqual(stopped.token_ids, (2,))

    def test_runtime_rejects_incompatible_state_metadata(self):
        prefill_path = self.folder / "prefill.onnx"
        decode_path = self.folder / "decode.onnx"
        self._export_synthetic_runtime(prefill_path, decode_path)

        decode_model = onnx.load(decode_path, load_external_data=False)
        state_input = next(
            value for value in decode_model.graph.input
            if value.name == "in_state"
        )
        state_input.type.tensor_type.elem_type = TensorProto.INT64
        for node in decode_model.graph.node:
            for index, input_name in enumerate(node.input):
                if input_name == "in_state":
                    node.input[index] = "in_state_as_float"
        decode_model.graph.node.insert(
            0,
            helper.make_node(
                "Cast",
                ["in_state"],
                ["in_state_as_float"],
                to=TensorProto.FLOAT,
            ),
        )
        onnx.checker.check_model(decode_model)
        onnx.save(decode_model, decode_path)

        with self.assertRaisesRegex(
            RuntimeError, "Recurrent state 0 from prefill to decode dtype mismatch"
        ):
            BeamSearchRunner(BeamModelSpec(
                prefill_model=prefill_path,
                decode_model=decode_path,
                state_count=1,
                max_sequence_length=16,
            ))

    def test_runtime_rejects_bfloat16_inputs_clearly(self):
        prefill_path = self.folder / "prefill_bfloat16.onnx"
        decode_path = self.folder / "decode_bfloat16.onnx"
        self._export_synthetic_runtime(prefill_path, decode_path)

        prefill_model = onnx.load(prefill_path, load_external_data=False)
        state_input = next(
            value for value in prefill_model.graph.input
            if value.name == "in_state"
        )
        state_input.type.tensor_type.elem_type = TensorProto.BFLOAT16
        for node in prefill_model.graph.node:
            for index, input_name in enumerate(node.input):
                if input_name == "in_state":
                    node.input[index] = "in_state_as_float"
        prefill_model.graph.node.insert(
            0,
            helper.make_node(
                "Cast",
                ["in_state"],
                ["in_state_as_float"],
                to=TensorProto.FLOAT,
            ),
        )
        onnx.checker.check_model(prefill_model)
        onnx.save(prefill_model, prefill_path)

        with self.assertRaisesRegex(
            ValueError, "BF16 inputs cannot be constructed"
        ):
            BeamSearchRunner(BeamModelSpec(
                prefill_model=prefill_path,
                decode_model=decode_path,
                state_count=1,
                max_sequence_length=16,
            ))

    def test_manifest_rejects_malformed_field_types(self):
        base_model = {
            "prefill_model": "prefill.onnx",
            "decode_model": "decode.onnx",
            "state_count": 1,
            "max_sequence_length": 16,
        }
        cases = (
            (
                "StringBoolean",
                {"model": {**base_model, "activations_fp16": "false"}},
                "model.activations_fp16 must be a boolean",
            ),
            (
                "StringStateCount",
                {"model": {**base_model, "state_count": "1"}},
                "model.state_count must be an integer",
            ),
            (
                "StringSaveIds",
                {
                    "model": {
                        **base_model,
                        "io": {"decode_save_ids": "save_id_in"},
                    }
                },
                "model.io.decode_save_ids must be a JSON array",
            ),
            (
                "NullStopIds",
                {"model": base_model, "stop_token_ids": None},
                "stop_token_ids must be a JSON array of integers",
            ),
        )
        for name, manifest_data, message in cases:
            with self.subTest(name=name):
                path = self.folder / f"{name}.json"
                path.write_text(json.dumps(manifest_data), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    load_beam_search_manifest(path)


if __name__ == "__main__":
    unittest.main()