"""Exercise real manager/worker methods and metadata-only DataProto without Ray."""

import ast
import asyncio
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List
from unittest.mock import Mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]


def load_dispatch():
    namespace = {
        "np": np,
        "torch": torch,
        "dataclass": dataclass,
        "field": field,
        "Dict": Dict,
        "List": List,
        "TensorDict": object,
        "DataProtoConfig": SimpleNamespace(auto_padding=False, auto_padding_key="_verl_auto_padding"),
    }
    path = ROOT / "verl/protocol.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    protocol = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DataProto")
    methods = {
        "__post_init__",
        "check_consistency",
        "__len__",
        "__getitem__",
        "slice",
        "chunk",
        "concat",
        "repeat",
        "is_padding_enabled",
    }
    protocol.body = [
        node for node in protocol.body if isinstance(node, ast.AnnAssign) or getattr(node, "name", None) in methods
    ]
    helper = next(node for node in tree.body if getattr(node, "name", None) == "list_of_dict_to_dict_of_list")
    exec(compile(ast.Module(body=[helper, protocol], type_ignores=[]), str(path), "exec"), namespace)

    path = ROOT / "verl/experimental/agent_loop/agent_loop.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    manager = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AgentLoopManager")
    manager.body = [node for node in manager.body if getattr(node, "name", None) == "generate_sequences"]
    worker = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AgentLoopWorker")
    worker.body = [node for node in worker.body if getattr(node, "name", None) == "generate_sequences"]
    worker.decorator_list = []
    namespace.update(asyncio=asyncio, ray=SimpleNamespace(get=lambda refs: refs))
    exec(compile(ast.Module(body=[manager, worker], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["DataProto"], namespace["AgentLoopManager"], namespace["AgentLoopWorker"]


DataProto, Manager, Worker = load_dispatch()


class Config(dict):
    __getattr__ = dict.__getitem__


def object_array(values):
    result = np.empty(len(values), dtype=object)
    result[:] = values
    return result


class RemoteWorker:
    """Run the real async worker locally; replace model generation and tensor padding."""

    def __init__(self, config):
        self.config = config
        self.received = []
        self.samples = []
        self.generate_sequences = SimpleNamespace(remote=self.remote)

    def remote(self, batch):
        self.received.append(batch)
        return asyncio.run(Worker.generate_sequences(self, batch))

    async def _run_agent_loop(self, agent_name, messages, sampling_params, context):
        result = {"case": messages[0]["content"], "context": context["case"], "agent": agent_name}
        self.samples.append(result)
        await asyncio.sleep(0)
        return result

    def _postprocess(self, outputs):
        self.assert_nonempty(outputs)
        return DataProto(
            non_tensor_batch={"bfcl_outcome": object_array(outputs)},
            meta_info={"metrics": [{"generate_sequences": 1.0, "tool_calls": 0.0} for _ in outputs]},
        )

    @staticmethod
    def assert_nonempty(outputs):
        if not outputs:
            raise AssertionError("An idle worker must not receive an empty batch")


def inputs(size, validate=False, auto_padding=False):
    return DataProto(
        non_tensor_batch={
            "uid": object_array([f"uid_{i}" for i in range(size)]),
            "raw_prompt": object_array([[{"role": "user", "content": f"case_{i}"}] for i in range(size)]),
            "bfcl_context": object_array([{"case": f"case_{i}"} for i in range(size)]),
            "agent_name": object_array(["tool_agent"] * size),
        },
        meta_info={"validate": validate, "_verl_auto_padding": auto_padding},
    )


def manager(num_workers, n=2, free_cache_engine=True):
    instance = Manager()
    rollout = Config(
        n=n,
        temperature=0.6,
        top_p=1.0,
        free_cache_engine=free_cache_engine,
        val_kwargs=Config(temperature=0.0, top_p=1.0),
    )
    instance.config = Config(actor_rollout_ref=Config(rollout=rollout))
    instance.agent_loop_workers = [RemoteWorker(instance.config) for _ in range(num_workers)]
    instance.wake_up = Mock()
    instance.sleep = Mock()
    instance._performance_metrics = Mock(side_effect=lambda metrics, output: {"rows": len(output)})
    return instance


class TestAgentDispatch(unittest.TestCase):
    def assert_dispatch(self, size, workers, n, validate=False, auto_padding=False):
        dispatch = manager(workers, n)
        batch = inputs(size, validate, auto_padding)
        result = dispatch.generate_sequences(batch)
        repeats = 1 if validate else n
        expected = [f"case_{i}" for i in range(size) for _ in range(repeats)]
        actual = result.non_tensor_batch["bfcl_outcome"].tolist()
        self.assertEqual([row["case"] for row in actual], expected)
        self.assertEqual([row["context"] for row in actual], expected)
        self.assertTrue(all(row["agent"] == "tool_agent" for row in actual))
        self.assertEqual(len(result), size * repeats)
        chunks = [chunk for worker in dispatch.agent_loop_workers for chunk in worker.received]
        self.assertEqual(len(chunks), min(size, workers))
        self.assertTrue(all(len(chunk) > 0 for chunk in chunks))
        sizes = [len(chunk) for chunk in chunks]
        self.assertLessEqual(max(sizes) - min(sizes), 1)
        self.assertEqual(
            [uid for chunk in chunks for uid in chunk.non_tensor_batch["uid"]], [f"uid_{i}" for i in range(size)]
        )
        self.assertTrue(all(chunk.meta_info["validate"] == validate for chunk in chunks))
        metrics, combined = dispatch._performance_metrics.call_args.args
        self.assertEqual(sum(len(chunk) for chunk in metrics), size * repeats)
        self.assertIs(combined, result)
        dispatch.wake_up.assert_called_once()
        dispatch.sleep.assert_called_once()
        self.assertEqual(result.meta_info["timing"], {"rows": size * repeats})

    def test_smoke_two_prompts_eight_workers_repeat_only_after_dispatch(self):
        self.assert_dispatch(2, 8, 2)

    def test_single_prompt_subset_full_batches(self):
        self.assert_dispatch(1, 8, 8)

    def test_uneven_and_divisible_batches_preserve_groups_and_contexts(self):
        for size, workers, n in ((3, 2, 2), (5, 2, 3), (10, 8, 2), (16, 8, 2), (2, 1, 2)):
            with self.subTest(size=size, workers=workers, n=n):
                self.assert_dispatch(size, workers, n)

    def test_validation_does_not_repeat_each_prompt(self):
        self.assert_dispatch(3, 8, 8, validate=True)

    def test_auto_padding_flag_does_not_create_extra_rollouts(self):
        self.assert_dispatch(2, 8, 2, auto_padding=True)

    def test_empty_batch_or_worker_pool_fails_before_waking_models(self):
        for size, workers in ((0, 8), (2, 0)):
            with self.subTest(size=size, workers=workers):
                dispatch = manager(workers)
                with self.assertRaises(ValueError):
                    dispatch.generate_sequences(inputs(size))
                dispatch.wake_up.assert_not_called()

    def test_dispatch_without_cache_sleep_wakeup(self):
        dispatch = manager(8, free_cache_engine=False)
        result = dispatch.generate_sequences(inputs(2))
        self.assertEqual(len(result), 4)
        dispatch.wake_up.assert_not_called()
        dispatch.sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
