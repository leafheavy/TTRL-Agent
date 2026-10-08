"""Check consensus, BFCL sessions, action masks and post-normalization weights."""

import asyncio
import json
import re
import sys
import unittest
from copy import deepcopy
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import torch
from jsonschema.exceptions import SchemaError

from data.preprocess import make_bfcl_map_fn
from verl.experimental.agent_loop.agent_loop import AgentLoopWorker
from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.reward import compute_reward, load_reward_manager
from verl.trainer.ppo.ttrl_utils import _majority_vote, apply_ttrl_q_weights, validate_q_weight_config
from verl.utils.dataset.rl_dataset import bfcl_chat_template_kwargs
from verl.utils.reward_score import ttrl_bfcl
from verl.utils.reward_score.ttrl_consensus import consensus_key, vote_consensus

try:
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import execute_multi_turn_func_call
    from bfcl_eval.model_handler.utils import convert_to_function_call
except ImportError:
    BFCL_RUNTIME_AVAILABLE = False
else:
    BFCL_RUNTIME_AVAILABLE = True

FUNCTIONS = [
    {
        "name": "weather.get",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "days": {"type": "array", "items": {"type": "integer"}}},
            "required": ["city"],
        },
    }
]


def tool_call(arguments, name="weather.get"):
    return "<tool_call>" + json.dumps({"name": name, "arguments": arguments}) + "</tool_call>"


def parser_stub(text):
    """Model BFCL's documented behavior of silently skipping malformed JSON."""
    calls = []
    for payload in re.findall(r"<tool_call>\n(.*?)\n</tool_call>", text, re.DOTALL):
        try:
            calls.append(json.loads(payload))
        except ValueError:
            continue
    return calls


class PoisonOracle(dict):
    def __getitem__(self, key):
        raise AssertionError("A reference answer was accessed")

    def get(self, key, default=None):
        raise AssertionError("A reference answer was accessed")


class Batch:
    def __init__(self, length, tensors=None, metadata=None):
        self.length = length
        self.batch = tensors or {}
        self.non_tensor_batch = metadata or {}

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return SimpleNamespace(
            batch={key: value[index] for key, value in self.batch.items()},
            non_tensor_batch={key: value[index] for key, value in self.non_tensor_batch.items()},
        )


class Tokenizer:
    eos_token = "<END>"

    def decode(self, ids, skip_special_tokens=False):
        assert not skip_special_tokens  # Preserve the model's tool-call delimiters.
        values = ids.tolist() if hasattr(ids, "tolist") else ids
        return "".join(chr(i) for i in values)


def make_batches(responses, n):
    prompt_count = len(responses) // n
    batch = Batch(
        prompt_count,
        metadata={
            "extra_info": np.array(
                [{"bfcl": {"functions": deepcopy(FUNCTIONS)}} for _ in range(prompt_count)], dtype=object
            ),
            "reward_model": np.array([PoisonOracle() for _ in range(prompt_count)], dtype=object),
        },
    )
    width = max(map(len, responses)) + 3
    ids = torch.zeros((len(responses), width), dtype=torch.long)
    valid = torch.zeros_like(ids)
    for row, text in enumerate(responses):
        ids[row, : len(text)] = torch.tensor([ord(char) for char in text], dtype=torch.long)
        valid[row, : len(text)] = 1
    generated = Batch(
        len(responses),
        tensors={
            "prompts": torch.zeros((len(responses), 2), dtype=torch.long),
            "responses": ids,
            "attention_mask": torch.cat([torch.ones((len(responses), 2), dtype=torch.long), valid], dim=1),
            "response_mask": valid.clone(),
        },
    )
    return batch, generated


class TestBFCLRewards(unittest.TestCase):
    def setUp(self):
        self.decoder = Mock(side_effect=parser_stub)
        self.decoder_patch = patch.object(ttrl_bfcl, "get_bfcl_decoder", return_value=self.decoder)
        self.decoder_patch.start()
        self.addCleanup(self.decoder_patch.stop)

    def vote(self, responses, functions=FUNCTIONS):
        validators = ttrl_bfcl._function_validators(functions)
        keys = []
        for response in responses:
            calls = ttrl_bfcl.decode_bfcl_calls(response, validators, self.decoder)
            keys.append(
                consensus_key({"calls": calls, "explicit_no_call": response.strip() == "[]"}, "calls")
                if calls is not None else None
            )
        return vote_consensus(keys)

    def test_parameter_dictionary_order_is_normalized(self):
        result = self.vote(
            [
                tool_call({"city": "Shanghai", "days": [1, 2]}),
                tool_call({"days": [1, 2], "city": "Shanghai"}),
                tool_call({"city": "Beijing", "days": [1, 2]}),
            ]
        )
        self.assertEqual(result["scores"], [1, 1, 0])
        self.assertEqual(result["keys"][0], result["keys"][1])
        self.assertAlmostEqual(result["majority_ratio"], 2 / 3)

    def test_shared_voting_helper_preserves_the_math_recipe(self):
        math = ModuleType("verl.utils.reward_score.ttrl_math")
        math.extract_answer = lambda output: output
        math.simplify_expression_string = lambda answer: answer.strip()
        with patch.dict(sys.modules, {math.__name__: math}):
            self.assertEqual(_majority_vote([None, " B ", "A", "B"]), ("B", 0.5))
            self.assertEqual(_majority_vote([None, None]), ("None", 0))
            self.assertEqual(_majority_vote(["B", "A"]), ("B", 0.5))

    def test_call_and_parameter_list_order_are_preserved(self):
        first, second = tool_call({"city": "A"}), tool_call({"city": "B"})
        result = self.vote([first + second, second + first])
        self.assertNotEqual(*result["keys"])
        self.assertTrue(result["tied"])
        self.assertEqual(result["scores"], [1, 0])  # Preserve the original first-observed TTRL tie rule.
        result = self.vote([tool_call({"city": "A", "days": [1, 2]}), tool_call({"city": "A", "days": [2, 1]})])
        self.assertNotEqual(*result["keys"])

    def test_hallucinated_or_invalid_arguments_do_not_vote(self):
        responses = [
            tool_call({"city": "A"}, "unknown"),
            tool_call({}),
            tool_call({"city": 42}),
            tool_call({"city": "A", "extra": 1}),
            tool_call({"city": "A", "days": [True]}),
            tool_call("not-an-object"),
            tool_call({"city": "A", "days": [float("nan")]}),
        ]
        result = self.vote(responses)
        self.assertEqual(result["scores"], [0] * len(responses))
        self.assertIsNone(result["majority_key"])
        self.assertFalse(result["group_valid"])
        self.assertEqual(result["majority_ratio"], 0)

    def test_malformed_call_cannot_disappear_from_a_valid_prefix(self):
        good = tool_call({"city": "A"})
        result = self.vote([good, good, good + "<tool_call>{broken}</tool_call>", good + "<tool_call>{truncated"])
        self.assertEqual(result["scores"], [1, 1, 0, 0])
        self.assertEqual(result["valid"], [True, True, False, False])
        self.assertEqual(result["majority_ratio"], 0.5)  # Includes invalid rollouts in N.

    def test_parser_failures_and_empty_text_are_not_no_call_results(self):
        result = self.vote(["", "unstructured refusal", "{broken}", "<tool_call>{broken}</tool_call>"])
        self.assertEqual(result["scores"], [0, 0, 0, 0])
        self.assertFalse(result["informative"])
        result = self.vote(["[]", " [] ", "", tool_call({"city": "A"})])
        self.assertEqual(result["scores"], [1, 1, 0, 0])
        self.assertEqual(result["majority_key"], consensus_key({"calls": [], "explicit_no_call": True}, "calls"))

    def test_invalid_input_schema_is_a_data_error(self):
        malformed = deepcopy(FUNCTIONS)
        malformed[0]["parameters"]["properties"]["city"]["type"] = "unsupported-type"
        with self.assertRaises(SchemaError):
            self.vote([tool_call({"city": "A"})], malformed)
        duplicate = FUNCTIONS + FUNCTIONS
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.vote([tool_call({"city": "A"})], duplicate)

    def test_openai_wrapped_function_schema_is_accepted(self):
        result = self.vote([tool_call({"city": "A"})], [{"type": "function", "function": FUNCTIONS[0]}])
        self.assertEqual(result["scores"], [1])

    def test_token_rewards_and_grpo_use_the_same_unlabeled_samples(self):
        good, other = tool_call({"city": "A"}), tool_call({"city": "B"})
        responses = [good, good, other, "", "broken", "", "bad", ""]
        batch, generated = make_batches(responses, 4)
        self.assertIs(ttrl_bfcl.apply_bfcl_rewards(batch, generated, 4, Tokenizer()), batch)
        scores = generated.batch["ttrl_bfcl_scores"]
        self.assertEqual(scores.sum(-1).tolist(), [1, 1, 0, 0, 0, 0, 0, 0])
        self.assertTrue(torch.all(scores[:, -3:] == 0))  # No reward on padding or at negative index.
        self.assertEqual(batch.non_tensor_batch["majority_ratio_list"].tolist(), [0.5, 0])
        advantages, _ = compute_grpo_outcome_advantage(
            scores, generated.batch["response_mask"], np.array(["a"] * 4 + ["b"] * 4)
        )
        self.assertGreater(advantages[0, 0].item(), 0)
        self.assertLess(advantages[2, 0].item(), 0)
        self.assertTrue(torch.all(advantages[4:] == 0))  # All-invalid group contributes no task advantage.
        self.assertEqual(ttrl_bfcl.bfcl_reward(generated, True)["reward_tensor"].data_ptr(), scores.data_ptr())
        returned_scores, diagnostics = compute_reward(generated, ttrl_bfcl.bfcl_reward)
        self.assertIs(returned_scores, scores)
        self.assertEqual(diagnostics["bfcl_majority_ratio"], [0.5] * 4 + [0] * 4)
        metrics = ttrl_bfcl.compute_bfcl_metrics(generated)
        self.assertAlmostEqual(metrics["bfcl_consensus_reward"], 0.25)
        self.assertNotIn("reward_model", generated.non_tensor_batch)
        self.assertNotIn("original_gt", metrics)

    def test_no_call_eos_is_normalized_and_observations_are_rejected(self):
        batch, generated = make_batches(["[]<END>", "[]<END>"], 2)
        ttrl_bfcl.apply_bfcl_rewards(batch, generated, 2, Tokenizer())
        self.assertEqual(generated.batch["ttrl_bfcl_scores"].sum(-1).tolist(), [1, 1])
        batch, generated = make_batches([tool_call({"city": "A"})] * 2, 2)
        generated.batch["response_mask"][0, 0] = 0
        with self.assertRaisesRegex(ValueError, "tool observations"):
            ttrl_bfcl.apply_bfcl_rewards(batch, generated, 2, Tokenizer())

    def test_missing_metadata_or_unprepared_reward_fails(self):
        batch, generated = make_batches([tool_call({"city": "A"})] * 2, 2)
        batch.non_tensor_batch["extra_info"][0] = {}
        with self.assertRaisesRegex(ValueError, "extra_info.bfcl.functions"):
            ttrl_bfcl.apply_bfcl_rewards(batch, generated, 2, Tokenizer())
        with self.assertRaisesRegex(ValueError, "complete shared rollout group"):
            ttrl_bfcl.bfcl_reward(generated)

    def test_loader_ignores_custom_ground_truth_scorers(self):
        config = {"ttrl": {"enable": True, "reward_mode": "bfcl"}, "custom_reward_function": PoisonOracle()}
        self.assertIs(load_reward_manager(config, Tokenizer(), 0), ttrl_bfcl.bfcl_reward)

    def test_trainer_diagnostics_do_not_restore_ground_truth(self):
        batch, generated = make_batches([tool_call({"city": "A"}), "bad"], 2)
        ttrl_bfcl.apply_bfcl_rewards(batch, generated, 2, Tokenizer())
        generated.non_tensor_batch["reward_model"] = np.array([PoisonOracle(), PoisonOracle()], dtype=object)
        trainer = object.__new__(RayPPOTrainer)
        trainer.config = SimpleNamespace(ttrl={"reward_mode": "bfcl"})
        trainer.reward_fn = Mock(side_effect=AssertionError("Ground-truth scoring was called"))
        metrics = trainer._compute_ttrl_metrics(generated)
        trainer.reward_fn.assert_not_called()
        self.assertEqual(metrics["bfcl_consensus_reward"], 0.5)


class TestTrajectoryConsensus(unittest.TestCase):
    def test_action_observation_and_final_state_define_different_equivalence_relations(self):
        first = {"calls": [{"name": "add", "arguments": {"sku": "A", "quantity": 2}}],
                 "observations": [{"ok": True}], "final_state": {"cart": {"A": 2}}}
        other_item = {"calls": [{"name": "add", "arguments": {"sku": "B", "quantity": 2}}],
                      "observations": [{"ok": True}], "final_state": {"cart": {"B": 2}}}
        split = {"calls": [{"name": "add", "arguments": {"sku": "A", "quantity": 1}}] * 2,
                 "observations": [{"ok": True}] * 2, "final_state": {"cart": {"A": 2}}}
        self.assertNotEqual(consensus_key(first, "calls"), consensus_key(other_item, "calls"))
        self.assertEqual(consensus_key(first, "observations"), consensus_key(other_item, "observations"))
        self.assertNotEqual(consensus_key(first, "observations"), consensus_key(split, "observations"))
        self.assertEqual(consensus_key(first, "state"), consensus_key(split, "state"))

    def test_missing_evidence_does_not_turn_wording_or_empty_feedback_into_positive_rewards(self):
        trace = {"calls": [], "observations": [], "reply": "Please provide the SKU"}
        self.assertIsNone(consensus_key(trace, "calls"))
        self.assertIsNone(consensus_key(trace, "observations"))
        self.assertEqual(vote_consensus([None, None])["scores"], [0, 0])
        with self.assertRaisesRegex(ValueError, "mode"):
            consensus_key(trace, "calls_unordered")
        with self.assertRaisesRegex(ValueError, "mode"):
            consensus_key(trace, ["calls"])
        with self.assertRaisesRegex(ValueError, "final-state projection"):
            consensus_key({"final_state": {}}, "state")

    def test_vote_preserves_binary_rewards_denominator_and_first_observed_ties(self):
        result = vote_consensus([None, "B", "A", "B", "A", None])
        self.assertEqual(result["scores"], [0, 1, 0, 1, 0, 0])
        self.assertEqual(result["majority_ratio"], 2 / 6)
        self.assertTrue(result["tied"])
        self.assertEqual(vote_consensus(["A"] * 3)["scores"], [1, 1, 1])


FS_FUNCTIONS = [
    {
        "name": "mkdir",
        "parameters": {"type": "object", "properties": {"dir_name": {"type": "string"}}, "required": ["dir_name"]},
    },
    {
        "name": "cd",
        "parameters": {"type": "object", "properties": {"folder": {"type": "string"}}, "required": ["folder"]},
    },
    {"name": "pwd", "parameters": {"type": "object", "properties": {}}},
]


def fs_context(**overrides):
    context = {"functions": deepcopy(FS_FUNCTIONS), "initial_config": {}, "involved_classes": ["GorillaFileSystem"]}
    context.update(overrides)
    return context


def fs_call(name, **arguments):
    return {"name": name, "arguments": arguments}


@unittest.skipUnless(BFCL_RUNTIME_AVAILABLE, "Requires the existing BFCL executor package")
class TestBFCLSession(unittest.TestCase):
    def setUp(self):
        self.runtime_patch = patch.object(
            ttrl_bfcl, "get_bfcl_runtime", return_value=(execute_multi_turn_func_call, convert_to_function_call)
        )
        self.runtime_patch.start()
        self.addCleanup(self.runtime_patch.stop)
        self.decoder_patch = patch.object(ttrl_bfcl, "get_bfcl_decoder", return_value=parser_stub)
        self.decoder_patch.start()
        self.addCleanup(self.decoder_patch.stop)

    def session(self, **overrides):
        session = ttrl_bfcl.BFCLSession(fs_context(**overrides))
        self.addCleanup(session.close)
        return session

    def test_state_is_persistent_sequential_and_isolated_between_rollouts(self):
        first, second = self.session(), self.session()
        first.execute([fs_call("mkdir", dir_name="A"), fs_call("cd", folder="A")])
        response = first.execute([fs_call("pwd")])
        self.assertIn("A", response[0]["content"])
        self.assertIn("A", first.instances["GorillaFileSystem"].root.contents)
        self.assertNotIn("A", second.instances["GorillaFileSystem"].root.contents)
        first.finish_turn("done")
        self.assertTrue(first.outcome("completed")["valid"])
        self.assertEqual(first.outcome("completed")["num_calls"], 3)
        keys = first.namespace_keys.copy()
        first.close()
        self.assertTrue(all(key not in execute_multi_turn_func_call.__globals__ for key in keys))

    def test_state_projection_merges_equivalent_call_sequences(self):
        options = {"consensus_mode": "state", "observable_state": {"GorillaFileSystem": ["root"]}}
        first, second = self.session(**options), self.session(**options)
        first.execute([fs_call("mkdir", dir_name="A"), fs_call("mkdir", dir_name="B")])
        second.execute([fs_call("mkdir", dir_name="B"), fs_call("mkdir", dir_name="A")])
        first.finish_turn("done")
        second.finish_turn("done")
        self.assertEqual(first.outcome("completed")["key"], second.outcome("completed")["key"])

    def test_execution_errors_and_malformed_calls_can_be_recovered(self):
        missing = {"name": "missing_api", "parameters": {"type": "object", "properties": {}}}
        session = self.session(functions=FS_FUNCTIONS + [missing])
        session.execute([fs_call("missing_api")])
        session.finish_turn("unable to execute")
        self.assertFalse(session.outcome("completed")["valid"])
        session.malformed_response()
        session.execute([fs_call("pwd")])
        session.finish_turn("done")
        self.assertTrue(session.outcome("completed")["valid"])
        self.assertEqual(session.execution_errors, 1)
        self.assertEqual(session.parse_errors, 1)

    def test_clarification_and_scheduled_tools_are_observable_user_events(self):
        session = self.session(
            functions=[],
            future_user_turns=[[{"role": "user", "content": "new tools"}]],
            function_schedule={"1": [FS_FUNCTIONS[2]]},
        )
        self.assertIsNone(session.parse_calls(tool_call({}, "pwd")))
        next_turn = session.finish_turn("No suitable tool is available.")
        self.assertIn("pwd", next_turn[0]["content"])
        self.assertEqual(session.parse_calls(tool_call({}, "pwd")), [fs_call("pwd")])
        session.execute([fs_call("pwd")])
        self.assertIsNone(session.finish_turn("done"))
        self.assertEqual(len(json.loads(session.outcome("completed")["key"])["value"]), 1)

    def test_oracles_are_ignored_and_invalid_projections_clean_up(self):
        session = self.session(ground_truth=PoisonOracle(), expected_state=PoisonOracle())
        session.finish_turn("Please specify a directory.")
        self.assertFalse(session.outcome("completed")["valid"])  # No tool evidence; never vote on wording.
        namespace = execute_multi_turn_func_call.__globals__
        before = {key for key in namespace if key.startswith("ttrl_")}
        with self.assertRaises(AttributeError):
            self.session(consensus_mode="state", observable_state={"GorillaFileSystem": ["missing_attribute"]})
        self.assertEqual(before, {key for key in namespace if key.startswith("ttrl_")})
        with self.assertRaisesRegex(ValueError, "callable"):
            self.session(consensus_mode="state", observable_state={"GorillaFileSystem": ["pwd"]})
        self.assertEqual(before, {key for key in namespace if key.startswith("ttrl_")})
        with self.assertRaisesRegex(ValueError, "private"):
            self.session(observable_state={"GorillaFileSystem": ["_current_dir"]})
        with self.assertRaisesRegex(ValueError, "projection"):
            self.session(consensus_mode="state")
        with self.assertRaisesRegex(ValueError, "eight local"):
            self.session(involved_classes=["WebSearchAPI"])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.session(
                future_user_turns=[[{"role": "user", "content": "next"}]], function_schedule={"1": [FS_FUNCTIONS[2]]}
            )

    def test_calls_include_every_generation_and_preserve_duplicates(self):
        session = self.session(consensus_mode="calls")
        session.execute([fs_call("mkdir", dir_name="A")])
        session.execute([fs_call("pwd"), fs_call("pwd")])
        session.finish_turn("done")
        value = json.loads(session.outcome("completed")["key"])["value"]
        self.assertEqual(value, [fs_call("mkdir", dir_name="A"), fs_call("pwd"), fs_call("pwd")])

    def test_final_state_ignores_intermediate_user_turn_states_and_wording(self):
        options = {
            "consensus_mode": "state", "observable_state": {"GorillaFileSystem": ["root"]},
            "future_user_turns": [[{"role": "user", "content": "finish the task"}]],
        }
        first, second = self.session(**options), self.session(**options)
        first.execute([fs_call("mkdir", dir_name="A")])
        second.execute([fs_call("mkdir", dir_name="B")])
        first.finish_turn("first stage")
        second.finish_turn("different wording")
        self.assertFalse(first.outcome("completed")["valid"])  # Pending final user turn.
        first.execute([fs_call("mkdir", dir_name="B")])
        second.execute([fs_call("mkdir", dir_name="A")])
        first.finish_turn("complete")
        second.finish_turn("completed successfully")
        self.assertEqual(first.outcome("completed")["key"], second.outcome("completed")["key"])

    def test_text_only_observations_abstain_and_clarification_does_not_split_later_feedback(self):
        first, second = self.session(), self.session()
        first.finish_turn("Please provide the directory")
        second.finish_turn("Tell me the directory")
        self.assertIsNone(first.outcome("completed")["key"])
        self.assertIsNone(second.outcome("completed")["key"])
        options = {"future_user_turns": [[{"role": "user", "content": "directory A"}]]}
        first, second = self.session(**options), self.session(**options)
        first.finish_turn("Please provide the directory")
        second.finish_turn("Tell me the directory")
        for session in (first, second):
            session.execute([fs_call("pwd")])
            session.finish_turn("done")
        self.assertEqual(first.outcome("completed")["key"], second.outcome("completed")["key"])

    def test_observations_cover_all_user_turns_and_call_order_stays_significant(self):
        session = self.session(future_user_turns=[[{"role": "user", "content": "continue"}]])
        session.execute([fs_call("pwd")])
        session.finish_turn("next")
        session.execute([fs_call("mkdir", dir_name="A"), fs_call("cd", folder="A"), fs_call("pwd")])
        session.finish_turn("done")
        observations = json.loads(session.outcome("completed")["key"])["value"]
        self.assertEqual(len(observations), 4)
        self.assertNotEqual(observations[0], observations[-1])

    def test_structured_no_call_is_only_comparable_in_call_mode(self):
        calls, observations = self.session(consensus_mode="calls"), self.session()
        calls.finish_turn("[]")
        observations.finish_turn("[]")
        self.assertTrue(calls.outcome("completed")["valid"])
        self.assertFalse(observations.outcome("completed")["valid"])


class Config(dict):
    def __getattr__(self, key):
        return self[key]


class ChatTokenizer(Tokenizer):
    padding_side = "right"
    pad_token_id = 0

    def apply_chat_template(self, messages, tools=None, add_generation_prompt=False, tokenize=True):
        text = "<SYS>" + (json.dumps(tools) if tools else "")
        text += "".join(f"<{message.get('role', '')}>{message.get('content', '')}" for message in messages if message)
        if add_generation_prompt:
            text += "<assistant>"
        return [ord(char) for char in text] if tokenize else text

    def pad(self, rows, padding, max_length, return_tensors, return_attention_mask):
        ids = torch.zeros((len(rows), max_length), dtype=torch.long)
        mask = torch.zeros_like(ids)
        for index, row in enumerate(rows):
            tokens = row["input_ids"]
            if len(tokens) > max_length:
                raise ValueError("Test prompt/response exceeds its declared budget")
            start = max_length - len(tokens) if self.padding_side == "left" else 0
            ids[index, start : start + len(tokens)] = torch.tensor(tokens, dtype=torch.long)
            mask[index, start : start + len(tokens)] = 1
        return {"input_ids": ids, "attention_mask": mask}


class ScriptedServer:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    async def generate(self, request_id, prompt_ids, sampling_params):
        self.requests.append(dict(sampling_params))
        response = next(self.responses)
        if isinstance(response, BaseException):
            raise response
        return [ord(char) for char in response + "<END>"]


def loop_config(**overrides):
    multi = Config(
        enable=True,
        max_user_turns=16,
        max_assistant_turns=16,
        max_parallel_calls=8,
        max_tool_response_length=256,
        tool_response_truncate_side="middle",
        tool_config_path=None,
        format="hermes",
    )
    rollout = Config(multi_turn=multi, prompt_length=4096, response_length=4096, n=2, temperature=1.0, top_p=1.0)
    for key, value in overrides.items():
        if key in multi:
            multi[key] = value
        else:
            rollout[key] = value
    return Config(actor_rollout_ref=Config(rollout=rollout), ttrl=Config(enable=True, reward_mode="bfcl"))


@unittest.skipUnless(BFCL_RUNTIME_AVAILABLE, "Requires the existing BFCL executor package")
class TestBFCLAgentLoop(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        ToolAgentLoop._class_initialized = False
        self.runtime_patch = patch.object(
            ttrl_bfcl, "get_bfcl_runtime", return_value=(execute_multi_turn_func_call, convert_to_function_call)
        )
        self.runtime_patch.start()
        self.addCleanup(self.runtime_patch.stop)
        self.decoder_patch = patch.object(ttrl_bfcl, "get_bfcl_decoder", return_value=parser_stub)
        self.decoder_patch.start()
        self.addCleanup(self.decoder_patch.stop)

    def make_loop(self, responses, **overrides):
        server = ScriptedServer(responses)
        loop = ToolAgentLoop(loop_config(**overrides), server, ChatTokenizer())
        return loop, server

    async def test_scripted_user_messages_observations_and_group_rewards_share_one_trajectory(self):
        reply = "Continue after this clarification: " + "x" * 300 + " KEEP_FULL"
        context = fs_context(future_user_turns=[[{"role": "user", "content": reply}]])
        loop, server = self.make_loop(
            [
                "Which directory?",
                tool_call({"dir_name": "A"}, "mkdir"),
                tool_call({"folder": "A"}, "cd"),
                "done",
            ]
        )
        output = await loop.run([{"role": "user", "content": "task"}], {}, context)
        self.assertTrue(output.extra_fields["bfcl_outcome"]["valid"])
        self.assertEqual(output.extra_fields["bfcl_outcome"]["num_calls"], 2)
        text = ChatTokenizer().decode(torch.tensor(output.response_ids), False)
        self.assertIn("KEEP_FULL", text)  # User input must not be truncated as a tool response.
        self.assertEqual(len(server.requests), 4)
        self.assertLess(server.requests[-1]["max_tokens"], server.requests[0]["max_tokens"])
        action_text = ChatTokenizer().decode(
            torch.tensor(output.response_ids)[torch.tensor(output.response_mask).bool()], False
        )
        self.assertNotIn("KEEP_FULL", action_text)
        self.assertTrue(all(value in (0, 1) for value in output.response_mask))
        worker_class = getattr(getattr(AgentLoopWorker, "__ray_metadata__", None), "modified_class", AgentLoopWorker)
        worker = object.__new__(worker_class)
        worker.config, worker.tokenizer = loop.config, ChatTokenizer()
        generated = worker._postprocess([output, output])
        batch, _ = make_batches(["unused"] * 2, 2)
        batch.non_tensor_batch["extra_info"][0] = {"bfcl": context}
        ttrl_bfcl.apply_bfcl_rewards(batch, generated, 2, ChatTokenizer())
        rewards = generated.batch["ttrl_bfcl_scores"]
        self.assertEqual(rewards.sum(-1).tolist(), [1, 1])
        self.assertTrue(torch.all(rewards[generated.batch["response_mask"] == 0] == 0))
        metrics = ttrl_bfcl.compute_bfcl_metrics(generated)
        self.assertEqual(metrics["bfcl_num_calls"], 2)
        self.assertEqual(metrics["bfcl_generated_tokens"], sum(output.response_mask))
        self.assertEqual(metrics["bfcl_feedback_tokens"], len(output.response_mask) - sum(output.response_mask))

    async def test_malformed_then_repaired_calls_remain_a_valid_episode(self):
        loop, _ = self.make_loop(["<tool_call>{broken}</tool_call>", tool_call({}, "pwd"), "done"])
        output = await loop.run([{"role": "user", "content": "task"}], {}, fs_context())
        outcome = output.extra_fields["bfcl_outcome"]
        self.assertTrue(outcome["valid"])
        self.assertEqual(outcome["parse_errors"], 1)

    async def test_forced_stop_and_unrepaired_errors_never_become_positive_outcomes(self):
        cases = [
            ([tool_call({"dir_name": "A"}, "mkdir")], {"max_assistant_turns": 1}, "max_assistant_turns"),
            (["<tool_call>{broken}</tool_call>", "unable"], {}, "completed"),
            (["A" * 100], {"response_length": 20}, "response_length"),
        ]
        for responses, overrides, termination in cases:
            with self.subTest(termination=termination):
                ToolAgentLoop._class_initialized = False
                loop, _ = self.make_loop(responses, **overrides)
                output = await loop.run([{"role": "user", "content": "task"}], {}, fs_context())
                self.assertFalse(output.extra_fields["bfcl_outcome"]["valid"])
                self.assertEqual(output.extra_fields["bfcl_outcome"]["termination"], termination)

    async def test_cleanup_on_generation_exception_and_legacy_tool_path(self):
        namespace = execute_multi_turn_func_call.__globals__
        before = {key for key in namespace if key.startswith("ttrl_")}
        loop, _ = self.make_loop([RuntimeError("generation failed")])
        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            await loop.run([{"role": "user", "content": "task"}], {}, fs_context())
        self.assertEqual(before, {key for key in namespace if key.startswith("ttrl_")})
        ToolAgentLoop._class_initialized = False
        loop, _ = self.make_loop([tool_call({}, "native"), "done"])
        tool = SimpleNamespace(
            create=AsyncMock(return_value="instance"),
            execute=AsyncMock(return_value=("native observation", 0, {})),
            release=AsyncMock(),
        )
        loop.tools = {"native": tool}
        output = await loop.run([{"role": "user", "content": "task"}], {})
        tool.create.assert_awaited_once()
        tool.release.assert_awaited_once_with("instance")
        self.assertEqual(output.extra_fields, {})

    async def test_cancelled_rollout_releases_its_environment(self):
        started, pending = asyncio.Event(), asyncio.Event()
        namespace = execute_multi_turn_func_call.__globals__
        before = {key for key in namespace if key.startswith("ttrl_")}

        async def wait_for_cancel(**kwargs):
            started.set()
            await pending.wait()

        loop, server = self.make_loop([])
        server.generate = wait_for_cancel
        task = asyncio.create_task(loop.run([{"role": "user", "content": "task"}], {}, fs_context()))
        await asyncio.wait_for(started.wait(), timeout=5)
        self.assertNotEqual(before, {key for key in namespace if key.startswith("ttrl_")})
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(before, {key for key in namespace if key.startswith("ttrl_")})

    async def test_call_limit_rejects_the_whole_response_and_allows_repair(self):
        loop, _ = self.make_loop(
            [tool_call({"dir_name": "A"}, "mkdir") + tool_call({}, "pwd"), tool_call({}, "pwd"), "done"],
            max_parallel_calls=1,
        )
        output = await loop.run([{"role": "user", "content": "task"}], {}, fs_context())
        outcome = output.extra_fields["bfcl_outcome"]
        self.assertTrue(outcome["valid"])
        self.assertEqual(outcome["num_calls"], 1)  # No successful prefix from the over-limit response.
        self.assertEqual(outcome["parse_errors"], 1)

    async def test_execution_outcomes_flow_through_grpo_and_q_weights(self):
        for mode in ("calls", "observations", "state"):
            with self.subTest(mode=mode):
                context = fs_context(
                    consensus_mode=mode,
                    observable_state={"GorillaFileSystem": ["root"]} if mode == "state" else {},
                )
                outputs = []
                for directory in ("A", "A", "B"):
                    loop, _ = self.make_loop(
                        [
                            tool_call({"dir_name": directory}, "mkdir"),
                            tool_call({"folder": directory}, "cd"),
                            tool_call({}, "pwd"),
                            "done",
                        ],
                        n=3,
                    )
                    outputs.append(await loop.run([{"role": "user", "content": "task"}], {}, context))
                worker_class = getattr(
                    getattr(AgentLoopWorker, "__ray_metadata__", None), "modified_class", AgentLoopWorker
                )
                worker = object.__new__(worker_class)
                worker.config, worker.tokenizer = loop.config, ChatTokenizer()
                generated = worker._postprocess(outputs)
                batch, _ = make_batches(["unused"] * 3, 3)
                batch.non_tensor_batch["extra_info"][0] = {"bfcl": context}
                ttrl_bfcl.apply_bfcl_rewards(batch, generated, 3, worker.tokenizer)
                scores, diagnostics = compute_reward(generated, ttrl_bfcl.bfcl_reward)
                self.assertEqual(scores.sum(-1).tolist(), [1, 1, 0])
                mask, uids = generated.batch["response_mask"], np.array(["task"] * 3)
                advantages, returns = compute_grpo_outcome_advantage(scores, mask, uids)
                original, original_returns = advantages.clone(), returns.clone()
                generated.batch.update(advantages=advantages, returns=returns, token_level_scores=scores)
                generated.non_tensor_batch.update(uid=uids, majority_ratio_list=diagnostics["bfcl_majority_ratio"])
                apply_ttrl_q_weights(generated, {"enabled": True, "mode": "u_shaped", "power": 2, "floor": 0.1})
                torch.testing.assert_close(generated.batch["advantages"], original * 0.2)
                torch.testing.assert_close(generated.batch["returns"], original_returns)
                self.assertTrue(torch.all(generated.batch["advantages"][mask == 0] == 0))
                self.assertTrue(torch.all(scores[mask == 0] == 0))
                self.assertGreater(generated.batch["advantages"][0].sum(), 0)
                self.assertLess(generated.batch["advantages"][2].sum(), 0)

    async def test_async_worker_repeats_single_turn_schemas_without_dropping_them(self):
        worker_class = getattr(getattr(AgentLoopWorker, "__ray_metadata__", None), "modified_class", AgentLoopWorker)
        worker = object.__new__(worker_class)
        worker.config, worker.tokenizer = loop_config(enable=False), ChatTokenizer()
        worker.server_manager = ScriptedServer(["[]"] * 4)
        inputs = Batch(
            2,
            metadata={
                "raw_prompt": np.array([[{"role": "user", "content": "task"}]] * 2, dtype=object),
                "agent_name": np.array(["single_turn_agent"] * 2, dtype=object),
                "bfcl_context": np.array(
                    [json.dumps({"functions": [FS_FUNCTIONS[0]]}), json.dumps({"functions": [FS_FUNCTIONS[2]]})],
                    dtype=object,
                ),
            },
        )
        inputs.meta_info = {}
        generated = await worker.generate_sequences(inputs)
        self.assertEqual(len(generated), 4)
        self.assertEqual(len(worker.server_manager.requests), 4)
        for row, expected in enumerate(("mkdir", "mkdir", "pwd", "pwd")):
            tokens = generated.batch["prompts"][row]
            mask = generated.batch["attention_mask"][row, : len(tokens)].bool()
            self.assertIn(expected, worker.tokenizer.decode(tokens[mask]))
        self.assertNotIn("bfcl_outcome", generated.non_tensor_batch)  # Single-turn calls use the existing parser.


@unittest.skipUnless(BFCL_RUNTIME_AVAILABLE, "Requires BFCL's existing data/schema helpers")
class TestBFCLInputPreparation(unittest.TestCase):
    def test_single_turn_conversion_exposes_only_input_schema_and_messages(self):
        original = {
            "id": "simple_python_0",
            "question": [{"role": "user", "content": "weather task"}],
            "function": [
                {
                    "name": "weather.get",
                    "description": "weather",
                    "parameters": {
                        "type": "dict",
                        "properties": {"days": {"type": "list", "items": {"type": "float"}}},
                        "required": ["days"],
                    },
                }
            ],
            "possible_answer": PoisonOracle(),
            "ground_truth": PoisonOracle(),
        }
        row = make_bfcl_map_fn("train")(original, 0)
        context = json.loads(row["extra_info"]["bfcl"])
        function = context["functions"][0].get("function", context["functions"][0])
        self.assertEqual(function["name"], "weather_get")
        self.assertEqual(function["parameters"]["type"], "object")
        self.assertEqual(function["parameters"]["properties"]["days"]["type"], "array")
        self.assertEqual(function["parameters"]["properties"]["days"]["items"]["type"], "number")
        self.assertEqual(bfcl_chat_template_kwargs(row)["tools"][0]["function"], function)
        self.assertEqual(row["prompt"], original["question"])
        self.assertNotIn("reward_model", row)
        self.assertNotIn("ground_truth", context)
        self.assertNotIn("possible_answer", context)
        self.assertEqual(original["function"][0]["name"], "weather.get")  # Original input was not mutated.
        self.assertEqual(bfcl_chat_template_kwargs({"extra_info": {}}), {})

    def test_original_function_population_and_delayed_release_are_preserved(self):
        from bfcl_eval import utils as bfcl_utils

        docs = [{**deepcopy(function), "description": function["name"]} for function in FS_FUNCTIONS]
        original = {
            "id": "multi_turn_miss_func_0",
            "question": [[{"role": "user", "content": "task"}], []],
            "involved_classes": ["GorillaFileSystem"],
            "initial_config": {},
            "missed_function": {"1": ["pwd"]},
            "ground_truth": PoisonOracle(),
        }
        with patch.object(bfcl_utils, "load_file", side_effect=lambda _: deepcopy(docs)):
            row = make_bfcl_map_fn("train")(original, 0)
        context = json.loads(row["extra_info"]["bfcl"])
        self.assertEqual(row["agent_name"], "tool_agent")
        self.assertNotIn("pwd", [tool.get("function", tool)["name"] for tool in context["functions"]])
        released = context["function_schedule"]["1"][0]
        self.assertEqual(released.get("function", released)["name"], "pwd")
        self.assertEqual(context["future_user_turns"][0][0]["role"], "user")
        self.assertTrue(context["future_user_turns"][0][0]["content"])
        self.assertNotIn("outcome_mode", context)  # Training config chooses the common consensus implementation.
        self.assertEqual(original["missed_function"], {"1": ["pwd"]})

    def test_heterogeneous_schemas_round_trip_as_json_without_arrow_struct_nulls(self):
        contexts = [fs_context(functions=[FS_FUNCTIONS[0]]), fs_context(functions=[FS_FUNCTIONS[2]])]
        rows = [{"extra_info": {"bfcl": json.dumps(context)}} for context in contexts]
        for row, context in zip(rows, contexts):
            self.assertEqual(ttrl_bfcl.decode_bfcl_context(row["extra_info"]["bfcl"]), context)
            self.assertEqual(bfcl_chat_template_kwargs(row)["tools"][0]["function"], context["functions"][0])


class TestTTRLQWeights(unittest.TestCase):
    def make_batch(self, qs=(0.25, 0.5, 1.0)):
        scores = torch.tensor([[1.0] * int(q * 4) + [0.0] * (4 - int(q * 4)) for q in qs]).flatten()
        rewards = scores.unsqueeze(-1).repeat(1, 2) / 2
        mask = torch.ones_like(rewards)
        uids = np.repeat(np.arange(len(qs)), 4)
        advantages, returns = compute_grpo_outcome_advantage(rewards, mask, uids)
        return Batch(
            len(uids),
            tensors={
                "advantages": advantages,
                "returns": returns,
                "token_level_scores": rewards,
                "response_mask": mask,
            },
            metadata={
                "uid": uids,
                "majority_ratio_list": np.repeat(qs, 4),
                "bfcl_valid": np.ones(len(uids), dtype=bool),
            },
        )

    def test_disabled_and_uniform_weights_preserve_original_grpo(self):
        batch = self.make_batch()
        original = batch.batch["advantages"].clone()
        self.assertEqual(apply_ttrl_q_weights(batch, {"enabled": False}), {})
        self.assertNotIn("ttrl_q_weight", batch.batch)
        apply_ttrl_q_weights(batch, {"enabled": True, "mode": "uniform"})
        torch.testing.assert_close(batch.batch["advantages"], original)

    def test_weights_survive_group_normalization_without_changing_rewards_or_returns(self):
        batch = self.make_batch()
        before = {key: value.clone() for key, value in batch.batch.items()}
        metrics = apply_ttrl_q_weights(batch, {"enabled": True, "mode": "linear", "floor": 0, "power": 1})
        weights = torch.tensor(np.repeat([0.25, 0.5, 1], 4), dtype=torch.float32)
        torch.testing.assert_close(batch.batch["advantages"], before["advantages"] * weights.unsqueeze(-1))
        torch.testing.assert_close(batch.batch["token_level_scores"], before["token_level_scores"])
        torch.testing.assert_close(batch.batch["returns"], before["returns"])
        self.assertLess(metrics["q_adv_abs_after"], metrics["q_adv_abs_before"])
        self.assertEqual(metrics["q_zero_variance_groups"], 1 / 3)
        with self.assertRaisesRegex(ValueError, "exactly once"):
            apply_ttrl_q_weights(batch, {"enabled": True})

    def test_u_shaped_weight_and_degenerate_group_cannot_create_advantages(self):
        batch = self.make_batch((0, 0.25, 0.5, 0.75, 1))
        apply_ttrl_q_weights(batch, {"enabled": True, "mode": "u_shaped", "floor": 0.1, "power": 2})
        expected = torch.tensor(np.repeat([1, 0.325, 0.1, 0.325, 1], 4), dtype=torch.float32)
        torch.testing.assert_close(batch.batch["ttrl_q_weight"], expected)
        self.assertTrue(torch.all(batch.batch["advantages"][:4] == 0))
        self.assertTrue(torch.all(batch.batch["advantages"][-4:] == 0))

    def test_uid_grouping_handles_reordering_and_valid_rate_and_threshold(self):
        batch = self.make_batch((0.25, 0.75))
        order = np.array([0, 4, 1, 5, 2, 6, 3, 7])
        for key in batch.batch:
            batch.batch[key] = batch.batch[key][order]
        for key in batch.non_tensor_batch:
            batch.non_tensor_batch[key] = batch.non_tensor_batch[key][order]
        batch.non_tensor_batch["bfcl_valid"][1] = False
        metrics = apply_ttrl_q_weights(
            batch, {"enabled": True, "mode": "uniform", "min_q": 0.5, "multiply_valid_rate": True}
        )
        torch.testing.assert_close(batch.batch["ttrl_q_weight"], torch.tensor([0, 0.75] * 4))
        self.assertEqual(metrics["q_active_groups"], 0.5)

    def test_invalid_configuration_or_inconsistent_q_fails(self):
        for invalid in ({"mode": "bad"}, {"power": 0}, {"floor": -1}, {"min_q": float("nan")}, {"power": True}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_q_weight_config({"enabled": True, **invalid})
        for invalid in (-0.1, float("nan"), 0.6):
            batch = self.make_batch()
            batch.non_tensor_batch["majority_ratio_list"][1] = invalid
            with self.subTest(q=invalid), self.assertRaises(ValueError):
                apply_ttrl_q_weights(batch, {"enabled": True})


if __name__ == "__main__":
    unittest.main()
