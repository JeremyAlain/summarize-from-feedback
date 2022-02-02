import json
import os
from dataclasses import dataclass, field

import blobfile as bf
import numpy as np
import torch

from summarize_from_feedback.datasets import jsonl_encoding
from summarize_from_feedback.query_response_model import ModelSpec
from summarize_from_feedback.reward_model import RewardModel
from summarize_from_feedback.task_data import make_jsonl_samples_iter
from summarize_from_feedback.tasks import TaskHParams
from summarize_from_feedback.utils import Timer, hyperparams
from summarize_from_feedback.utils.assertions import assert_shape_eq, assert_eq
from summarize_from_feedback.utils.logging_utils import setup_logging_with_pacific_tz
from summarize_from_feedback.utils.torch_utils import to_numpy

"""
Evaluates a reward model on a set of query-responses examples. The output will contain the same
json data as the input along with an extra key containing the predicted reward.
"""


@dataclass
class HParams(hyperparams.HParams):
    reward_model_spec: ModelSpec = field(default_factory=ModelSpec)
    task: TaskHParams = field(default_factory=TaskHParams)
    results_file_path: str = None
    output_folder: str = None
    fp16_activations: bool = True
    output_key: str = "predicted_reward"

def main(H: HParams):
    assert os.path.isfile(H.results_file_path), H.results_file_path
    layout = H.reward_model_spec.run_params.all_gpu_layout()
    setup_logging_with_pacific_tz()
    act_dtype = torch.float16 if H.fp16_activations else torch.float32
    results_dir = H.output_folder
    bf.makedirs(results_dir)

    experiment_name = os.path.split(H.results_file_path)[1].split(".")[1] + ".jsonl"
    if not os.path.isdir(H.output_folder):
        os.mkdir(H.output_folder)
    output_file_name = os.path.join(H.output_folder, experiment_name)
    print(f"Outputs will be written to {output_file_name}")

    if layout.is_logging_rank:
        with open(os.path.join(results_dir, experiment_name +"task_hparams.json"), "w") as f:
            json.dump(H.task.to_json(), f)
        with open(os.path.join(results_dir, experiment_name +"hparams.json"), "w") as f:
            json.dump(H.to_json(), f)

    input_iter = make_jsonl_samples_iter(H.results_file_path, layout=layout)

    replica_rewards = []
    replica_original_summary_rewards = []
    replica_target_rewards = []

    reward_model = RewardModel(task_hparams=H.task, spec=H.reward_model_spec, layout=layout)
    with open(output_file_name, "a") as out_f:
        input_idx = 0
        for input in input_iter:
            with Timer() as timer:
                query_tokens = torch.tensor(input["context_tokens"])
                assert_shape_eq(
                    query_tokens, (H.task.query.length,), "Context tokens shape mismatch"
                )
                response_tokens = torch.tensor(input["sample_tokens"])
                assert_eq(response_tokens.dim(), 2)

                n_responses = response_tokens.size(0)

                results = reward_model.reward(
                    query_tokens=query_tokens.unsqueeze(0),
                    response_tokens=response_tokens.unsqueeze(0),
                    act_dtype=act_dtype,
                )
                rewards = to_numpy(results["reward"].reshape((n_responses,)))

                if "human_preference_policy" in H.results_file_path:
                    target_tokens = torch.tensor(input["target_tokens"])
                    target_results = reward_model.reward(
                        query_tokens=query_tokens.unsqueeze(0),
                        response_tokens=target_tokens.unsqueeze(0),
                        act_dtype=act_dtype,
                    )
                    target_reward = to_numpy(target_results["reward"])


                if layout.is_replica_root:
                    replica_rewards.append(rewards)
                    if "human_preference_policy" in H.results_file_path:
                        replica_target_rewards.append(target_reward)
                        output = {**input, H.output_key: rewards, "target_reward": target_reward}
                    else:
                        output = {**input, H.output_key: rewards}

                    out_f.write((json.dumps(jsonl_encoding.encode_example(output)) + "\n"))
            input_idx += 1
            if layout.is_replica_root:
                print(f"Batch {input_idx}.  Took {timer.interval} seconds")

        if layout.is_replica_root:
            print(f"Wrote {input_idx} batches to {output_file_name}")

            replica_rewards = np.stack(replica_rewards, axis=0)
            replica_target_rewards = np.stack(replica_target_rewards, axis=0)

            all_rewards = reward_model.dp_comm.mpi_all_gather(replica_rewards, "rewards")
            if "human_preference_policy" in H.results_file_path:
                all_target_rewards = reward_model.dp_comm.mpi_all_gather(replica_target_rewards, "target_rewards")

            if layout.replica_idx == 0:
                all_rewards = np.concatenate(all_rewards, axis=0)

                print(f"Mean predicted reward: {all_rewards.mean():.3f}")
                if all_rewards.shape[1] > 1:
                    print(f"Stddev within a query: {all_rewards.std(axis=1, ddof=1).mean():.3}")
                print(f"Stddev across queries: {all_rewards.std(axis=0, ddof=1).mean():.3}")

                if "human_preference_policy" in H.results_file_path:
                    all_target_rewards = np.concatenate(all_target_rewards, axis=0)
                    print("-------")
                    print(f"Mean target reward: {all_target_rewards.mean():.3f}")
                    print(f"Stddev across queries: {all_target_rewards.std(axis=0, ddof=1).mean():.3}")

    return dict(output_path=results_dir)
