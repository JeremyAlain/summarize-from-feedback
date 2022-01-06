import json
import os
from dataclasses import dataclass, field

import blobfile as bf
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
    input_path_folder: str = None
    input_path_index: int = None
    output_folder: str = None
    fp16_activations: bool = True
    output_key: str = "predicted_reward"
    number_of_iterations: int = 5


all_input_paths = ["samples.davinci_summary_iterative_refinement_results.jsonl", "samples.davinci_summary_refinement_summary_iterative_refinement_results.jsonl"]

def main(H: HParams):
    H.input_path = os.path.join(H.input_path_folder, all_input_paths[H.input_path_index])
    assert os.path.isfile(H.input_path), H.input_path
    layout = H.reward_model_spec.run_params.all_gpu_layout()

    reward_model = RewardModel(task_hparams=H.task, spec=H.reward_model_spec, layout=layout)

    setup_logging_with_pacific_tz()

    act_dtype = torch.float16 if H.fp16_activations else torch.float32

    results_dir = H.output_folder
    bf.makedirs(results_dir)


    # Creates files for printing. Only the replica root prints the files

    experiment_name = os.path.split(H.input_path)[1].split(".")[1] + ".jsonl"
    if not os.path.isdir(H.output_folder):
        os.mkdir(H.output_folder)
    output_file_name = os.path.join(H.output_folder, experiment_name)
    print(f"Outputs will be written to {output_file_name}")

    if layout.is_logging_rank:
        with open(os.path.join(results_dir, experiment_name +"task_hparams.json"), "w") as f:
            json.dump(H.task.to_json(), f)
        with open(os.path.join(results_dir, experiment_name +"hparams.json"), "w") as f:
            json.dump(H.to_json(), f)

    input_iter = make_jsonl_samples_iter(H.input_path, layout=layout)

    with open(output_file_name, "a") as out_f:
        input_idx = 0
        for input in input_iter:
            with Timer() as timer:
                query_tokens = torch.tensor(input["context_tokens"])
                assert_shape_eq(
                    query_tokens, (H.task.query.length,), "Context tokens shape mismatch"
                )

                original_summary_tokens = torch.tensor(input["original_summary_tokens"])

                original_summary_results = reward_model.reward(
                    query_tokens=query_tokens.unsqueeze(0),
                    response_tokens=original_summary_tokens.unsqueeze(0),
                    act_dtype=act_dtype,
                )
                original_summary_reward = to_numpy(original_summary_results["reward"])

                target_tokens = torch.tensor(input["target_tokens"])
                target_results = reward_model.reward(
                    query_tokens=query_tokens.unsqueeze(0),
                    response_tokens=target_tokens.unsqueeze(0),
                    act_dtype=act_dtype,
                )
                target_reward = to_numpy(target_results["reward"])

                if layout.is_replica_root:
                    output = {**input, "original_summary_reward": original_summary_reward,
                              "target_reward": target_reward}

                for iteration in range(H.number_of_iterations):
                    # response_tokens = torch.tensor(input["iteration_{}_sample_tokens".format(iteration)])
                    # assert_eq(response_tokens.dim(), 2)
                    # n_responses = response_tokens.size(0)
                    # results = reward_model.reward(
                    #     query_tokens=query_tokens.unsqueeze(0),
                    #     response_tokens=response_tokens.unsqueeze(0),
                    #     act_dtype=act_dtype,
                    # )
                    # rewards = to_numpy(results["reward"].reshape((n_responses,)))

                    chosen_refinement_tokens = torch.unsqueeze(torch.tensor(input["iteration_{}_chosen_refinement_tokens".format(iteration)]),dim=0)
                    assert_eq(chosen_refinement_tokens.dim(), 2)
                    n_responses = 1
                    results = reward_model.reward(
                        query_tokens=query_tokens.unsqueeze(0),
                        response_tokens=chosen_refinement_tokens.unsqueeze(0),
                        act_dtype=act_dtype,
                    )
                    chosen_refinement_reward = to_numpy(results["reward"].reshape((n_responses,)))

                    if layout.is_replica_root:
                        output["iteration_{}_chosen_refinement_reward".format(iteration)] = chosen_refinement_reward
                        #output["iteration_{}".format(iteration) + H.output_key] = rewards

                if layout.is_replica_root:
                    out_f.write((json.dumps(jsonl_encoding.encode_example(output)) + "\n"))

            input_idx += 1
            if layout.is_replica_root:
                print(f"Batch {input_idx}.  Took {timer.interval} seconds")

        if layout.is_replica_root:
            print(f"Wrote {input_idx} batches to {output_file_name}")
    return dict(output_path=results_dir)
