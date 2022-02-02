import os
import pandas as pd
import numpy as np
import json
import click
from typing import Tuple

from summarize_from_feedback.tasks import TaskResponseHParams, TaskHParams, TaskQueryHParams
import summarize_from_feedback
from summarize_from_feedback import tasks
from summarize_from_feedback.datasets.jsonl_encoding import encode_example

results_folder_name = "end_to_end"
number_of_generated_samples = 5
context_format = "SUBREDDIT: r/{subreddit}\n\nTITLE: {title}\n\nPOST: {post}\n\nTL;DR:"


@click.command()
@click.option("--results_file_path", required=True, type=str)
def prepare_results(results_file_path):
    is_cluster, root_path, results_folder = setup()
    task_response_hparams = TaskResponseHParams()
    task_response_hparams.length = 48
    task_response_hparams.ref_format_str = " {reference}"
    task_response_hparams.truncate_token = 50256

    task_query_hparams = TaskQueryHParams()
    task_query_hparams.format_str = context_format
    task_query_hparams.length = 512
    task_query_hparams.truncate_text = "\n"
    task_query_hparams.truncate_field = "post"
    task_query_hparams.pad_side = "left"
    task_query_hparams.padding = None

    task_hparams = TaskHParams
    task_hparams.response = task_response_hparams
    task_hparams.query = task_query_hparams



    assert os.path.isfile(results_file_path), results_file_path

    assert "summary_feedback_refinement" in results_file_path or "summary_refinement" in results_file_path or "human_preference_policy" in results_file_path \
        or "summary" in results_file_path

    if "human_preference_policy" in results_file_path:
        results_df = pd.read_json(results_file_path, lines=True)
    else:
        results_df = pd.read_json(results_file_path)

    number_of_samples = results_df.shape[0]
    reformated_results_list = []
    reformated_result = {}
    for sample_id in range(number_of_samples):
        results_sample = results_df.iloc[sample_id]
        context_text = context_format.format(subreddit=results_sample["subreddit"],
                                             title=results_sample["title"],
                                             post=results_sample["post"])
        reformated_result["context"] = context_text
        samples = []
        if "human_preference_policy" in results_file_path:
            samples.append(results_sample["text"])
        elif "summary_feedback_refinement" in results_file_path or "summary_refinement" in results_file_path:
            for i in range(number_of_generated_samples):
                samples.append(results_sample["refinement_{}".format(i)])
        elif "summary" in results_file_path:
            for i in range(number_of_generated_samples):
                samples.append(results_sample["summary"][i])
        else:
            print("Results file:", results_file_path)
            raise NotImplementedError()

        reformated_result["samples"] = samples
        extra_fields = {}
        extra_fields["id"] = results_sample["id"]
        extra_fields["subreddit"] = results_sample["subreddit"]
        extra_fields["title"] = results_sample["title"]
        extra_fields["post"] = results_sample["post"]

        reformated_result["extra_fields"] = extra_fields

        response_encoder = tasks.ResponseEncoder(task_response_hparams, summarize_from_feedback.encoder)
        query_data_fields = {"subreddit": results_sample["subreddit"],
                             "title": results_sample["title"],
                             "post": results_sample["post"]}
        query_info = tasks.process_query(query_data_fields, encoder=summarize_from_feedback.encoder,
                                         hparams=task_hparams.query)

        all_sample_tokens = []
        if "human_preference_policy" in results_file_path:
            sample_tokens = response_encoder.encode_response(reformated_result["samples"][0], allow_truncate=True)
            all_sample_tokens.append(sample_tokens)

            reformated_result["target"] = results_sample["target"]
            target_tokens = response_encoder.encode_response(reformated_result["target"], allow_truncate=True)
            reformated_result["target_tokens"] = np.array([target_tokens])

        else:
            for i in range(number_of_generated_samples):
                sample_tokens = response_encoder.encode_response(reformated_result["samples"][i], allow_truncate=True)
                all_sample_tokens.append(sample_tokens)

        reformated_result["context_tokens"] = np.array(query_info["tokens"])
        reformated_result["sample_tokens"] = np.array(all_sample_tokens)

        reformated_results_list.append(encode_example(reformated_result))

    reformated_results_path = os.path.join(results_folder, "reformated_results")
    if not os.path.isdir(reformated_results_path):
        os.mkdir(reformated_results_path)
    with open(os.path.join(reformated_results_path,"{}_reformated.jsonl".format(os.path.split(results_file_path)[1].split(".")[0])), "w") as outfile:
        for entry in reformated_results_list:
            json.dump(entry, outfile)
            outfile.write('\n')

def setup() -> Tuple[bool, str, str]:
    root_path = os.getcwd()
    if "js12882" in root_path:
        return True, root_path, "home/js12882/" + results_folder_name
    else:
        return False, root_path, "/Users/jeremyscheurer/Code/language_alignment/language_feedback_learning/data/results/" + results_folder_name

if __name__ == '__main__':
    prepare_results()