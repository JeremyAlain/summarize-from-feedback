import fire
import os

from summarize_from_feedback import eval_reward_model
from summarize_from_feedback.utils.combos import combos, bind, bind_nested
from summarize_from_feedback.utils.experiments import experiment_def_launcher


def experiment_definitions():
    root_path = os.getcwd()
    if "js12882" in root_path:
        device = "cuda"
    else:
        device = "cpu"
    reward_model_spec = combos(
        bind("device", device),
        bind("load_path", "https://openaipublic.blob.core.windows.net/summarize-from-feedback/models/rm4"),
        bind("short_name", "rm4"),
    )
    tldr_task = combos(
    bind(
        "query.format_str", "SUBREDDIT: r/{subreddit}\n\nTITLE: {title}\n\nPOST: {post}\n\nTL;DR:"
    ),
    bind("query.dataset", "tldr_3_filtered"),
    bind("query.length", 512),
    bind("response.length", 48),
    bind("query.truncate_text", "\n"),
    bind("query.truncate_field", "post"),
    bind("query.pad_side", "left"),
    bind("response.truncate_token", 50256),  # endoftext
    bind("response.ref_format_str", " {reference}"),  # add a leading space
)

    reward_model_gpu = combos(
        bind_nested("task", tldr_task),
        bind("mpi", 1),
        bind_nested("reward_model_spec", reward_model_spec),
    )
    reward_model_cpu = combos(
        bind_nested("task", tldr_task),
        bind("mpi", 1),
        bind_nested("reward_model_spec", reward_model_spec),
        bind("fp16_activations", False)
    )
    return locals()

if __name__ == "__main__":
    fire.Fire(
        experiment_def_launcher(
            experiment_dict=experiment_definitions(), main_fn=eval_reward_model.main, mode="local"
        )
    )
