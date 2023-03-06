"""
    First we're going to just follow the thing and play with the metrics once done

"""
import json
import warnings
from pathlib import Path

from langchain import HuggingFaceHub, PromptTemplate, FewShotPromptTemplate, LLMChain
from langchain.prompts.example_selector import LengthBasedExampleSelector
from setfit.trainer import set_seed
import click
import random
import numpy.random
import torch
from datasets import load_dataset, DatasetDict
import numpy as np
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from os import devnull
from sentence_transformers.losses import CosineSimilarityLoss
import math
import os
from mytorch.utils.goodies import FancyDict
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoConfig

from overrides import CustomTrainer, CustomModel

random.seed(42)
torch.manual_seed(42)
numpy.random.seed(42)


@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, "w") as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


def case0(
    dataset: DatasetDict,
    seed: int,
    num_sents: int,
    num_epochs: int,
    num_epochs_finetune: int,
    batch_size: int,
    test_on_test: bool = False,
) -> dict:
    """
    Do exactly what the blogpost does

    Get SetFit model (with ST and LogClf)
    # Step 1:
    Create pairs
    Fine tune ST on faux task (cosine thing)
    Fit Log reg on main task

    # Step 2:
    Run model to classify on main task
    Report Accuracy
    """
    model = CustomModel.from_pretrained(
        "sentence-transformers/paraphrase-mpnet-base-v2"
    )

    # Sample num_sents from the dataset. Divide them in 80/20
    train_ds = dataset["train"].shuffle(seed=seed).select(range(num_sents))
    if test_on_test:
        test_ds = dataset["test"]
    else:
        train_ds, test_ds = train_ds.select(
            range(int(len(train_ds) * 0.8))
        ), train_ds.select(range(int(len(train_ds) * 0.8), len(train_ds)))

    trainer = CustomTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        loss_class=CosineSimilarityLoss,
        batch_size=batch_size,
        num_iterations=20,  # Number of text S to generate for contrastive learning
        num_epochs=num_epochs,  # Number of epochs to use for contrastive learning
    )

    # Fit the ST on the cosinesim task; Fit the entire thing on the main task
    trainer.train(num_epochs=num_epochs, num_epochs_finetune=num_epochs_finetune)

    metrics = trainer.evaluate()
    # print(metrics)
    return metrics


def case1(
    dataset: DatasetDict,
    seed: int,
    num_sents: int,
    num_epochs: int,
    num_epochs_finetune: int,
    batch_size: int,
    test_on_test: bool = False,
) -> dict:
    """
    This is regular fine-tuning. Noisy.
    Skip ST Finetuning; Slap a classifier and train the thing together.

    Get SetFit model (with ST and DenseHead).
    # Step 1
    Create pairs
    DONT Fine tune ST on faux task (Cosine)
    Fit DenseHead + ST on main task

    # Step 2
    Run model to classify on main task
    Report Accuracy
    """
    model = CustomModel.from_pretrained(
        "sentence-transformers/paraphrase-mpnet-base-v2", use_differentiable_head=True
    )

    # Sample num_sents from the dataset. Divide them in 80/20
    train_ds = dataset["train"].shuffle(seed=seed).select(range(num_sents))
    if test_on_test:
        test_ds = dataset["test"]
    else:
        train_ds, test_ds = train_ds.select(
            range(int(len(train_ds) * 0.8))
        ), train_ds.select(range(int(len(train_ds) * 0.8), len(train_ds)))

    trainer = CustomTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        loss_class=CosineSimilarityLoss,
        batch_size=batch_size,
        num_iterations=20,  # Number of text pairs to generate for contrastive learning
        num_epochs=num_epochs,  # Number of epochs to use for contrastive learning
    )
    # trainer.se

    # Fit the ST on the cosinesim task; Fit the entire thing on the main task
    # trainer.train(num_epochs=num_epochs, num_epochs_finetune=num_epochs_finetune) #, do_fitclf_trainencoder=False)
    trainer.train(
        num_epochs=num_epochs,
        num_epochs_finetune=num_epochs_finetune,
        do_fitclf_trainencoder=True,
        do_finetune=False,
    )

    metrics = trainer.evaluate()
    # print(metrics)
    return metrics


def case2(
    dataset: DatasetDict,
    seed: int,
    num_sents: int,
    num_epochs: int,
    num_epochs_finetune: int,
    batch_size: int,
    test_on_test: bool = False,
):
    """
    Get SetFit model (ST + LogClf Head)

    # Step 1
    Do not fine-tune ST on faux task (Cosine)
    Just fit LogClf on the main task (freeze body)

    # Step 2
    Run model to classify on main task
    Report Accuracy
    """
    model = CustomModel.from_pretrained(
        "sentence-transformers/paraphrase-mpnet-base-v2", use_differentiable_head=False
    )

    # Sample num_sents from the dataset. Divide them in 80/20
    train_ds = dataset["train"].shuffle(seed=seed).select(range(num_sents))
    if test_on_test:
        test_ds = dataset["test"]
    else:
        train_ds, test_ds = train_ds.select(
            range(int(len(train_ds) * 0.8))
        ), train_ds.select(range(int(len(train_ds) * 0.8), len(train_ds)))

    # Freeze the head (so we never train/finetune ST)
    trainer = CustomTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        loss_class=CosineSimilarityLoss,
        batch_size=batch_size,
        num_iterations=20,  # Number of text pairs to generate for contrastive learning
        num_epochs=num_epochs,  # Number of epochs to use for contrastive learning
    )

    trainer.train(
        num_epochs=num_epochs,
        num_epochs_finetune=num_epochs_finetune,
        do_finetune=False,
    )

    metrics = trainer.evaluate()
    # print(metrics)
    return metrics


def case3(
    dataset: DatasetDict,
    seed: int,
    num_sents: int,
    batch_size: int,
    test_on_test: bool = False,
    *args, **kwargs
):
    """
    Uses langchain to throw questions to HF model Flan t5 xl.
    """
    # First figure out the length of the model
    config = AutoConfig.from_pretrained("google/flan-t5-xl")
    max_len = config.n_positions

    # Read HuggingFace API key
    try:
        with (Path(".") / "hf_token.key").open("r") as f:
            hf_token_key = f.read().strip()
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token_key
    except FileNotFoundError:
        raise FileNotFoundError(
            f"No HuggingFace API key found at {(Path('.') / 'hf_token.key').absolute()}"
            f"You need to generate yours at https://huggingface.co/settings/tokens"
            f"and paste it in this file."
        )

    # # initialize Hub LLM
    hub_llm = HuggingFaceHub(
        repo_id="google/flan-t5-xl", model_kwargs={"temperature": 1e-10}
    )

    label_to_id = {"negative": 0, "positive": 1}
    id_to_label = {v: k for k, v in label_to_id.items()}

    # Go through the dataset, generate train and testset
    # Sample num_sents from the dataset. Divide them in 80/20
    train_ds = dataset["train"].shuffle(seed=seed).select(range(num_sents))
    if test_on_test:
        test_ds = dataset["test"]
    else:
        train_ds, test_ds = train_ds.select(
            range(int(len(train_ds) * 0.8))
        ), train_ds.select(range(int(len(train_ds) * 0.8), len(train_ds)))

    """ Prompt stuff """
    # create a example template
    example_template = """
    Review: {query}
    Sentiment: {answer}
    """
    # create a prompt example from above template
    example_prompt = PromptTemplate(
        input_variables=["query", "answer"], template=example_template
    )
    examples = [
        {"query": x["text"], "answer": id_to_label[x["label"]]} for x in train_ds
    ]
    prefix = """Classify into positive or negative. Here are some examples: """
    suffix = """
    Review: {query}
    Sentiment: 
    """
    # We'll use the `LengthBasedExampleSelector` to select the examples.
    example_selector = LengthBasedExampleSelector(
        # These are the examples is has available to choose from.
        examples=examples,
        # This is the PromptTemplate being used to format the examples.
        example_prompt=example_prompt,
        # This is the maximum length that the formatted examples should be.
        # Length is measured by the get_text_length function below.
        max_length=max_len,
    )

    # now create the few shot prompt template
    few_shot_prompt_template = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix=prefix,
        suffix=suffix,
        input_variables=["query"],
        example_separator="\n\n",
    )
    llm_chain = LLMChain(prompt=few_shot_prompt_template, llm=hub_llm)

    score = []
    for batch in tqdm(DataLoader(test_ds, batch_size=batch_size)):
        texts = [{"query": instance} for instance in batch["text"]]
        answers = llm_chain.generate(texts)

        # Iterate through answers and labels
        for i, generation in enumerate(answers.generations):
            answer = generation[0].text.strip().lower()
            try:
                if batch["label"][i] == label_to_id[answer]:
                    score.append(1)
                else:
                    score.append(0)
            except KeyError:
                warnings.warn(
                    f"The answer to {i}th element is `{answer}`. Marking this as a wrong instance."
                )
                score.append(0)

    return {"accuracy": np.mean(score)}


def merge_metrics(list_of_metrics):
    pooled = {}
    for metrics in list_of_metrics:
        for k, v in metrics.items():
            pooled.setdefault(k, []).append(v)

    return pooled


def normalize_dataset(dataset: DatasetDict):
    """Check if text and label exist or not. Further if label_text doesn't exist makes 0 as neg 1 as pos"""


@click.command()
@click.option(
    "--dataset-name",
    "-d",
    type=str,
    default="SetFit/SentEval-CR",
    help="The name of the dataset eg SetFit/SentEval-CR | imdb | ...",
)
@click.option(
    "--case",
    "-c",
    type=int,
    required=True,
    help="Which case to run. See case docstrings for info. Values between 0,1, and 2. You can run 3 but its useless.",
)
@click.option(
    "--repeat",
    "-r",
    type=int,
    default=1,
    help="The number of times we should run the entire experiment (changing the seed).",
)
@click.option(
    "--batch-size", "-bs", type=int, default=16, help="... you know what it is."
)
@click.option("--num-sents", "-ns", type=int, default=64, help="Size of our train set.")
@click.option(
    "--num-epochs",
    "-e",
    type=int,
    default=1,
    help="Epochs for fitting Clf+ST on the classification task.",
)
@click.option(
    "--num-epochs-finetune",
    "-eft",
    type=int,
    default=1,
    help="Epochs for both fine-tuning ST on the cosinesimilarity task.",
)
@click.option(
    "--test-on-test",
    "-tot",
    is_flag=True,
    default=False,
    help="If true, we report metrics on testset.",
)
def run(
    dataset_name: str,
    repeat: int,
    batch_size: int,
    num_epochs: int,
    num_epochs_finetune: int,
    num_sents: int,
    case: int,
    test_on_test: bool,
):
    try:
        fname = globals()[f"case{case}"]
    except KeyError:
        raise ValueError(f"No function called case{case}")

    if repeat < 1:
        raise ValueError("Repeats must be greater than 0")

    config = FancyDict(
        **{
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "num_epochs_finetune": num_epochs_finetune,
            "num_sents": num_sents,
            "test_on_test": test_on_test,
        }
    )

    if case == 3:
        repeat = 1
        warnings.warn(f"On case 3 i.e. prompting LLMs, we do not repeat to respect the rate limits.")

    metrics = []
    for _ in range(repeat):
        seed = random.randint(0, 200)
        set_seed(seed)
        with suppress_stdout_stderr():
            # suppress prints, allow exceptions
            dataset = load_dataset(dataset_name)
        metric = fname(dataset, seed=seed, **config)
        metrics.append(metric)

    print(f"---------- FINALLY over {repeat} runs -----------")
    metrics = merge_metrics(metrics)
    print({k: f"{np.mean(v):.3f} +- {np.std(v):.3f}" for k, v in metrics.items()})
    metrics["config"] = config

    # Dump them to disk
    dumpdir = Path(f"summaries") / f"{dataset_name.split('/')[-1]}_{num_sents}"
    dumpdir.mkdir(parents=True, exist_ok=True)
    with (dumpdir / f"case_{case}.json").open("w+") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    run()
    # config = FancyDict(**{"test_on_test": True, "num_epochs": 1, "batch_size": 16})
    # test_on_test = True
    #
    # print("------------")
    # print("---Case 0---")
    # # dataset = load_dataset("SetFit/SentEval-CR")
    # # case0(dataset, 42, 50, **config)
    # print("------------")
    # print("---Case 1---")
    # dataset = load_dataset()
    # case1(dataset, 42, 50, **config)
    # print("------------")
    # print("---Case 1---")
    # # dataset = load_dataset("SetFit/SentEval-CR")
    # # case2(dataset, 42, 50, **config)
