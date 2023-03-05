"""
    First we're going to just follow the thing and play with the metrics once done

"""
from typing import Optional

import click
import random
import numpy.random
import torch
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from datasets import load_dataset, DatasetDict
import numpy as np
from setfit.modeling import sentence_pairs_generation
from torch.utils.data import DataLoader
from sentence_transformers.losses import CosineSimilarityLoss
import math
from mytorch.utils.goodies import FancyDict
from setfit import SetFitModel

from overrides import CustomTrainer

random.seed(42)
torch.manual_seed(42)
numpy.random.seed(42)


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
    model = SetFitModel.from_pretrained(
        "sentence-transformers/paraphrase-mpnet-base-v2"
    )

    # Sample num_sents from the dataset. Divide them in 80/20
    train_ds = dataset["train"].shuffle(seed=42).select(range(num_sents))
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

    # Fit the ST on the cosinesim task; Fit the entire thing on the main task
    trainer.train(num_epochs=num_epochs, num_epochs_finetune=num_epochs_finetune)

    metrics = trainer.evaluate()
    print(metrics)
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
    Get SetFit model (with ST and DenseHead).
    # Step 1
    Create pairs
    Fine tune ST on faux task (Cosine)
    Fit DenseHead on main task

    # Step 2
    Run model to classify on main task
    Report Accuracy
    """
    model = SetFitModel.from_pretrained(
        "sentence-transformers/paraphrase-mpnet-base-v2", use_differentiable_head=True
    )

    # Sample num_sents from the dataset. Divide them in 80/20
    train_ds = dataset["train"].shuffle(seed=42).select(range(num_sents))
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

    # Fit the ST on the cosinesim task; Fit the entire thing on the main task
    trainer.train(num_epochs=num_epochs, num_epochs_finetune=num_epochs_finetune)

    metrics = trainer.evaluate()
    print(metrics)
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
    Get SetFit model (ST + Dense Head)

    # Step 1
    Do not fine-tune ST on faux task (Cosine)
    Just fit DenseHead on the main task (freeze body)

    # Step 2
    Run model to classify on main task
    Report Accuracy
    """
    model = SetFitModel.from_pretrained(
        "sentence-transformers/paraphrase-mpnet-base-v2", use_differentiable_head=True
    )

    # Sample num_sents from the dataset. Divide them in 80/20
    train_ds = dataset["train"].shuffle(seed=42).select(range(num_sents))
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


def case3(dataset: DatasetDict, seed: int, num_sents: int, test_on_test: bool = False):
    """
    Get SetFit model (ST + Dense Head)

    # Step 1
    Do not Fine-tune ST on faux task (cosine)
    Fit ST+DenseHead on the main task

    # Step 2:
    same as always
    """
    ...


def merge_metrics(list_of_metrics):
    pooled = {}
    for metrics in list_of_metrics:
        for k, v in metrics.items():
            pooled.setdefault(k, []).append(v)

    return {k: np.mean(v) for k, v in pooled.items()}


@click.command()
@click.option(
    "--dataset-name",
    "-d",
    type=str,
    default="SetFit/SentEval-CR",
    help="The name of the dataset eg SetFit/SentEval-CR",
)
@click.option(
    "--case",
    "-c",
    type=int,
    required=True,
    help="Which case to run. See case docstrings for info.",
)
@click.option(
    "--repeat",
    "-r",
    type=int,
    default=1,
    help="The number of times we should run the entire codebase",
)
@click.option(
    "--batch-size", "-bs", type=int, default=16, help="... you know what it is"
)
@click.option("--num-sents", "-ns", type=int, default=64, help="Size of our train set")
@click.option(
    "--num-epochs",
    "-e",
    type=int,
    default=1,
    help="Epochs for fitting Clf+ST",
)
@click.option(
    "--num-epochs-finetune",
    "-eft",
    type=int,
    default=1,
    help="Epochs for both fine-tuning ST",
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

    metrics = []
    for _ in range(repeat):
        dataset = load_dataset(dataset_name)
        metric = fname(dataset, seed=42, **config)
        metrics.append(metric)

    print(f"---------- FINALLY over {repeat} runs -----------")
    print(merge_metrics(metrics))


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
