"""
    Here we be bussin about that active learning thingamajing
"""
import gc
import json
from pathlib import Path

import click
import datasets
import numpy as np
import torch
from tqdm.auto import trange
from sklearn.metrics import accuracy_score
from small_text import (
    PoolBasedActiveLearner,
    random_initialization_balanced,
    BreakingTies,
    ContrastiveActiveLearning,
    SubsamplingQueryStrategy,
)
from small_text import TextDataset
from small_text.integrations.transformers.classifiers.factories import SetFitClassificationFactory
from small_text.integrations.transformers.classifiers.setfit import SetFitModelArguments


# TODO: click stuff comes here
@click.command()
@click.option(
    "--dataset-name",
    "-d",
    type=str,
    default="imdb",
    help=(
        "The name of the dataset as it appears on the HuggingFace hub "
        "e.g. SetFit/SentEval-CR | SetFit/bbc-news | SetFit/enron_spam | imdb ... "
    ),
)
@click.option(
    "--num-sents",
    "-ns",
    type=int,
    default=100,
    help="Size of our train set. I.e., the dataset at the END of AL. Not the start of it.",
)
@click.option(
    "--num-queries",
    "-nq",
    type=int,
    default=10,
    help="Number of times we query the unlabeled set and pick some labeled examples. Set short values (under 10)",
)
@click.option(
    "--full-test",
    "-ft",
    is_flag=True,
    default=False,
    help=(
        "We truncate the testset of every dataset to have 100 instances. "
        "If you know what you're doing, you can test on the full dataset."
        "NOTE that if you're running this in case 3 you should probably be a premium member and not be paying per use."
    ),
)
def run(dataset_name: str, num_sents: int, num_queries: int, full_test: bool):
    dataset = datasets.load_dataset(dataset_name)
    if "text" not in dataset["train"].column_names and "label" not in dataset["train"].column_names:
        raise ValueError(
            f"The dataset {dataset_name} does not have either 'text' or 'label' field. "
            f"The fields instead are: {dataset['train'].column_names}"
        )

    # Trim the test set to 100 unless specified otherwise
    if (len(dataset["test"]) > 100) and not full_test:
        dataset["test"] = dataset["test"].shuffle(42).select(range(100))

    num_classes = np.unique(dataset["train"]["label"]).shape[0]
    target_labels = np.arange(num_classes)

    # Put dataset into small-text specific classes because ... reasons.
    train = TextDataset.from_arrays(
        dataset["train"]["text"], np.array(dataset["train"]["label"]), target_labels=target_labels
    )
    test = TextDataset.from_arrays(
        dataset["test"]["text"], np.array(dataset["test"]["label"]), target_labels=target_labels
    )

    # Load the model in small-text specific manner (which is understandable but i dont like it :P)
    sentence_transformer_model_name = "sentence-transformers/paraphrase-mpnet-base-v2"
    setfit_model_args = SetFitModelArguments(sentence_transformer_model_name)
    # @pri: this can generate SetFitClassification models which has both setfitmodel and setfittrainer inside
    # we can replace this by a custom factory and import our overrides stuff but we don't need to for now.
    clf_factory = SetFitClassificationFactory(setfit_model_args, num_classes)

    # define a query strategy and initialize a pool-based active learner
    query_strategy = SubsamplingQueryStrategy(ContrastiveActiveLearning(batch_size=20, k=5))
    # suppress progress bars in jupyter notebook
    setfit_train_kwargs = {"show_progress_bar": False}
    active_learner = PoolBasedActiveLearner(
        clf_factory, query_strategy, train, fit_kwargs={"setfit_train_kwargs": setfit_train_kwargs}
    )

    # Do a warm start.
    def initialize_active_learner(active_learner, y_train):
        x_indices_initial = random_initialization_balanced(y_train, n_samples=num_sents // num_queries)
        y_initial = y_train[x_indices_initial]
        active_learner.initialize_data(x_indices_initial, y_initial)
        return x_indices_initial

    initial_indices = initialize_active_learner(active_learner, train.y)
    labeled_indices = initial_indices

    # How many more iterations
    num_queries -= 1

    def evaluate(active_learner, train, test):
        y_pred = active_learner.classifier.predict(train)
        y_pred_test = active_learner.classifier.predict(test)

        test_acc = accuracy_score(y_pred_test, test.y)

        print("Train accuracy: {:.2f}".format(accuracy_score(y_pred, train.y)))
        print("Test accuracy: {:.2f}".format(test_acc))

        return test_acc

    results_setfit = []
    results_setfit.append(evaluate(active_learner, train[labeled_indices], test))

    for i in trange(num_queries):
        # ...where each iteration consists of labelling 20 samples
        q_indices = active_learner.query(num_samples=num_sents // (num_queries+1))

        # Simulate user interaction here. Replace this for real-world usage.
        y = train.y[q_indices]

        # Return the labels for the current query to the active learner.
        active_learner.update(y)

        # memory fix: https://github.com/UKPLab/sentence-transformers/issues/1793
        gc.collect()
        torch.cuda.empty_cache()

        labeled_indices = np.concatenate([q_indices, labeled_indices])

        print("---------------")
        print("Iteration #{:d} ({} samples)".format(i, len(labeled_indices)))
        results_setfit.append(evaluate(active_learner, train[labeled_indices], test))

    print(results_setfit)

    # also dump them to disk.
    # this is now case 4
    dumpdir = Path('summaries') / f"{dataset_name.split('/')[-1]}_{num_sents}"
    dumpdir.mkdir(parents=True, exist_ok=True)
    with (dumpdir / f"case_4.json").open("w+") as f:
        json.dump({'accuracy': results_setfit[-1]}, f)


if __name__ == '__main__':
    run()