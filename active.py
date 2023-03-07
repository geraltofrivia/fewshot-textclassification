"""
    Here we be bussin about that active learning thingamajing
"""
import datasets
import logging
import numpy as np

from small_text.integrations.transformers.classifiers.setfit import SetFitModelArguments
from small_text.integrations.transformers.classifiers.factories import SetFitClassificationFactory
from small_text import TextDataset

from small_text import (
    PoolBasedActiveLearner,
    random_initialization_balanced,
    BreakingTies,
    SubsamplingQueryStrategy
)


raw_dataset = datasets.load_dataset("imdb")
num_classes = np.unique(raw_dataset["train"]["label"]).shape[0]
target_labels = np.arange(num_classes)

train = TextDataset.from_arrays(
    raw_dataset["train"]["text"], np.array(raw_dataset["train"]["label"]), target_labels=target_labels
)
test = TextDataset.from_arrays(
    raw_dataset["test"]["text"], np.array(raw_dataset["test"]["label"]), target_labels=target_labels
)


sentence_transformer_model_name = "sentence-transformers/paraphrase-mpnet-base-v2"
setfit_model_args = SetFitModelArguments(sentence_transformer_model_name)
# @pri: this can generate SetFitClassification models which has both setfit trainer and setfit model and does some other stuff
# TODO: we can replace that
clf_factory = SetFitClassificationFactory(setfit_model_args, num_classes)


# define a query strategy and initialize a pool-based active learner
# @pri: this seems fine to change the strategy. we'll have to import two and mix their scores maybe
# TODO: find a strategy and implement it
# TODO: change num of subsamples
query_strategy = SubsamplingQueryStrategy(BreakingTies())
# suppress progress bars in jupyter notebook
setfit_train_kwargs = {'show_progress_bar': False}
active_learner = PoolBasedActiveLearner(clf_factory, query_strategy, train, fit_kwargs={'setfit_train_kwargs': setfit_train_kwargs})


# simulate a warm start
def initialize_active_learner(active_learner, y_train):

    x_indices_initial = random_initialization_balanced(y_train, n_samples=20)
    y_initial = y_train[x_indices_initial]

    active_learner.initialize_data(x_indices_initial, y_initial)

    return x_indices_initial


initial_indices = initialize_active_learner(active_learner, train.y)
labeled_indices = initial_indices

import gc
import torch
from sklearn.metrics import accuracy_score

num_queries = 10


def evaluate(active_learner, train, test):
    y_pred = active_learner.classifier.predict(train)
    y_pred_test = active_learner.classifier.predict(test)

    test_acc = accuracy_score(y_pred_test, test.y)

    print('Train accuracy: {:.2f}'.format(accuracy_score(y_pred, train.y)))
    print('Test accuracy: {:.2f}'.format(test_acc))

    return test_acc

results_setfit = []
results_setfit.append(evaluate(active_learner, train[labeled_indices], test))

for i in range(num_queries):
    # ...where each iteration consists of labelling 20 samples
    q_indices = active_learner.query(num_samples=10)

    # Simulate user interaction here. Replace this for real-world usage.
    y = train.y[q_indices]

    # Return the labels for the current query to the active learner.
    active_learner.update(y)

    # memory fix: https://github.com/UKPLab/sentence-transformers/issues/1793
    gc.collect()
    torch.cuda.empty_cache()

    labeled_indices = np.concatenate([q_indices, labeled_indices])

    print('---------------')
    print('Iteration #{:d} ({} samples)'.format(i, len(labeled_indices)))
    results_setfit.append(evaluate(active_learner, train[labeled_indices], test))


print('potato')
print(results_setfit)
