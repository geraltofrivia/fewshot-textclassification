"""
    First we're going to just follow the thing and play with the metrics once done

"""

from datasets import load_dataset
from sentence_transformers.losses import CosineSimilarityLoss
from collections import Counter
from setfit import SetFitModel, SetFitTrainer


ds_senteval = load_dataset("SetFit/SentEval-CR")
# ds_imdb = load_dataset("imdb")

train_ds = ds_senteval["train"].shuffle(seed=42).select(range(8 * 2))
test_ds = ds_senteval["test"]

print(ds_senteval)

model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2")
trainer = SetFitTrainer(
    model=model,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    loss_class=CosineSimilarityLoss,
    batch_size=16,
    num_iterations=20, # Number of text pairs to generate for contrastive learning
    num_epochs=1 # Number of epochs to use for contrastive learning
)
trainer.train(body_learning_rate=0.0)
metrics = trainer.evaluate()

print(metrics)
print('potato')