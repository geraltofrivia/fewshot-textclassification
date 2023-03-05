from typing import Optional, Any, Dict, Union
from setfit import SetFitTrainer, SetFitModel
import numpy as np
import math
import torch
from setfit.trainer import (
    logger,
    losses,
    sentence_pairs_generation,
)
from torch.utils.data import DataLoader


class CustomModel(SetFitModel):

    def is_frozen_head(self):
        requires_grad = False
        for param in self.model_head.parameters():
            requires_grad = requires_grad or param.requires_grad
        return not requires_grad

    def is_frozen_body(self):
        requires_grad = False
        for param in self.model_head.parameters():
            requires_grad = requires_grad or param.requires_grad
        return not requires_grad

    def _prepare_optimizer(
        self,
        learning_rate: float,
        body_learning_rate: Optional[float],
        l2_weight: float,
    ) -> torch.optim.Optimizer:
        body_learning_rate = body_learning_rate or learning_rate
        l2_weight = l2_weight or self.l2_weight
        param_groups = {}
        if not self.is_frozen_body():
            param_groups.append(
                {
                    "params": self.model_body.parameters(),
                    "lr": body_learning_rate,
                    "weight_decay": l2_weight,
                }
            )
        if not self.is_frozen_head():
            param_groups.append(
                {
                    "params": self.model_head.parameters(),
                    "lr": learning_rate,
                    "weight_decay": l2_weight,
                }
            )

        optimizer = torch.optim.AdamW(param_groups)
        return optimizer


class CustomTrainer(SetFitTrainer):
    def train(
        self,
        num_epochs: Optional[int] = None,
        num_epochs_finetune: Optional[int] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        body_learning_rate: Optional[float] = None,
        l2_weight: Optional[float] = None,
        max_length: Optional[int] = None,
        trial: Optional[Union["optuna.Trial", Dict[str, Any]]] = None,
        silent: bool = True,
        do_finetune: bool = True,   # finetune the sentence transformer on the cosine thing
        do_fitclf: bool = True,     # if true, we train the haed+(encoder|None) on the actual classification task
        do_fitclf_trainencoder: bool = False,  # if true it makes sure that the model also trains when doing fitclf
    ):
        """
            We be overwritin'
        """
        # set_seed(
        #     self.seed
        # )  # Seed must be set before instantiating the model when using model_init.

        if trial:  # Trial and model initialization
            self._hp_search_setup(
                trial
            )  # sets trainer parameters and initializes model

        if self.train_dataset is None:
            raise ValueError(
                "Training requires a `train_dataset` given to the `SetFitTrainer` initialization."
            )

        self._validate_column_mapping(self.train_dataset)
        train_dataset = self.train_dataset
        if self.column_mapping is not None:
            logger.info("Applying column mapping to training dataset")
            train_dataset = self._apply_column_mapping(
                self.train_dataset, self.column_mapping
            )

        x_train = train_dataset["text"]
        y_train = train_dataset["label"]
        if self.loss_class is None:
            logger.warning(
                "No `loss_class` detected! Using `CosineSimilarityLoss` as the default."
            )
            self.loss_class = losses.CosineSimilarityLoss

        num_epochs = num_epochs or self.num_epochs
        batch_size = batch_size or self.batch_size
        learning_rate = learning_rate or self.learning_rate

        # If head is logclf -> always do this: why??
        if not self.model.has_differentiable_head or self._freeze:
            # sentence-transformers adaptation

            train_examples = []

            for _ in range(self.num_iterations):
                train_examples = sentence_pairs_generation(
                    np.array(x_train), np.array(y_train), train_examples
                )

            train_dataloader = DataLoader(
                train_examples, shuffle=True, batch_size=batch_size
            )
            train_loss = self.loss_class(self.model.model_body)

            total_train_steps = len(train_dataloader) * num_epochs
            if not silent:
                logger.info("***** Running training *****")
                logger.info(f"  Num examples = {len(train_examples)}")
                logger.info(f"  Num epochs = {num_epochs}")
                logger.info(f"  Total optimization steps = {total_train_steps}")
                logger.info(f"  Total train batch size = {batch_size}")

            warmup_steps = math.ceil(total_train_steps * self.warmup_proportion)
            self.model.model_body.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=num_epochs_finetune,
                optimizer_params={"lr": learning_rate},
                warmup_steps=warmup_steps,
                show_progress_bar=not silent,
                use_amp=self.use_amp,
            )

        if do_fitclf:
            # Train the final classifier
            if not do_fitclf_trainencoder:
                self.model.freeze("body")

            self.model.fit(
                x_train,
                y_train,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,           # for the head or for both head and body
                body_learning_rate=body_learning_rate, # for model body
                l2_weight=l2_weight,
                max_length=max_length,
                show_progress_bar=True,
            )
