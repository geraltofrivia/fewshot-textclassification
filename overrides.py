from typing import Optional, Any, Dict, Union
from setfit import SetFitTrainer
import numpy as np
import math
from setfit.trainer import (
    set_seed,
    logger,
    losses,
    sentence_pairs_generation,
)
from torch.utils.data import DataLoader


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
    ):
        """
        Main training entry point.

        Args:
            num_epochs (`int`, *optional*):
                Temporary change the number of epochs to train the Sentence Transformer body/head for.
                If ignore, will use the value given in initialization.
            batch_size (`int`, *optional*):
                Temporary change the batch size to use for contrastive training or logistic regression.
                If ignore, will use the value given in initialization.
            learning_rate (`float`, *optional*):
                Temporary change the learning rate to use for contrastive training or SetFitModel's head in logistic regression.
                If ignore, will use the value given in initialization.
            body_learning_rate (`float`, *optional*):
                Temporary change the learning rate to use for SetFitModel's body in logistic regression only.
                If ignore, will be the same as `learning_rate`.
            l2_weight (`float`, *optional*):
                Temporary change the weight of L2 regularization for SetFitModel's differentiable head in logistic regression.
            max_length (int, *optional*, defaults to `None`):
                The maximum number of tokens for one data sample. Currently only for training the differentiable head.
                If `None`, will use the maximum number of tokens the model body can accept.
                If `max_length` is greater than the maximum number of acceptable tokens the model body can accept, it will be set to the maximum number of acceptable tokens.
            trial (`optuna.Trial` or `Dict[str, Any]`, *optional*):
                The trial run or the hyperparameter dictionary for hyperparameter search.
            show_progress_bar (`bool`, *optional*, defaults to `True`):
                Whether to show a bar that indicates training progress.
        """
        set_seed(
            self.seed
        )  # Seed must be set before instantiating the model when using model_init.

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

        if not self.model.has_differentiable_head or not self._freeze:
            # Train the final classifier
            self.model.fit(
                x_train,
                y_train,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                body_learning_rate=body_learning_rate,
                l2_weight=l2_weight,
                max_length=max_length,
                show_progress_bar=True,
            )
