import math
import os
from pathlib import Path
from typing import Optional, Any, Dict, Union

import joblib
import numpy as np
import requests
import torch
from setfit import SetFitTrainer, SetFitModel
from setfit.modeling import (
    OneVsRestClassifier,
    SentenceTransformer,
    MODEL_HEAD_NAME,
    hf_hub_download,
    SetFitHead,
    LogisticRegression,
    MultiOutputClassifier,
    ClassifierChain
)
from setfit.trainer import (
    logger,
    losses,
    sentence_pairs_generation
)
from torch.utils.data import DataLoader


class CustomModel(SetFitModel):
    @classmethod
    def _from_pretrained(
        cls,
        model_id: str,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        force_download: Optional[bool] = None,
        proxies: Optional[Dict] = None,
        resume_download: Optional[bool] = None,
        local_files_only: Optional[bool] = None,
        use_auth_token: Optional[Union[bool, str]] = None,
        multi_target_strategy: Optional[str] = None,
        use_differentiable_head: bool = False,
        normalize_embeddings: bool = False,
        silent: bool = True,
        **model_kwargs,
    ) -> "SetFitModel":
        """Their classmethod was bad >.< >.<"""
        model_body = SentenceTransformer(
            model_id, cache_folder=cache_dir, use_auth_token=use_auth_token
        )
        target_device = model_body._target_device
        model_body.to(target_device)  # put `model_body` on the target device

        if os.path.isdir(model_id):
            if MODEL_HEAD_NAME in os.listdir(model_id):
                model_head_file = os.path.join(model_id, MODEL_HEAD_NAME)
            else:
                if not silent:
                    logger.info(
                        f"{MODEL_HEAD_NAME} not found in {Path(model_id).resolve()},"
                        " initialising classification head with random weights."
                        " You should TRAIN this model on a downstream task to use it for predictions and inference."
                    )
                model_head_file = None
        else:
            try:
                model_head_file = hf_hub_download(
                    repo_id=model_id,
                    filename=MODEL_HEAD_NAME,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    use_auth_token=use_auth_token,
                    local_files_only=local_files_only,
                )
            except requests.exceptions.RequestException:
                logger.info(
                    f"{MODEL_HEAD_NAME} not found on HuggingFace Hub, initialising classification head with random weights."
                    " You should TRAIN this model on a downstream task to use it for predictions and inference."
                )
                model_head_file = None

        if model_head_file is not None:
            model_head = joblib.load(model_head_file)
        else:
            head_params = model_kwargs.get("head_params", {})
            if use_differentiable_head:
                if multi_target_strategy is None:
                    use_multitarget = False
                else:
                    if multi_target_strategy in ["one-vs-rest", "multi-output"]:
                        use_multitarget = True
                    else:
                        raise ValueError(
                            f"multi_target_strategy '{multi_target_strategy}' is not supported for differentiable head"
                        )
                # Base `model_head` parameters
                # - get the sentence embedding dimension from the `model_body`
                # - follow the `model_body`, put `model_head` on the target device
                base_head_params = {
                    "in_features": model_body.get_sentence_embedding_dimension(),
                    "device": target_device,
                    "multitarget": use_multitarget,
                }
                model_head = SetFitHead(**{**head_params, **base_head_params})
            else:
                clf = LogisticRegression(**head_params)
                if multi_target_strategy is not None:
                    if multi_target_strategy == "one-vs-rest":
                        multilabel_classifier = OneVsRestClassifier(clf)
                    elif multi_target_strategy == "multi-output":
                        multilabel_classifier = MultiOutputClassifier(clf)
                    elif multi_target_strategy == "classifier-chain":
                        multilabel_classifier = ClassifierChain(clf)
                    else:
                        raise ValueError(
                            f"multi_target_strategy {multi_target_strategy} is not supported."
                        )

                    model_head = multilabel_classifier
                else:
                    model_head = clf

        return cls(
            model_body=model_body,
            model_head=model_head,
            multi_target_strategy=multi_target_strategy,
            normalize_embeddings=normalize_embeddings,
        )

    def is_frozen_head(self):
        requires_grad = False
        for param in self.model_head.parameters():
            requires_grad = requires_grad or param.requires_grad
        return not requires_grad

    def is_frozen_body(self):
        requires_grad = False
        for param in self.model_body.parameters():
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
        param_groups = []
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

    # def fit(
    #     self,
    #     x_train: List[str],
    #     y_train: Union[List[int], List[List[int]]],
    #     num_epochs: int,
    #     batch_size: Optional[int] = None,
    #     learning_rate: Optional[float] = None,
    #     body_learning_rate: Optional[float] = None,
    #     l2_weight: Optional[float] = None,
    #     max_length: Optional[int] = None,
    #     show_progress_bar: Optional[bool] = None,
    # ) -> None:
    #     # TODO: delete this function, we dont actually need to override it
    #     if self.has_differentiable_head:  # train with pyTorch
    #         device = self.model_body.device
    #         self.model_body.train()
    #         self.model_head.train()
    #
    #         dataloader = self._prepare_dataloader(x_train, y_train, batch_size, max_length)
    #         criterion = self.model_head.get_loss_fn()
    #         optimizer = self._prepare_optimizer(learning_rate, body_learning_rate, l2_weight)
    #         scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    #         for epoch_idx in tqdm(range(num_epochs), desc="Epoch", disable=not show_progress_bar):
    #             for batch in dataloader:
    #                 features, labels = batch
    #                 optimizer.zero_grad()
    #
    #                 # to model's device
    #                 features = {k: v.to(device) for k, v in features.items()}
    #                 labels = labels.to(device)
    #
    #                 outputs = self.model_body(features)
    #                 print("O: ", outputs)
    #                 print("L: ", labels)
    #                 if self.normalize_embeddings:
    #                     outputs = torch.nn.functional.normalize(outputs, p=2, dim=1)
    #                 outputs = self.model_head(outputs)
    #                 logits = outputs["logits"]
    #
    #                 loss = criterion(logits, labels)
    #                 loss.backward()
    #                 optimizer.step()
    #
    #             scheduler.step()
    #     else:  # train with sklearn
    #         embeddings = self.model_body.encode(x_train, normalize_embeddings=self.normalize_embeddings)
    #         self.model_head.fit(embeddings, y_train)


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
        do_finetune: bool = True,  # finetune the sentence transformer on the cosine thing
        do_fitclf: bool = True,  # if true, we train the haed+(encoder|None) on the actual classification task
        do_fitclf_trainencoder: bool = True,  # if true it makes sure that the model also trains when doing fitclf
    ):
        """
        We be overwritin'
        """
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
        # batch_size = batch_size or self.batch_size
        learning_rate = learning_rate or self.learning_rate

        # If head is logclf -> always do this: why??
        if (not self.model.has_differentiable_head or self._freeze) and (do_finetune):
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

            # y_train = np.array(y_train, dtype=np.float)

            self.model.fit(
                x_train,
                y_train,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,  # for the head or for both head and body
                body_learning_rate=body_learning_rate,  # for model body
                l2_weight=l2_weight,
                max_length=max_length,
                show_progress_bar=True,
            )
