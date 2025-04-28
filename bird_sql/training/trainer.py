"""
Training pipeline for SQL generation models.
"""

import logging
import os
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    PreTrainedModel,
    T5ForConditionalGeneration,
    get_linear_schedule_with_warmup,
)

from ..config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    EVAL_STEPS,
    FP16_TRAINING,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    LOGGING_STEPS,
    MAX_INPUT_LENGTH,
    MAX_OUTPUT_LENGTH,
    NUM_BEAMS,
    NUM_EPOCHS,
    SAVE_STEPS,
    WARMUP_RATIO,
    WEIGHT_DECAY,
)
from ..data.loader import SQLDataset
from ..model.modeling import BirdSQLModel
from ..model.tokenization import SQLTokenizer
from .metrics import EvaluationMetrics


class TrainingPipeline:
    """Training pipeline for SQL generation models."""

    def __init__(
        self,
        model_name_or_path: str,
        train_dataset: Union[SQLDataset, Dataset],
        eval_dataset: Optional[Union[SQLDataset, Dataset]] = None,
        tokenizer: Optional[SQLTokenizer] = None,
        output_dir: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = BATCH_SIZE,
        gradient_accumulation_steps: int = GRADIENT_ACCUMULATION_STEPS,
        learning_rate: float = LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY,
        warmup_ratio: float = WARMUP_RATIO,
        num_epochs: int = NUM_EPOCHS,
        eval_steps: int = EVAL_STEPS,
        save_steps: int = SAVE_STEPS,
        logging_steps: int = LOGGING_STEPS,
        fp16: bool = FP16_TRAINING,
        num_beams: int = NUM_BEAMS,
        max_input_length: int = MAX_INPUT_LENGTH,
        max_output_length: int = MAX_OUTPUT_LENGTH,
        **kwargs,
    ):
        """
        Initialize the training pipeline.

        Args:
            model_name_or_path: Name or path of the pre-trained model
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            tokenizer: SQLTokenizer instance
            output_dir: Directory to save model checkpoints
            device: Device to use for training ('cpu', 'cuda', or specific GPU index)
            batch_size: Batch size for training
            gradient_accumulation_steps: Number of steps to accumulate gradients
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            warmup_ratio: Ratio of warmup steps
            num_epochs: Number of training epochs
            eval_steps: Number of steps between evaluations
            save_steps: Number of steps between saving checkpoints
            logging_steps: Number of steps between logging
            fp16: Whether to use mixed precision training
            num_beams: Number of beams for beam search during evaluation
            max_input_length: Maximum input sequence length
            max_output_length: Maximum output sequence length
            **kwargs: Additional arguments to pass to the model
        """
        # Set device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Set training parameters
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.num_epochs = num_epochs
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.logging_steps = logging_steps
        self.fp16 = fp16
        self.num_beams = num_beams
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

        # Set output directory
        self.output_dir = output_dir or CHECKPOINT_DIR
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize logger
        self.logger = logging.getLogger(__name__)

        # Load tokenizer if not provided
        if tokenizer is None:
            self.tokenizer = SQLTokenizer.from_pretrained(
                model_name_or_path,
                max_input_length=max_input_length,
                max_output_length=max_output_length,
            )
        else:
            self.tokenizer = tokenizer

        # Load model
        self.model = self._load_model(model_name_or_path, **kwargs)
        self.model.to(self.device)

        # Set datasets
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        # Initialize optimizer and scheduler
        self.optimizer = None
        self.scheduler = None

        # Initialize metrics
        self.metrics = EvaluationMetrics()

        # Initialize training state
        self.global_step = 0
        self.epoch = 0
        self.best_metric = 0.0

    def _load_model(self, model_name_or_path: str, **kwargs) -> PreTrainedModel:
        """
        Load a pre-trained model.

        Args:
            model_name_or_path: Name or path of the pre-trained model
            **kwargs: Additional arguments to pass to the model

        Returns:
            Pre-trained model
        """
        # Check if model is T5-based
        if "t5" in model_name_or_path.lower():
            model = T5ForConditionalGeneration.from_pretrained(
                model_name_or_path, **kwargs
            )
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, **kwargs)

        # Resize token embeddings if needed
        model.resize_token_embeddings(len(self.tokenizer.get_tokenizer()))

        return model

    def _create_optimizer_and_scheduler(self, num_training_steps: int) -> None:
        """
        Create optimizer and learning rate scheduler.

        Args:
            num_training_steps: Total number of training steps
        """
        # Create optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            eps=1e-8,
        )

        # Create scheduler
        warmup_steps = int(num_training_steps * self.warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )

    def train(self) -> Dict[str, float]:
        """
        Train the model.

        Returns:
            Dictionary of training metrics
        """
        # Create data loader
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        # Calculate total number of training steps
        num_update_steps_per_epoch = (
            len(train_dataloader) // self.gradient_accumulation_steps
        )
        num_training_steps = num_update_steps_per_epoch * self.num_epochs

        # Create optimizer and scheduler
        self._create_optimizer_and_scheduler(num_training_steps)

        # Initialize progress bar
        progress_bar = tqdm(total=num_training_steps, desc="Training")

        # Initialize training metrics
        train_losses = []

        # Initialize mixed precision scaler if needed
        scaler = torch.amp.GradScaler("cuda") if self.fp16 else None

        # Training loop
        self.model.train()
        for epoch in range(self.num_epochs):
            self.epoch = epoch
            epoch_losses = []

            for step, batch in enumerate(train_dataloader):
                # Move batch to device
                batch = {
                    k: v.to(self.device)
                    for k, v in batch.items()
                    if isinstance(v, torch.Tensor)
                }

                # Forward pass with mixed precision if enabled
                if self.fp16:
                    with torch.amp.autocast("cuda"):
                        outputs = self.model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            labels=batch["labels"],
                        )
                        loss = outputs.loss
                else:
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                    )
                    loss = outputs.loss

                # Scale loss by gradient accumulation steps
                loss = loss / self.gradient_accumulation_steps

                # Backward pass with mixed precision if enabled
                if self.fp16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Record loss
                epoch_losses.append(loss.item() * self.gradient_accumulation_steps)

                # Update weights if gradient accumulation is complete
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    # Clip gradients
                    if self.fp16:
                        scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optimizer.step()

                    # Update learning rate
                    self.scheduler.step()

                    # Reset gradients
                    self.optimizer.zero_grad()

                    # Update global step
                    self.global_step += 1
                    progress_bar.update(1)

                    # Log training progress
                    if self.global_step % self.logging_steps == 0:
                        avg_loss = np.mean(
                            epoch_losses[-self.logging_steps :]
                        )  # noqa: E203
                        self.logger.info(
                            f"Step {self.global_step}: loss = {avg_loss:.4f}"
                        )

                    # Evaluate model
                    if (
                        self.eval_dataset is not None
                        and self.global_step % self.eval_steps == 0
                    ):
                        eval_metrics = self.evaluate()
                        self.model.train()

                        # Log evaluation metrics
                        self.logger.info(
                            f"Step {self.global_step}: eval_exact_match = {eval_metrics['exact_match']:.4f}"
                        )

                        # Save best model
                        if eval_metrics["exact_match"] > self.best_metric:
                            self.best_metric = eval_metrics["exact_match"]
                            self.save_model(os.path.join(self.output_dir, "best_model"))

                    # Save checkpoint
                    if self.global_step % self.save_steps == 0:
                        self.save_model(
                            os.path.join(
                                self.output_dir, f"checkpoint-{self.global_step}"
                            )
                        )

            # Record epoch loss
            train_losses.extend(epoch_losses)
            avg_epoch_loss = np.mean(epoch_losses)
            self.logger.info(
                f"Epoch {epoch + 1}/{self.num_epochs}: loss = {avg_epoch_loss:.4f}"
            )

        # Save final model
        self.save_model(os.path.join(self.output_dir, "final_model"))

        # Calculate training metrics
        train_metrics = {
            "loss": np.mean(train_losses),
        }

        # Evaluate final model
        if self.eval_dataset is not None:
            eval_metrics = self.evaluate()
            train_metrics.update(eval_metrics)

        return train_metrics

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model.

        Returns:
            Dictionary of evaluation metrics
        """
        if self.eval_dataset is None:
            return {}

        # Create data loader
        eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        # Initialize evaluation metrics
        eval_losses = []
        all_predictions = []
        all_references = []

        # Evaluation loop
        self.model.eval()
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            # Move batch to device
            batch = {
                k: v.to(self.device)
                for k, v in batch.items()
                if isinstance(v, torch.Tensor)
            }

            with torch.no_grad():
                # Calculate loss
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss
                eval_losses.append(loss.item())

                # Generate predictions
                generated_ids = self.model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    num_beams=self.num_beams,
                    max_length=self.max_output_length,
                    early_stopping=True,
                )

                # Decode predictions and references
                predictions = self.tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True
                )
                references = self.tokenizer.batch_decode(
                    batch["labels"], skip_special_tokens=True
                )

                # Extract SQL queries
                predictions = [
                    self.tokenizer.extract_sql_query(pred) for pred in predictions
                ]
                references = [
                    self.tokenizer.extract_sql_query(ref) for ref in references
                ]

                all_predictions.extend(predictions)
                all_references.extend(references)

        # Calculate metrics
        exact_match = EvaluationMetrics.exact_match(all_predictions, all_references)
        token_metrics = EvaluationMetrics.token_level_metrics(
            all_predictions, all_references
        )
        component_metrics = EvaluationMetrics.component_match(
            all_predictions, all_references
        )

        # Combine metrics
        metrics = {
            "loss": np.mean(eval_losses),
            "exact_match": exact_match,
            **token_metrics,
            **component_metrics,
        }

        return metrics

    def predict(
        self,
        questions: List[str],
        schemas: List[str],
        batch_size: Optional[int] = None,
    ) -> List[str]:
        """
        Generate SQL queries for a list of questions and schemas.

        Args:
            questions: List of natural language questions
            schemas: List of database schema strings
            batch_size: Batch size for prediction

        Returns:
            List of generated SQL queries
        """
        if len(questions) != len(schemas):
            raise ValueError("Number of questions and schemas must match")

        batch_size = batch_size or self.batch_size

        # Create BirdSQLModel for prediction
        bird_model = BirdSQLModel(
            model_name_or_path="",  # Not used since we're providing the model directly
            tokenizer=self.tokenizer,
            device=self.device,
        )
        bird_model.model = self.model

        # Generate SQL queries
        predictions = bird_model.generate_batch(
            questions=questions,
            schemas=schemas,
            batch_size=batch_size,
            num_beams=self.num_beams,
            max_length=self.max_output_length,
        )

        return predictions

    def save_model(self, output_dir: str) -> None:
        """
        Save the model and tokenizer.

        Args:
            output_dir: Directory to save the model and tokenizer
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save model
        self.model.save_pretrained(output_dir)

        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)

        # Save training arguments
        training_args = {
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_ratio": self.warmup_ratio,
            "num_epochs": self.num_epochs,
            "eval_steps": self.eval_steps,
            "save_steps": self.save_steps,
            "logging_steps": self.logging_steps,
            "fp16": self.fp16,
            "num_beams": self.num_beams,
            "max_input_length": self.max_input_length,
            "max_output_length": self.max_output_length,
        }

        with open(os.path.join(output_dir, "training_args.json"), "w") as f:
            import json

            json.dump(training_args, f, indent=2)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        train_dataset: Optional[Union[SQLDataset, Dataset]] = None,
        eval_dataset: Optional[Union[SQLDataset, Dataset]] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> "TrainingPipeline":
        """
        Load a training pipeline from a pre-trained model.

        Args:
            model_path: Path to the pre-trained model
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            device: Device to use for training
            **kwargs: Additional arguments to override training arguments

        Returns:
            TrainingPipeline instance
        """
        # Load tokenizer
        tokenizer = SQLTokenizer.from_pretrained(model_path)

        # Load training arguments
        training_args = {}
        args_path = os.path.join(model_path, "training_args.json")
        if os.path.exists(args_path):
            with open(args_path, "r") as f:
                import json

                training_args = json.load(f)

        # Override with provided kwargs
        training_args.update(kwargs)

        # Create training pipeline
        pipeline = cls(
            model_name_or_path=model_path,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            device=device,
            **training_args,
        )

        return pipeline
