"""
Main entry point for the BIRD-SQL system.
"""

import argparse
import logging
import os
import sys
from typing import Dict, List, Optional

import torch

from bird_sql.config import (
    BATCH_SIZE,
    DEV_DATA_PATH,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    MODEL_NAME,
    NUM_EPOCHS,
    TRAIN_DATA_PATH,
    WARMUP_RATIO,
)
from bird_sql.data.loader import SQLDataset
from bird_sql.model.modeling import BirdSQLModel, T5SQLModel
from bird_sql.model.tokenization import SQLTokenizer
from bird_sql.training.trainer import TrainingPipeline


def setup_logging(log_level: str = "INFO") -> None:
    """
    Set up logging configuration.

    Args:
        log_level: Logging level
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


def train(
    model_name: str = MODEL_NAME,
    train_data_path: str = TRAIN_DATA_PATH,
    dev_data_path: Optional[str] = DEV_DATA_PATH,
    output_dir: str = "./checkpoints",
    batch_size: int = BATCH_SIZE,
    gradient_accumulation_steps: int = GRADIENT_ACCUMULATION_STEPS,
    learning_rate: float = LEARNING_RATE,
    warmup_ratio: float = WARMUP_RATIO,
    num_epochs: int = NUM_EPOCHS,
    fp16: bool = torch.cuda.is_available(),
    device: Optional[str] = None,
) -> Dict:
    """
    Train a BIRD-SQL model.

    Args:
        model_name: Name or path of the pre-trained model
        train_data_path: Path to the training data
        dev_data_path: Path to the development data
        output_dir: Directory to save model checkpoints
        batch_size: Batch size for training
        gradient_accumulation_steps: Number of steps to accumulate gradients
        learning_rate: Learning rate for optimizer
        warmup_ratio: Ratio of warmup steps
        num_epochs: Number of training epochs
        fp16: Whether to use mixed precision training
        device: Device to use for training

    Returns:
        Dictionary of training metrics
    """
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load tokenizer
    logger.info(f"Loading tokenizer from {model_name}")
    tokenizer = SQLTokenizer.from_pretrained(model_name)

    # Load datasets
    logger.info(f"Loading training data from {train_data_path}")
    train_dataset = SQLDataset(
        base_path=train_data_path,
        split="train",
        tokenizer=tokenizer,
    )

    if dev_data_path:
        logger.info(f"Loading development data from {dev_data_path}")
        dev_dataset = SQLDataset(
            base_path=dev_data_path,
            split="dev",
            tokenizer=tokenizer,
        )
    else:
        dev_dataset = None

    # Create training pipeline
    logger.info(f"Creating training pipeline with model {model_name}")
    pipeline = TrainingPipeline(
        model_name_or_path=model_name,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        output_dir=output_dir,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        num_epochs=num_epochs,
        fp16=fp16,
        device=device,
    )

    # Train model
    logger.info("Starting training")
    metrics = pipeline.train()

    # Log metrics
    logger.info("Training completed")
    for key, value in metrics.items():
        logger.info(f"{key}: {value:.4f}")

    return metrics


def evaluate(
    model_path: str,
    data_path: str = DEV_DATA_PATH,
    batch_size: int = BATCH_SIZE,
    device: Optional[str] = None,
) -> Dict:
    """
    Evaluate a BIRD-SQL model.

    Args:
        model_path: Path to the model
        data_path: Path to the evaluation data
        batch_size: Batch size for evaluation
        device: Device to use for evaluation

    Returns:
        Dictionary of evaluation metrics
    """
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)

    # Load tokenizer
    logger.info(f"Loading tokenizer from {model_path}")
    tokenizer = SQLTokenizer.from_pretrained(model_path)

    # Load dataset
    logger.info(f"Loading evaluation data from {data_path}")
    eval_dataset = SQLDataset(
        base_path=data_path,
        split="dev",
        tokenizer=tokenizer,
    )

    # Create training pipeline
    logger.info(f"Loading model from {model_path}")
    pipeline = TrainingPipeline.from_pretrained(
        model_path=model_path,
        eval_dataset=eval_dataset,
        batch_size=batch_size,
        device=device,
    )

    # Evaluate model
    logger.info("Starting evaluation")
    metrics = pipeline.evaluate()

    # Log metrics
    logger.info("Evaluation completed")
    for key, value in metrics.items():
        logger.info(f"{key}: {value:.4f}")

    return metrics


def predict(
    model_path: str,
    questions: List[str],
    schemas: List[str],
    batch_size: int = BATCH_SIZE,
    device: Optional[str] = None,
) -> List[str]:
    """
    Generate SQL queries for a list of questions and schemas.

    Args:
        model_path: Path to the model
        questions: List of natural language questions
        schemas: List of database schema strings
        batch_size: Batch size for prediction
        device: Device to use for prediction

    Returns:
        List of generated SQL queries
    """
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)

    # Load tokenizer
    logger.info(f"Loading tokenizer from {model_path}")
    tokenizer = SQLTokenizer.from_pretrained(model_path)

    # Load model
    logger.info(f"Loading model from {model_path}")
    if "t5" in model_path.lower():
        model = T5SQLModel.from_pretrained(
            model_name_or_path=model_path,
            tokenizer=tokenizer,
            device=device,
        )
    else:
        model = BirdSQLModel.from_pretrained(
            model_name_or_path=model_path,
            tokenizer=tokenizer,
            device=device,
        )

    # Generate SQL queries
    logger.info(f"Generating SQL queries for {len(questions)} questions")
    predictions = model.generate_batch(
        questions=questions,
        schemas=schemas,
        batch_size=batch_size,
    )

    return predictions


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Alb-SQL: SQL generation from natural language"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument(
        "--model", type=str, default=MODEL_NAME, help="Model name or path"
    )
    train_parser.add_argument(
        "--train-data", type=str, default=TRAIN_DATA_PATH, help="Path to training data"
    )
    train_parser.add_argument(
        "--dev-data", type=str, default=DEV_DATA_PATH, help="Path to development data"
    )
    train_parser.add_argument(
        "--output-dir", type=str, default="./checkpoints", help="Output directory"
    )
    train_parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE, help="Batch size"
    )
    train_parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=GRADIENT_ACCUMULATION_STEPS,
        help="Gradient accumulation steps",
    )
    train_parser.add_argument(
        "--learning-rate", type=float, default=LEARNING_RATE, help="Learning rate"
    )
    train_parser.add_argument(
        "--warmup-ratio", type=float, default=WARMUP_RATIO, help="Warmup ratio"
    )
    train_parser.add_argument(
        "--num-epochs", type=int, default=NUM_EPOCHS, help="Number of epochs"
    )
    train_parser.add_argument(
        "--fp16", action="store_true", help="Use mixed precision training"
    )
    train_parser.add_argument(
        "--no-fp16",
        action="store_false",
        dest="fp16",
        help="Disable mixed precision training",
    )
    train_parser.add_argument("--device", type=str, help="Device to use for training")

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a model")
    eval_parser.add_argument("--model", type=str, required=True, help="Model path")
    eval_parser.add_argument(
        "--data", type=str, default=DEV_DATA_PATH, help="Path to evaluation data"
    )
    eval_parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE, help="Batch size"
    )
    eval_parser.add_argument("--device", type=str, help="Device to use for evaluation")

    # Parse arguments
    args = parser.parse_args()

    # Run command
    if args.command == "train":
        train(
            model_name=args.model,
            train_data_path=args.train_data,
            dev_data_path=args.dev_data,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            warmup_ratio=args.warmup_ratio,
            num_epochs=args.num_epochs,
            fp16=args.fp16,
            device=args.device,
        )
    elif args.command == "evaluate":
        evaluate(
            model_path=args.model,
            data_path=args.data,
            batch_size=args.batch_size,
            device=args.device,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
