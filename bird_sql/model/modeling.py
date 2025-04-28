"""
Model implementation for SQL generation.
"""

import os
from pathlib import Path
from typing import List, Optional, Union

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    PreTrainedModel,
    T5ForConditionalGeneration,
)

from ..config import MAX_OUTPUT_LENGTH, NUM_BEAMS
from .tokenization import SQLTokenizer


class BirdSQLModel:
    """Custom transformer model wrapper for SQL generation."""

    def __init__(
        self,
        model_name_or_path: str,
        tokenizer: Optional[SQLTokenizer] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the BirdSQL model.

        Args:
            model_name_or_path: Name or path of the pre-trained model
            tokenizer: SQLTokenizer instance
            device: Device to use for inference ('cpu', 'cuda', or specific GPU index)
            **kwargs: Additional arguments to pass to the model
        """
        # Set device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        self.model = self._load_model(model_name_or_path, **kwargs)
        self.model.to(self.device)

        # Load tokenizer if not provided
        if tokenizer is None:
            self.tokenizer = SQLTokenizer.from_pretrained(model_name_or_path)
        else:
            self.tokenizer = tokenizer

    def _load_model(self, model_name_or_path: str, **kwargs) -> PreTrainedModel:
        """
        Load a pre-trained model.

        Args:
            model_name_or_path: Name or path of the pre-trained model
            **kwargs: Additional arguments to pass to the model

        Returns:
            Pre-trained model
        """
        # Load model from Hugging Face
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, **kwargs)

        # Resize token embeddings if needed
        if hasattr(self, "tokenizer") and self.tokenizer is not None:
            model.resize_token_embeddings(len(self.tokenizer.get_tokenizer()))

        return model

    def generate_sql(
        self,
        question: str,
        schema: str,
        num_beams: int = NUM_BEAMS,
        max_length: int = MAX_OUTPUT_LENGTH,
        early_stopping: bool = True,
        **kwargs,
    ) -> str:
        """
        Generate a SQL query for a given question and schema.

        Args:
            question: Natural language question
            schema: Database schema string
            num_beams: Number of beams for beam search
            max_length: Maximum output sequence length
            early_stopping: Whether to stop generation when all beams are finished
            **kwargs: Additional arguments to pass to the model's generate method

        Returns:
            Generated SQL query
        """
        # Encode input
        inputs = self.tokenizer.encode_input(
            question=question,
            schema=schema,
            return_tensors="pt",
        )

        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate output
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                num_beams=num_beams,
                max_length=max_length,
                early_stopping=early_stopping,
                **kwargs,
            )

        # Decode output
        output_text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[
            0
        ]

        # Extract SQL query
        sql_query = self.tokenizer.extract_sql_query(output_text)

        return sql_query

    def generate_batch(
        self,
        questions: List[str],
        schemas: List[str],
        batch_size: int = 8,
        num_beams: int = NUM_BEAMS,
        max_length: int = MAX_OUTPUT_LENGTH,
        **kwargs,
    ) -> List[str]:
        """
        Generate SQL queries for a batch of questions and schemas.

        Args:
            questions: List of natural language questions
            schemas: List of database schema strings
            batch_size: Batch size for generation
            num_beams: Number of beams for beam search
            max_length: Maximum output sequence length
            **kwargs: Additional arguments to pass to the model's generate method

        Returns:
            List of generated SQL queries
        """
        if len(questions) != len(schemas):
            raise ValueError("Number of questions and schemas must match")

        results = []

        # Process in batches
        for i in range(0, len(questions), batch_size):
            batch_questions = questions[i : i + batch_size]  # noqa: E203
            batch_schemas = schemas[i : i + batch_size]  # noqa: E203

            # Encode inputs
            batch_inputs = [
                self.tokenizer.encode_input(
                    question=q,
                    schema=s,
                    return_tensors="pt",
                )
                for q, s in zip(batch_questions, batch_schemas)
            ]

            # Combine batch inputs
            batch_input_ids = torch.cat(
                [inputs["input_ids"] for inputs in batch_inputs], dim=0
            ).to(self.device)
            batch_attention_mask = torch.cat(
                [inputs["attention_mask"] for inputs in batch_inputs], dim=0
            ).to(self.device)

            # Generate outputs
            with torch.no_grad():
                batch_output_ids = self.model.generate(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    num_beams=num_beams,
                    max_length=max_length,
                    **kwargs,
                )

            # Decode outputs
            batch_outputs = self.tokenizer.batch_decode(
                batch_output_ids, skip_special_tokens=True
            )

            # Extract SQL queries
            batch_sql_queries = [
                self.tokenizer.extract_sql_query(output) for output in batch_outputs
            ]

            results.extend(batch_sql_queries)

        return results

    def save_pretrained(self, save_directory: Union[str, Path]) -> None:
        """
        Save the model and tokenizer to a directory.

        Args:
            save_directory: Directory to save the model and tokenizer
        """
        save_directory = (
            Path(save_directory) if isinstance(save_directory, str) else save_directory
        )
        os.makedirs(save_directory, exist_ok=True)

        # Save model
        self.model.save_pretrained(save_directory)

        # Save tokenizer
        self.tokenizer.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        tokenizer: Optional[SQLTokenizer] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> "BirdSQLModel":
        """
        Load a model from a pre-trained model.

        Args:
            model_name_or_path: Name or path of the pre-trained model
            tokenizer: SQLTokenizer instance
            device: Device to use for inference ('cpu', 'cuda', or specific GPU index)
            **kwargs: Additional arguments to pass to the model

        Returns:
            BirdSQLModel instance
        """
        return cls(
            model_name_or_path=model_name_or_path,
            tokenizer=tokenizer,
            device=device,
            **kwargs,
        )


class T5SQLModel(BirdSQLModel):
    """T5-based model for SQL generation."""

    def _load_model(self, model_name_or_path: str, **kwargs) -> PreTrainedModel:
        """
        Load a pre-trained T5 model.

        Args:
            model_name_or_path: Name or path of the pre-trained model
            **kwargs: Additional arguments to pass to the model

        Returns:
            Pre-trained T5 model
        """
        # Load T5 model from Hugging Face
        model = T5ForConditionalGeneration.from_pretrained(model_name_or_path, **kwargs)

        # Resize token embeddings if needed
        if hasattr(self, "tokenizer") and self.tokenizer is not None:
            model.resize_token_embeddings(len(self.tokenizer.get_tokenizer()))

        return model
