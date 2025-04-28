"""
Data loading utilities for SQL datasets.
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd
from torch.utils.data import Dataset

from ..config import MAX_INPUT_LENGTH, MAX_OUTPUT_LENGTH
from .schemas import SchemaProcessor


class SQLDataset(Dataset):
    """Dataset for SQL generation tasks."""

    def __init__(
        self,
        base_path: str,
        split: str = "train",
        tokenizer=None,
        max_input_length: int = MAX_INPUT_LENGTH,
        max_output_length: int = MAX_OUTPUT_LENGTH,
    ):
        """
        Initialize the SQLDataset.

        Args:
            base_path: Path to the dataset directory
            split: Dataset split ('train' or 'dev')
            tokenizer: Tokenizer for encoding inputs and outputs
            max_input_length: Maximum input sequence length
            max_output_length: Maximum output sequence length
        """
        self.base_path = Path(base_path)
        self.split = split
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

        # Paths to dataset files
        self.data_file = self.base_path / f"{split}.json"
        self.tables_file = self.base_path / f"{split}_tables.json"
        self.db_dir = self.base_path / f"{split}_databases"

        # Validate paths
        self._validate_paths()

        # Load data
        self.examples = self._load_examples()
        self.schema_processor = SchemaProcessor(str(self.tables_file))

    def _validate_paths(self) -> None:
        """Validate that all required dataset files exist."""
        if not self.data_file.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_file}")
        if not self.tables_file.exists():
            raise FileNotFoundError(f"Tables file not found: {self.tables_file}")
        if not self.db_dir.exists():
            raise FileNotFoundError(f"Database directory not found: {self.db_dir}")

    def _load_examples(self) -> List[Dict]:
        """Load examples from the data file."""
        with open(self.data_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a dataset item by index.

        Returns:
            A dictionary containing:
                - input_ids: Tokenized input sequence
                - attention_mask: Attention mask for input sequence
                - labels: Tokenized output sequence (if training)
                - example: Original example data
        """
        example = self.examples[idx]

        # Get schema for the database
        db_id = example["db_id"]
        schema_str = self.schema_processor.format_schema_for_model(db_id)

        # Format input and output
        input_text = self._format_input(example["question"], schema_str)
        output_text = self._format_output(example["SQL"])

        # Tokenize input and output
        if self.tokenizer:
            # Tokenize input
            inputs = self.tokenizer.encode_input(
                question=example["question"],
                schema=schema_str,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            # Tokenize output for training
            outputs = self.tokenizer.encode_output(
                sql_query=example["SQL"],
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            # Create the final item
            item = {
                "input_ids": inputs.input_ids.squeeze(),
                "attention_mask": inputs.attention_mask.squeeze(),
                "labels": outputs.input_ids.squeeze(),
                "example": example,
            }
        else:
            # If no tokenizer is provided, return text only
            item = {
                "input_text": input_text,
                "output_text": output_text,
                "example": example,
            }

        return item

    def _format_input(self, question: str, schema: str) -> str:
        """Format the input for the model."""
        return f"Question: {question} | Schema: {schema}"

    def _format_output(self, query: str) -> str:
        """Format the output for the model."""
        return f"SQL: {query}"

    def get_database_path(self, db_id: str) -> Path:
        """Get the path to a specific database file."""
        # Handle nested structure in dev_databases
        if self.split == "dev":
            db_path = self.db_dir / "dev_databases" / db_id / f"{db_id}.sqlite"
        else:
            db_path = self.db_dir / "train_databases" / db_id / f"{db_id}.sqlite"

        if not db_path.exists():
            raise FileNotFoundError(f"Database file not found: {db_path}")

        return db_path

    def execute_query(
        self, db_id: str, query: str
    ) -> Tuple[bool, Union[pd.DataFrame, str]]:
        """
        Execute a SQL query against the database.

        Args:
            db_id: Database ID
            query: SQL query to execute

        Returns:
            Tuple of (success, result)
                - success: Boolean indicating if query executed successfully
                - result: DataFrame with results if successful, error message if not
        """
        try:
            db_path = self.get_database_path(db_id)
            conn = sqlite3.connect(str(db_path))

            # Execute query and fetch results
            result = pd.read_sql_query(query, conn)
            conn.close()

            return True, result
        except Exception as e:
            return False, str(e)
