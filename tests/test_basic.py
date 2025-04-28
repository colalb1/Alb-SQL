"""
Basic tests for the BIRD-SQL system.
"""

import os
import sys
import unittest
from pathlib import Path

# Add the parent directory to the Python path
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, parent_dir)

# Import after path setup
# fmt: off
from bird_sql.config import DEV_DATA_PATH, MODEL_NAME  # noqa: E402
from bird_sql.data.loader import SQLDataset  # noqa: E402
from bird_sql.data.schemas import SchemaProcessor  # noqa: E402
from bird_sql.model.tokenization import SQLTokenizer  # noqa: E402

# fmt: on


class TestBirdSQL(unittest.TestCase):
    """Basic tests for the BIRD-SQL system."""

    def setUp(self):
        """Set up test fixtures."""
        # Skip tests if data paths don't exist
        if not os.path.exists(DEV_DATA_PATH):
            self.skipTest(f"Dev data path does not exist: {DEV_DATA_PATH}")

    def test_schema_processor(self):
        """Test that the schema processor works correctly."""
        tables_json_path = os.path.join(DEV_DATA_PATH, "dev_tables.json")
        if not os.path.exists(tables_json_path):
            self.skipTest(f"Tables JSON file does not exist: {tables_json_path}")

        schema_processor = SchemaProcessor(tables_json_path)
        self.assertIsNotNone(schema_processor)

        # Get all database IDs
        db_ids = set()
        for key in schema_processor.tables.keys():
            db_id = key.split(".")[0]
            db_ids.add(db_id)

        # Check that we have at least one database
        self.assertGreater(len(db_ids), 0)

        # Check that we can format a schema for a database
        db_id = list(db_ids)[0]
        schema_str = schema_processor.format_schema_for_model(db_id)
        self.assertIsNotNone(schema_str)
        self.assertGreater(len(schema_str), 0)

    def test_dataset_loader(self):
        """Test that the dataset loader works correctly."""
        try:
            dataset = SQLDataset(
                base_path=DEV_DATA_PATH,
                split="dev",
                tokenizer=None,
            )
            self.assertIsNotNone(dataset)
            self.assertGreater(len(dataset), 0)

            # Check that we can get an example
            example = dataset.examples[0]
            self.assertIsNotNone(example)
            self.assertIn("db_id", example)
            self.assertIn("question", example)
            self.assertIn("query", example)
        except Exception as e:
            self.fail(f"Dataset loader raised an exception: {e}")

    def test_tokenizer(self):
        """Test that the tokenizer works correctly."""
        try:
            tokenizer = SQLTokenizer(MODEL_NAME)
            self.assertIsNotNone(tokenizer)

            # Test encoding
            encoded = tokenizer.encode_input(
                question="What is the average salary?",
                schema="[TABLE] employees\n[COLUMN] id int\n[COLUMN] name text\n[COLUMN] salary int",
            )
            self.assertIsNotNone(encoded)
            self.assertIn("input_ids", encoded)
            self.assertIn("attention_mask", encoded)

            # Test decoding
            decoded = tokenizer.decode(encoded["input_ids"][0])
            self.assertIsNotNone(decoded)
            self.assertGreater(len(decoded), 0)
        except Exception as e:
            self.fail(f"Tokenizer raised an exception: {e}")


if __name__ == "__main__":
    unittest.main()
