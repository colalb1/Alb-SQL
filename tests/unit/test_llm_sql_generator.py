"""
Test module for the LLM SQL Generator.

This module contains tests for the HuggingFace LLM-based SQL generation
functionality.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from core.llm_sql_generator import (
    HuggingFaceSQLGenerator,
    generate_sql_from_llm,
    generate_sql_from_text,
    generate_sql_json_response,
)


class TestLLMSQLGenerator:
    """Tests for the LLM SQL Generator."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Fixture for a mock tokenizer."""
        tokenizer = MagicMock()
        tokenizer.model_max_length = 512
        tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        tokenizer.decode.return_value = "SELECT * FROM users WHERE age > 18"
        return tokenizer

    @pytest.fixture
    def mock_model(self):
        """Fixture for a mock model."""
        model = MagicMock()
        model.generate.return_value = torch.tensor([[4, 5, 6]])
        return model

    @patch("core.llm_sql_generator.AutoTokenizer")
    @patch("core.llm_sql_generator.AutoModelForSeq2SeqLM")
    def test_init(
        self, mock_auto_model, mock_auto_tokenizer, mock_tokenizer, mock_model
    ):
        """Test initializing the generator."""
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        mock_auto_model.from_pretrained.return_value = mock_model

        # Initialize with default params
        generator = HuggingFaceSQLGenerator()

        # Verify tokenizer and model were loaded
        mock_auto_tokenizer.from_pretrained.assert_called_once()
        mock_auto_model.from_pretrained.assert_called_once()

        # Check device selection logic
        assert generator.device in ["cuda", "cpu"]

    def test_extract_sql_code_block(self):
        """Test SQL extraction from code blocks."""
        # Mock model loading to avoid actual initialization
        with (
            patch("core.llm_sql_generator.AutoTokenizer"),
            patch("core.llm_sql_generator.AutoModelForSeq2SeqLM"),
        ):
            generator = HuggingFaceSQLGenerator()

            # Test with SQL in code block
            text = """Here is the SQL query:

```sql
SELECT * FROM users WHERE age > 18
```

I hope this helps!"""

            sql = generator._extract_sql(text)
            assert sql == "SELECT * FROM users WHERE age > 18"

    def test_extract_sql_no_code_block(self):
        """Test SQL extraction without code blocks."""
        # Mock model loading to avoid actual initialization
        with (
            patch("core.llm_sql_generator.AutoTokenizer"),
            patch("core.llm_sql_generator.AutoModelForSeq2SeqLM"),
        ):
            generator = HuggingFaceSQLGenerator()

            # Test with SQL without code block
            text = "Here is the SQL query: SELECT * FROM users WHERE age > 18 Hope this helps!"

            sql = generator._extract_sql(text)
            assert "SELECT * FROM users WHERE age > 18" in sql

    def test_build_prompt(self):
        """Test prompt building."""
        # Mock model loading to avoid actual initialization
        with (
            patch("core.llm_sql_generator.AutoTokenizer"),
            patch("core.llm_sql_generator.AutoModelForSeq2SeqLM"),
        ):
            generator = HuggingFaceSQLGenerator()

            # Test without schema
            prompt = generator._build_prompt("Find all users over 18")
            assert prompt == "Question: Find all users over 18"

            # Test with schema
            schema = "Table: users\nColumns: id, name, age"
            prompt = generator._build_prompt("Find all users over 18", schema)
            assert "Schema: " in prompt
            assert "Question: Find all users over 18" in prompt

    def test_truncate_prompt(self):
        """Test prompt truncation."""
        # Mock model loading to avoid actual initialization
        with (
            patch("core.llm_sql_generator.AutoTokenizer"),
            patch("core.llm_sql_generator.AutoModelForSeq2SeqLM"),
        ):
            generator = HuggingFaceSQLGenerator()

            # Create a long prompt
            long_prompt = "word " * 1000
            truncated = generator._truncate_prompt(long_prompt, 100)

            # Should be truncated to expected length
            assert len(truncated.split()) <= 100

            # Test schema + question truncation
            schema_query = "Schema: " + "table " * 200 + "\nQuestion: find users"
            truncated = generator._truncate_prompt(schema_query, 50)

            # Should preserve the question part
            assert "Question: find users" in truncated

    @patch("core.llm_sql_generator.AutoTokenizer")
    @patch("core.llm_sql_generator.AutoModelForSeq2SeqLM")
    def test_generate_sql(
        self, mock_auto_model, mock_auto_tokenizer, mock_tokenizer, mock_model
    ):
        """Test SQL generation method."""
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        mock_auto_model.from_pretrained.return_value = mock_model

        # Initialize generator with mocks
        generator = HuggingFaceSQLGenerator()

        # Test basic generation
        sql = generator.generate_sql("Find all users over 18")

        # Verify model was called and result processed
        mock_model.generate.assert_called_once()
        assert isinstance(sql, str)
        assert "SELECT * FROM users WHERE age > 18" in sql

    @patch("core.llm_sql_generator.HuggingFaceSQLGenerator")
    def test_generate_sql_from_text(self, mock_generator_class):
        """Test the generate_sql_from_text function."""
        # Setup mock instance
        mock_instance = MagicMock()
        mock_instance.generate_sql.return_value = "SELECT * FROM users WHERE age > 18"
        mock_generator_class.return_value = mock_instance

        # Call the function
        result = generate_sql_from_text("Find all users over 18")

        # Verify results
        mock_generator_class.assert_called_once()
        mock_instance.generate_sql.assert_called_once()
        assert result == "SELECT * FROM users WHERE age > 18"

    @patch("core.llm_sql_generator.generate_sql_from_text")
    def test_generate_sql_json_response(self, mock_generate):
        """Test the JSON response function."""
        mock_generate.return_value = "SELECT * FROM users WHERE age > 18"

        # Call function
        result = generate_sql_json_response("Find all users over 18")

        # Verify results
        mock_generate.assert_called_once()
        assert isinstance(result, dict)
        assert "sql" in result
        assert result["sql"] == "SELECT * FROM users WHERE age > 18"

    @patch("core.llm_sql_generator.HuggingFaceSQLGenerator")
    def test_backwards_compatibility(self, mock_generator_class):
        """Test backwards compatibility with generate_sql_from_llm."""
        # Setup mock instance
        mock_instance = MagicMock()
        mock_instance.generate_sql.return_value = "SELECT * FROM users WHERE age > 18"
        mock_generator_class.return_value = mock_instance

        # Call with old function name
        result = generate_sql_from_llm("Find all users over 18")

        # Verify results
        mock_generator_class.assert_called_once()
        mock_instance.generate_sql.assert_called_once()
        assert result == "SELECT * FROM users WHERE age > 18"


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
