"""
Test module for the LLM SQL Generator.

This module contains tests for the HuggingFace LLM-based SQL generation
functionality.
"""

from unittest.mock import MagicMock, patch

import pytest

from core.llm_sql_generator import HuggingFaceSQLGenerator, generate_sql_from_llm


class TestLLMSQLGenerator:
    """Tests for the LLM SQL Generator."""

    def test_extract_sql_code_block(self):
        """Test SQL extraction from code blocks."""
        generator = HuggingFaceSQLGenerator(use_pipeline=False)

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
        generator = HuggingFaceSQLGenerator(use_pipeline=False)

        # Test with SQL without code block
        text = """SELECT * FROM users 
        WHERE age > 18 
        ORDER BY name"""

        sql = generator._extract_sql(text)
        assert "SELECT * FROM users" in sql
        assert "WHERE age > 18" in sql

    def test_build_prompt(self):
        """Test prompt building."""
        generator = HuggingFaceSQLGenerator(use_pipeline=False)

        question = "What is the average age of users?"
        schema = "Table: users\nColumns: id INT, name VARCHAR, age INT"

        prompt = generator._build_prompt(question, schema)

        assert "You are a SQL expert" in prompt
        assert question in prompt
        assert schema in prompt

    @patch("core.llm_sql_generator.pipeline")
    def test_generate_sql_pipeline(self, mock_pipeline):
        """Test SQL generation using pipeline API."""
        # Mock pipeline response
        mock_model = MagicMock()
        mock_pipeline.return_value = mock_model
        mock_model.return_value = [{"generated_text": "Input text SELECT * FROM users"}]

        # Create generator with mocked pipeline
        generator = HuggingFaceSQLGenerator(use_pipeline=True)

        # Test SQL generation
        sql = generator.generate_sql(
            question="Find all users", max_tokens=100, temperature=0.5
        )

        assert sql == "SELECT * FROM users"
        mock_pipeline.assert_called_once()

    @patch("core.llm_sql_generator.AutoTokenizer")
    @patch("core.llm_sql_generator.AutoModelForCausalLM")
    def test_generate_sql_automodel(self, mock_model_class, mock_tokenizer_class):
        """Test SQL generation using AutoModel API."""
        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        # Mock model
        mock_model = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model

        # Mock generate function
        mock_model.generate.return_value = [[1, 2, 3]]  # Token IDs

        # Mock decode
        mock_tokenizer.decode.return_value = "Input prompt SELECT id, name FROM users"

        # Create generator with mocked components
        generator = HuggingFaceSQLGenerator(use_pipeline=False)

        # Test SQL generation
        sql = generator.generate_sql(
            question="Find all users", max_tokens=100, temperature=0.5
        )

        assert sql == "SELECT id, name FROM users"
        mock_model_class.from_pretrained.assert_called_once()
        mock_tokenizer_class.from_pretrained.assert_called_once()

    def test_error_handling(self):
        """Test error handling in SQL generation."""
        generator = HuggingFaceSQLGenerator(use_pipeline=False)

        # Replace _generate_text with a function that raises an exception
        def mock_generate_text(*args, **kwargs):
            raise ValueError("Test error")

        generator._generate_text = mock_generate_text

        # Test SQL generation with error
        sql = generator.generate_sql(question="Find all users")

        assert "Error generating SQL" in sql

    @patch("core.llm_sql_generator.HuggingFaceSQLGenerator")
    def test_generate_sql_from_llm_function(self, mock_generator_class):
        """Test the generate_sql_from_llm convenience function."""
        # Mock the generator instance
        mock_generator = MagicMock()
        mock_generator_class.return_value = mock_generator

        # Mock generate_sql method
        mock_generator.generate_sql.return_value = "SELECT * FROM products"

        # Call the function
        result = generate_sql_from_llm(
            question="What products are available?",
            schema_context="Table: products\nColumns: id, name, price",
            model_name="test-model",
            temperature=0.3,
        )

        # Verify results
        assert result == "SELECT * FROM products"
        mock_generator_class.assert_called_once()
        mock_generator.generate_sql.assert_called_once()


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
