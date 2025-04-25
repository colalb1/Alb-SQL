"""
Unit tests for error analysis utilities.

This module tests the error analysis utilities used to analyze and log errors in SQL
generation.
"""

import json
import os
import tempfile

import pytest

from tests.utils.error_analysis import (
    BatchErrorAnalyzer,
    ErrorAnalyzer,
    compare_sql_execution_results,
)


class TestErrorAnalysis:
    """Tests for the error analysis utilities."""

    def test_error_analyzer_initialization(self):
        """Test that the ErrorAnalyzer initializes correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            analyzer = ErrorAnalyzer(log_dir=temp_dir)

            assert os.path.exists(temp_dir)
            assert analyzer.log_dir == temp_dir
            assert analyzer.error_stats["total_samples"] == 0
            assert analyzer.error_stats["total_errors"] == 0
            assert analyzer.error_stats["error_categories"] == {}

    def test_analyze_error(self):
        """Test analyzing errors between generated and gold SQL."""
        with tempfile.TemporaryDirectory() as temp_dir:
            analyzer = ErrorAnalyzer(log_dir=temp_dir)

            # Test with matching SQL
            query_text = "List all users"
            generated_sql = "SELECT * FROM users"
            gold_sql = "SELECT * FROM users"

            analysis = analyzer.analyze_error(query_text, generated_sql, gold_sql)

            assert analysis["query_text"] == query_text
            assert analysis["generated_sql"] == generated_sql
            assert analysis["gold_sql"] == gold_sql
            assert analysis["similarity_score"] > 0.9  # Should be very similar
            assert len(analysis["error_categories"]) == 0  # No errors expected

            # Test with differing SQL
            generated_sql = "SELECT id, name FROM users"
            gold_sql = "SELECT * FROM users WHERE age > 18"

            analysis = analyzer.analyze_error(query_text, generated_sql, gold_sql)

            assert analysis["similarity_score"] < 0.9  # Should be less similar
            assert len(analysis["error_categories"]) > 0  # Should have error categories
            assert "different_operations" in analysis
            assert "explanation" in analysis and len(analysis["explanation"]) > 0

    def test_categorize_error(self):
        """Test error categorization."""
        analyzer = ErrorAnalyzer()

        # Test table mismatch
        differences = {
            "similarity_score": 0.7,
            "different_tables": ["users", "customers"],
            "different_columns": [],
            "different_operations": [],
            "aliasing_differences": {"different_aliases": {}},
        }

        categories = analyzer._categorize_error(differences)
        assert "table_mismatch" in categories

        # Test column mismatch
        differences = {
            "similarity_score": 0.8,
            "different_tables": [],
            "different_columns": ["name", "email"],
            "different_operations": [],
            "aliasing_differences": {"different_aliases": {}},
        }

        categories = analyzer._categorize_error(differences)
        assert "column_mismatch" in categories

        # Test operation mismatch
        differences = {
            "similarity_score": 0.7,
            "different_tables": [],
            "different_columns": [],
            "different_operations": ["where", "group_by"],
            "aliasing_differences": {"different_aliases": {}},
        }

        categories = analyzer._categorize_error(differences)
        assert "filter_error" in categories
        assert "grouping_error" in categories

        # Test major structural error
        differences = {
            "similarity_score": 0.5,
            "different_tables": [],
            "different_columns": [],
            "different_operations": [],
            "aliasing_differences": {"different_aliases": {}},
        }

        categories = analyzer._categorize_error(differences)
        assert "major_structural_error" in categories

    def test_log_error(self):
        """Test logging errors to files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            analyzer = ErrorAnalyzer(log_dir=temp_dir)

            # Create sample error analysis
            analysis = {
                "query_text": "List all users",
                "generated_sql": "SELECT id, name FROM users",
                "gold_sql": "SELECT * FROM users",
                "similarity_score": 0.7,
                "error_categories": ["column_mismatch"],
                "different_tables": [],
                "different_columns": ["*", "id", "name"],
                "different_operations": [],
                "aliasing_differences": {"different_aliases": {}},
                "explanation": "Column mismatch detected",
                "metadata": {"sample_id": "test_1", "timestamp": "2025-01-01"},
            }

            # Log the error
            log_file = analyzer.log_error(analysis)

            # Check that the file was created
            assert os.path.exists(log_file)

            # Check file contents
            with open(log_file, "r") as f:
                logged_data = json.load(f)
                assert logged_data["query_text"] == analysis["query_text"]
                assert logged_data["error_categories"] == analysis["error_categories"]

    def test_get_error_statistics(self):
        """Test getting error statistics."""
        analyzer = ErrorAnalyzer()

        # Analyze some errors
        analyzer.analyze_error(
            "List all users",
            "SELECT id, name FROM users",
            "SELECT * FROM users",
            metadata={"complexity": "simple"},
        )

        analyzer.analyze_error(
            "Find users who ordered products last month",
            "SELECT u.id FROM users u JOIN orders o ON u.id = o.user_id",
            "SELECT u.id FROM users u JOIN orders o ON u.id = o.user_id WHERE o.order_date >= DATE_SUB(CURDATE(), INTERVAL 1 MONTH)",
            metadata={"complexity": "moderate"},
        )

        # Get statistics
        stats = analyzer.get_error_statistics()

        assert stats["total_samples"] == 2
        assert "total_errors" in stats
        assert "error_rate" in stats
        assert "error_categories" in stats
        assert "error_by_complexity" in stats
        assert "simple" in stats["error_by_complexity"]
        assert "moderate" in stats["error_by_complexity"]

    def test_batch_error_analyzer(self):
        """Test batch error analysis."""
        with tempfile.TemporaryDirectory() as temp_dir:
            batch_analyzer = BatchErrorAnalyzer(log_dir=temp_dir)

            # Create sample batch results
            batch_results = [
                {
                    "query_text": "List all users",
                    "generated_sql": "SELECT * FROM users",
                    "gold_sql": "SELECT * FROM users",
                    "timestamp": "2025-01-01",
                    "complexity": "simple",
                },
                {
                    "query_text": "Find users with email",
                    "generated_sql": "SELECT id, name FROM users WHERE email IS NOT NULL",
                    "gold_sql": "SELECT * FROM users WHERE email IS NOT NULL",
                    "timestamp": "2025-01-01",
                    "complexity": "simple",
                },
                {
                    "query_text": "Count orders per user",
                    "generated_sql": "SELECT user_id, COUNT(*) FROM orders GROUP BY user_id",
                    "gold_sql": "SELECT u.name, COUNT(o.id) FROM users u JOIN orders o ON u.id = o.user_id GROUP BY u.id, u.name",
                    "timestamp": "2025-01-01",
                    "complexity": "moderate",
                },
            ]

            # Analyze batch
            batch_analysis = batch_analyzer.analyze_batch(batch_results, threshold=0.8)

            assert batch_analysis["total_samples"] == 3
            assert "error_samples" in batch_analysis
            assert "error_logs" in batch_analysis
            assert "error_categories" in batch_analysis
            assert "average_similarity" in batch_analysis

            # Check that error logs were created
            for log_file in batch_analysis["error_logs"]:
                assert os.path.exists(log_file)

    def test_compare_sql_execution_results(self):
        """Test comparing SQL execution results."""
        # Test matching results
        result1 = {
            "success": True,
            "row_count": 2,
            "results": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}],
        }

        result2 = {
            "success": True,
            "row_count": 2,
            "results": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}],
        }

        comparison = compare_sql_execution_results(result1, result2)
        assert comparison["match"] is True

        # Test different column sets
        result3 = {
            "success": True,
            "row_count": 2,
            "results": [
                {"id": 1, "name": "Alice", "email": "alice@example.com"},
                {"id": 2, "name": "Bob", "email": "bob@example.com"},
            ],
        }

        comparison = compare_sql_execution_results(result1, result3)
        assert comparison["match"] is False
        assert comparison["reason"] == "Column set mismatch"

        # Test different row counts
        result4 = {
            "success": True,
            "row_count": 3,
            "results": [
                {"id": 1, "name": "Alice"},
                {"id": 2, "name": "Bob"},
                {"id": 3, "name": "Charlie"},
            ],
        }

        comparison = compare_sql_execution_results(result1, result4)
        assert comparison["match"] is False
        assert comparison["reason"] == "Row count mismatch"

        # Test execution failure
        result5 = {"success": False, "error": "Syntax error"}

        comparison = compare_sql_execution_results(result1, result5)
        assert comparison["match"] is False
        assert comparison["reason"] == "Execution failure"


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
