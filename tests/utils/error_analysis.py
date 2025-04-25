"""
Error Analysis Utilities

This module provides utilities for analyzing and logging errors in SQL generation,
including explanations of incorrect outputs.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from tests.utils.sql_comparison import analyze_sql_differences, normalize_sql

# Configure logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class ErrorAnalyzer:
    """
    A class for analyzing and logging errors in SQL generation.
    """

    def __init__(self, log_dir: str = "error_logs"):
        """
        Initialize the ErrorAnalyzer.

        Args:
            log_dir (str): Directory to store error logs.
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # Initialize error statistics
        self.error_stats = {
            "total_samples": 0,
            "total_errors": 0,
            "error_categories": {},
            "error_by_complexity": {
                "simple": {"count": 0, "total": 0},
                "moderate": {"count": 0, "total": 0},
                "complex": {"count": 0, "total": 0},
            },
        }

    def analyze_error(
        self,
        query_text: str,
        generated_sql: str,
        gold_sql: str,
        db_schema: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze differences between generated SQL and gold SQL.

        Args:
            query_text (str): Original natural language query.
            generated_sql (str): Generated SQL query.
            gold_sql (str): Gold/reference SQL query.
            db_schema (Optional[Dict[str, Any]]): Database schema for additional context.
            metadata (Optional[Dict[str, Any]]): Additional metadata about the sample.

        Returns:
            Dictionary with error analysis.
        """
        self.error_stats["total_samples"] += 1

        # Basic normalization
        normalized_generated = normalize_sql(generated_sql)
        normalized_gold = normalize_sql(gold_sql)

        # Get detailed differences
        differences = analyze_sql_differences(generated_sql, gold_sql)

        # Determine error categories
        error_categories = self._categorize_error(differences)

        # Increment error count if categories found
        if error_categories:
            self.error_stats["total_errors"] += 1

            # Track error categories
            for category in error_categories:
                if category not in self.error_stats["error_categories"]:
                    self.error_stats["error_categories"][category] = 0
                self.error_stats["error_categories"][category] += 1

            # Track by complexity if available
            if metadata and "complexity" in metadata:
                complexity = metadata["complexity"]
                if complexity in self.error_stats["error_by_complexity"]:
                    self.error_stats["error_by_complexity"][complexity]["total"] += 1
                    self.error_stats["error_by_complexity"][complexity]["count"] += 1

        # Structured error analysis
        analysis = {
            "query_text": query_text,
            "generated_sql": generated_sql,
            "gold_sql": gold_sql,
            "normalized_generated": normalized_generated,
            "normalized_gold": normalized_gold,
            "similarity_score": differences["similarity_score"],
            "different_tables": differences["different_tables"],
            "different_columns": differences["different_columns"],
            "different_operations": differences["different_operations"],
            "aliasing_differences": differences["aliasing_differences"],
            "error_categories": error_categories,
            "explanation": self._generate_explanation(differences, error_categories),
            "metadata": metadata or {},
        }

        return analysis

    def _categorize_error(self, differences: Dict[str, Any]) -> List[str]:
        """
        Categorize the error based on the differences.

        Args:
            differences (Dict[str, Any]): Differences between SQL queries.

        Returns:
            List of error categories.
        """
        error_categories = []

        # Missing or extra tables
        if differences["different_tables"]:
            error_categories.append("table_mismatch")

        # Missing or extra columns
        if differences["different_columns"]:
            error_categories.append("column_mismatch")

        # Missing operations
        for op in differences["different_operations"]:
            if op == "join":
                error_categories.append("join_error")
            elif op == "where":
                error_categories.append("filter_error")
            elif op == "group_by":
                error_categories.append("grouping_error")
            elif op == "order_by":
                error_categories.append("ordering_error")
            elif op == "limit":
                error_categories.append("limit_error")
            elif op in ["union", "intersect", "except"]:
                error_categories.append("set_operation_error")
            elif op == "subquery":
                error_categories.append("subquery_error")

        # Check aliasing differences
        if differences["aliasing_differences"]["different_aliases"]:
            error_categories.append("aliasing_error")

        # Check similarity score
        if differences["similarity_score"] < 0.6:
            error_categories.append("major_structural_error")
        elif differences["similarity_score"] < 0.8:
            error_categories.append("minor_structural_error")

        return error_categories

    def _generate_explanation(
        self, differences: Dict[str, Any], error_categories: List[str]
    ) -> str:
        """
        Generate a human-readable explanation of the error.

        Args:
            differences (Dict[str, Any]): Differences between SQL queries.
            error_categories (List[str]): Categorized errors.

        Returns:
            Explanation string.
        """
        explanations = []

        if "table_mismatch" in error_categories:
            tables = differences["different_tables"]
            explanations.append(
                f"Table mismatch: The SQL queries use different tables: {', '.join(tables)}."
            )

        if "column_mismatch" in error_categories:
            columns = differences["different_columns"]
            explanations.append(
                f"Column mismatch: The SQL queries reference different columns: {', '.join(columns)}."
            )

        if "join_error" in error_categories:
            explanations.append(
                "Join operation error: The SQL queries have different JOIN operations."
            )

        if "filter_error" in error_categories:
            explanations.append(
                "Filter error: The SQL queries have different WHERE clauses."
            )

        if "grouping_error" in error_categories:
            explanations.append(
                "Grouping error: The SQL queries have different GROUP BY clauses."
            )

        if "ordering_error" in error_categories:
            explanations.append(
                "Ordering error: The SQL queries have different ORDER BY clauses."
            )

        if "subquery_error" in error_categories:
            explanations.append(
                "Subquery error: The SQL queries have different subquery structures."
            )

        if "aliasing_error" in error_categories:
            aliases = differences["aliasing_differences"]["different_aliases"]
            explanations.append(
                "Aliasing error: The SQL queries use different aliases for similar elements."
            )

        # Add similarity score
        explanations.append(
            f"Overall similarity score: {differences['similarity_score']:.2f} out of 1.0"
        )

        return "\n".join(explanations)

    def log_error(
        self, error_analysis: Dict[str, Any], file_prefix: str = "error"
    ) -> str:
        """
        Log error analysis to a file.

        Args:
            error_analysis (Dict[str, Any]): Error analysis from analyze_error.
            file_prefix (str): Prefix for the log file name.

        Returns:
            Path to the log file.
        """
        # Create a unique file name
        timestamp = error_analysis.get("metadata", {}).get("timestamp", "unknown")
        sample_id = error_analysis.get("metadata", {}).get("sample_id", "unknown")
        log_file = os.path.join(
            self.log_dir, f"{file_prefix}_{sample_id}_{timestamp}.json"
        )

        # Write the error analysis to the file
        with open(log_file, "w") as f:
            json.dump(error_analysis, f, indent=2)

        # Log basic information
        logger.info(f"Logged error analysis to {log_file}")
        logger.info(f"Query: {error_analysis['query_text']}")
        logger.info(f"Error categories: {error_analysis['error_categories']}")
        logger.info(f"Similarity score: {error_analysis['similarity_score']:.2f}")

        return log_file

    def get_error_statistics(self) -> Dict[str, Any]:
        """
        Get error statistics.

        Returns:
            Dictionary with error statistics.
        """
        # Calculate error rate
        if self.error_stats["total_samples"] > 0:
            error_rate = (
                self.error_stats["total_errors"] / self.error_stats["total_samples"]
            )
        else:
            error_rate = 0.0

        # Calculate error rates by complexity
        complexity_rates = {}
        for complexity, stats in self.error_stats["error_by_complexity"].items():
            if stats["total"] > 0:
                complexity_rates[complexity] = stats["count"] / stats["total"]
            else:
                complexity_rates[complexity] = 0.0

        # Prepare statistics
        statistics = {
            "total_samples": self.error_stats["total_samples"],
            "total_errors": self.error_stats["total_errors"],
            "error_rate": error_rate,
            "error_categories": self.error_stats["error_categories"],
            "error_by_complexity": {
                complexity: {
                    "rate": rate,
                    "count": self.error_stats["error_by_complexity"][complexity][
                        "count"
                    ],
                }
                for complexity, rate in complexity_rates.items()
            },
        }

        return statistics

    def save_error_statistics(self, file_path: str) -> None:
        """
        Save error statistics to a file.

        Args:
            file_path (str): Path to save the statistics.
        """
        statistics = self.get_error_statistics()

        with open(file_path, "w") as f:
            json.dump(statistics, f, indent=2)

        logger.info(f"Saved error statistics to {file_path}")


class BatchErrorAnalyzer:
    """
    A class for analyzing and logging errors in batches of SQL generation results.
    """

    def __init__(self, log_dir: str = "error_logs"):
        """
        Initialize the BatchErrorAnalyzer.

        Args:
            log_dir (str): Directory to store error logs.
        """
        self.error_analyzer = ErrorAnalyzer(log_dir)
        self.log_dir = log_dir
        self.batch_results = []

    def analyze_batch(
        self, batch_results: List[Dict[str, Any]], threshold: float = 0.9
    ) -> Dict[str, Any]:
        """
        Analyze errors in a batch of results.

        Args:
            batch_results (List[Dict[str, Any]]): List of results with query_text,
                                                 generated_sql, gold_sql.
            threshold (float): Similarity threshold below which to log errors.

        Returns:
            Dictionary with batch analysis results.
        """
        batch_analysis = {
            "total_samples": len(batch_results),
            "error_samples": 0,
            "error_logs": [],
            "error_categories": {},
            "average_similarity": 0.0,
        }

        total_similarity = 0.0
        for i, result in enumerate(batch_results):
            # Extract data
            query_text = result["query_text"]
            generated_sql = result["generated_sql"]
            gold_sql = result["gold_sql"]
            db_schema = result.get("db_schema")
            metadata = {
                "sample_id": i,
                "timestamp": result.get("timestamp", "unknown"),
                "complexity": result.get("complexity", "unknown"),
                "db_id": result.get("db_id", "unknown"),
            }

            # Analyze error
            error_analysis = self.error_analyzer.analyze_error(
                query_text, generated_sql, gold_sql, db_schema, metadata
            )

            # Add to total similarity
            total_similarity += error_analysis["similarity_score"]

            # Log error if below threshold
            if error_analysis["similarity_score"] < threshold:
                batch_analysis["error_samples"] += 1
                log_file = self.error_analyzer.log_error(error_analysis, f"batch_{i}")
                batch_analysis["error_logs"].append(log_file)

                # Update error categories
                for category in error_analysis["error_categories"]:
                    if category not in batch_analysis["error_categories"]:
                        batch_analysis["error_categories"][category] = 0
                    batch_analysis["error_categories"][category] += 1

        # Calculate average similarity
        if batch_results:
            batch_analysis["average_similarity"] = total_similarity / len(batch_results)

        # Save batch analysis
        timestamp = (
            batch_results[0].get("timestamp", "unknown") if batch_results else "unknown"
        )
        batch_file = os.path.join(self.log_dir, f"batch_analysis_{timestamp}.json")
        with open(batch_file, "w") as f:
            json.dump(batch_analysis, f, indent=2)

        logger.info(
            f"Batch analysis: {batch_analysis['error_samples']} errors out of {batch_analysis['total_samples']} samples"
        )
        logger.info(
            f"Average similarity score: {batch_analysis['average_similarity']:.2f}"
        )
        logger.info(f"Saved batch analysis to {batch_file}")

        return batch_analysis

    def get_error_statistics(self) -> Dict[str, Any]:
        """
        Get error statistics from the underlying error analyzer.

        Returns:
            Dictionary with error statistics.
        """
        return self.error_analyzer.get_error_statistics()


def compare_sql_execution_results(
    result1: Dict[str, Any], result2: Dict[str, Any], tolerance: float = 0.0001
) -> Dict[str, Any]:
    """
    Compare the execution results of two SQL queries.

    Args:
        result1 (Dict[str, Any]): First execution result.
        result2 (Dict[str, Any]): Second execution result.
        tolerance (float): Tolerance for numeric comparisons.

    Returns:
        Dictionary with comparison results.
    """
    # Check if both executions were successful
    if not result1.get("success") or not result2.get("success"):
        return {
            "match": False,
            "reason": "Execution failure",
            "details": {
                "result1_success": result1.get("success", False),
                "result2_success": result2.get("success", False),
                "result1_error": result1.get("error"),
                "result2_error": result2.get("error"),
            },
        }

    # Check row counts
    row_count1 = result1.get("row_count", 0)
    row_count2 = result2.get("row_count", 0)

    if row_count1 != row_count2:
        return {
            "match": False,
            "reason": "Row count mismatch",
            "details": {
                "result1_row_count": row_count1,
                "result2_row_count": row_count2,
            },
        }

    # If no rows returned by both queries, they match
    if row_count1 == 0:
        return {"match": True, "reason": "Both queries returned no rows", "details": {}}

    # Get the result sets
    rows1 = result1.get("results", [])
    rows2 = result2.get("results", [])

    # If column sets don't match, they're different
    if rows1 and rows2:
        columns1 = set(rows1[0].keys())
        columns2 = set(rows2[0].keys())

        if columns1 != columns2:
            return {
                "match": False,
                "reason": "Column set mismatch",
                "details": {
                    "result1_columns": list(columns1),
                    "result2_columns": list(columns2),
                },
            }

    # If row counts are the same but one is empty and the other isn't
    if (not rows1 and rows2) or (rows1 and not rows2):
        return {
            "match": False,
            "reason": "One result set is empty, the other is not",
            "details": {
                "result1_empty": len(rows1) == 0,
                "result2_empty": len(rows2) == 0,
            },
        }

    # Compare actual data
    # This is a simplified approach; in practice, you might need more sophisticated comparison
    # such as comparing sorted results, handling NULLs, etc.

    # Convert rows to a comparable format (sorted by all values)
    def prepare_row(row):
        # Convert all values to strings for comparison
        return {k: str(v) if v is not None else "NULL" for k, v in row.items()}

    sorted_rows1 = [prepare_row(row) for row in rows1]
    sorted_rows2 = [prepare_row(row) for row in rows2]

    # Sort rows by their string representation
    sorted_rows1.sort(key=lambda x: str(x))
    sorted_rows2.sort(key=lambda x: str(x))

    # Compare sorted rows
    if sorted_rows1 != sorted_rows2:
        # Find the first differing row
        first_diff_index = None
        for i in range(min(len(sorted_rows1), len(sorted_rows2))):
            if sorted_rows1[i] != sorted_rows2[i]:
                first_diff_index = i
                break

        return {
            "match": False,
            "reason": "Data mismatch",
            "details": {
                "first_different_row_index": first_diff_index,
                "result1_row": sorted_rows1[first_diff_index]
                if first_diff_index is not None
                else None,
                "result2_row": sorted_rows2[first_diff_index]
                if first_diff_index is not None
                else None,
            },
        }

    # If we get here, the results match
    return {
        "match": True,
        "reason": "Results match exactly",
        "details": {
            "row_count": row_count1,
        },
    }


if __name__ == "__main__":
    # Example usage
    print("Error analysis utilities for SQL generation")
