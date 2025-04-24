"""
Execution-Aware Trainer Module

This module implements execution-aware training and validation for SQL generation,
ensuring that the generated SQL queries are not only syntactically correct but also
produce the expected results when executed against the database.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class QueryExecutionResult:
    """
    Data class for SQL query execution results.
    """

    success: bool
    rows: Optional[List[Dict[str, Any]]] = None
    row_count: int = 0
    execution_time: float = 0.0
    error_message: Optional[str] = None
    execution_plan: Optional[Dict[str, Any]] = None


@dataclass
class QueryEvalMetrics:
    """
    Data class for query evaluation metrics.
    """

    execution_success: bool
    syntax_correctness: float
    semantic_correctness: float
    result_match_score: float
    execution_efficiency: float
    overall_score: float


class ExecutionAwareTrainer:
    """
    A trainer that incorporates SQL execution results into the training loop
    to optimize for both query correctness and efficiency.
    """

    def __init__(
        self,
        db_connector=None,
        execution_timeout: int = 30,
        max_rows_to_compare: int = 1000,
        plan_weight: float = 0.3,
        result_weight: float = 0.7,
    ):
        """
        Initialize the ExecutionAwareTrainer.

        Args:
            db_connector: Connector to execute SQL against databases.
            execution_timeout (int): Maximum time (in seconds) to wait for query execution.
            max_rows_to_compare (int): Maximum number of rows to compare when evaluating results.
            plan_weight (float): Weight of the execution plan similarity in the loss function.
            result_weight (float): Weight of the result similarity in the loss function.
        """
        self.db_connector = db_connector
        self.execution_timeout = execution_timeout
        self.max_rows_to_compare = max_rows_to_compare
        self.plan_weight = plan_weight
        self.result_weight = result_weight
        self.query_history: Dict[str, List[Tuple[str, QueryExecutionResult]]] = {}

    def execute_query(
        self, db_name: str, query: str, get_execution_plan: bool = True
    ) -> QueryExecutionResult:
        """
        Execute an SQL query and return the results.

        Args:
            db_name (str): Database name to execute against.
            query (str): SQL query to execute.
            get_execution_plan (bool): Whether to return the execution plan.

        Returns:
            QueryExecutionResult containing execution results.
        """
        if self.db_connector is None:
            logger.warning("No database connector provided. Cannot execute query.")
            return QueryExecutionResult(
                success=False, error_message="No database connector"
            )

        start_time = time.time()
        try:
            # This would use the actual database connector
            # For now, we'll simulate execution results
            success = True
            rows = [{"id": 1, "name": "Sample"} for _ in range(10)]
            row_count = len(rows)
            execution_plan = (
                {"plan_type": "sequential_scan", "cost": 100}
                if get_execution_plan
                else None
            )
            error_message = None
        except Exception as e:
            success = False
            rows = None
            row_count = 0
            execution_plan = None
            error_message = str(e)
        finally:
            execution_time = time.time() - start_time

        result = QueryExecutionResult(
            success=success,
            rows=rows,
            row_count=row_count,
            execution_time=execution_time,
            error_message=error_message,
            execution_plan=execution_plan,
        )

        # Save query and result to history
        if db_name not in self.query_history:
            self.query_history[db_name] = []
        self.query_history[db_name].append((query, result))

        return result

    def get_execution_plan(self, db_name: str, query: str) -> Optional[Dict[str, Any]]:
        """
        Get the execution plan for an SQL query without executing it.

        Args:
            db_name (str): Database name.
            query (str): SQL query.

        Returns:
            Execution plan or None if not available.
        """
        if self.db_connector is None:
            logger.warning("No database connector provided. Cannot get execution plan.")
            return None

        try:
            # This would use the actual database connector to get the execution plan
            # For now, we'll simulate an execution plan
            execution_plan = {
                "plan_type": "sequential_scan",
                "cost": 100,
                "estimated_rows": 10,
            }
            return execution_plan
        except Exception as e:
            logger.error(f"Error getting execution plan: {e}")
            return None

    def compare_results(
        self, result1: List[Dict[str, Any]], result2: List[Dict[str, Any]]
    ) -> float:
        """
        Compare two query result sets and return a similarity score.

        Args:
            result1 (List[Dict[str, Any]]): First result set (list of row dictionaries).
            result2 (List[Dict[str, Any]]): Second result set (list of row dictionaries).

        Returns:
            Similarity score between 0 (completely different) and 1 (identical).
        """
        if result1 is None or result2 is None:
            return 0.0

        # For simple case: check if row counts match
        if len(result1) != len(result2):
            # Partial score based on relative difference in row counts
            max_rows = max(len(result1), len(result2))
            min_rows = min(len(result1), len(result2))
            return min_rows / max_rows if max_rows > 0 else 0.0

        # Limit rows to compare for efficiency
        result1 = result1[: self.max_rows_to_compare]
        result2 = result2[: self.max_rows_to_compare]

        # Check if all rows match exactly (order-sensitive)
        # In a real implementation, we might want to sort and compare or use a more sophisticated comparison
        matching_rows = 0
        for i in range(min(len(result1), len(result2))):
            if result1[i] == result2[i]:
                matching_rows += 1

        return matching_rows / len(result1) if result1 else 0.0

    def compare_execution_plans(
        self, plan1: Dict[str, Any], plan2: Dict[str, Any]
    ) -> float:
        """
        Compare two execution plans and return a similarity score.

        Args:
            plan1 (Dict[str, Any]): First execution plan.
            plan2 (Dict[str, Any]): Second execution plan.

        Returns:
            Similarity score between 0 (completely different) and 1 (identical).
        """
        if plan1 is None or plan2 is None:
            return 0.0

        # Simple comparison based on plan type
        if plan1.get("plan_type") == plan2.get("plan_type"):
            # If plan types match, compare costs
            cost1 = plan1.get("cost", 0)
            cost2 = plan2.get("cost", 0)

            # Calculate cost ratio (higher is better)
            if cost1 <= 0 or cost2 <= 0:
                cost_similarity = 0.0
            else:
                min_cost = min(cost1, cost2)
                max_cost = max(cost1, cost2)
                cost_similarity = min_cost / max_cost

            return 0.7 + (
                0.3 * cost_similarity
            )  # 70% for matching plan type, 30% for cost
        else:
            return 0.0  # Different plan types

    def sql_correctness_loss(
        self, generated_sql: str, reference_sql: str, db_name: str
    ) -> float:
        """
        Calculate a loss value based on SQL correctness.

        Args:
            generated_sql (str): Generated SQL query.
            reference_sql (str): Reference (correct) SQL query.
            db_name (str): Database name to execute against.

        Returns:
            Loss value (lower is better).
        """
        generated_result = self.execute_query(db_name, generated_sql)
        reference_result = self.execute_query(db_name, reference_sql)

        if not generated_result.success:
            # If generated SQL fails to execute, high loss
            return 10.0

        # Compare results
        result_similarity = self.compare_results(
            generated_result.rows, reference_result.rows
        )

        # Compare execution plans
        plan_similarity = self.compare_execution_plans(
            generated_result.execution_plan, reference_result.execution_plan
        )

        # Weighted sum of result and plan similarity
        similarity = (
            self.result_weight * result_similarity + self.plan_weight * plan_similarity
        )

        # Convert similarity to loss (higher similarity = lower loss)
        loss = 1.0 - similarity

        return loss

    def evaluate_query(
        self, generated_sql: str, reference_sql: str, db_name: str
    ) -> QueryEvalMetrics:
        """
        Evaluate a generated SQL query against a reference query.

        Args:
            generated_sql (str): Generated SQL query to evaluate.
            reference_sql (str): Reference (correct) SQL query.
            db_name (str): Database name to execute against.

        Returns:
            QueryEvalMetrics with detailed evaluation metrics.
        """
        generated_result = self.execute_query(db_name, generated_sql)
        reference_result = self.execute_query(db_name, reference_sql)

        # Basic metrics
        execution_success = generated_result.success

        # Check syntax correctness (simplified)
        syntax_correctness = 1.0 if execution_success else 0.0

        # Compare results
        result_match_score = (
            self.compare_results(generated_result.rows, reference_result.rows)
            if execution_success
            else 0.0
        )

        # Semantic correctness - this is a simplified version
        # In practice, we'd check for correct table usage, joins, and conditions
        semantic_correctness = result_match_score

        # Execution efficiency
        execution_efficiency = 0.0
        if execution_success and reference_result.execution_time > 0:
            # Lower ratio is better (generated query executes faster)
            time_ratio = (
                generated_result.execution_time / reference_result.execution_time
            )

            # Convert to a 0-1 score where 1 is better
            # time_ratio of 0.5 (twice as fast) would give 1.0
            # time_ratio of 1.0 (same speed) would give 0.5
            # time_ratio of 2.0 (twice as slow) would give 0.25
            execution_efficiency = min(1.0, 1.0 / (1.0 + time_ratio))

        # Overall score - weighted average of all metrics
        overall_score = 0.0
        if execution_success:
            weights = {
                "syntax": 0.1,
                "semantic": 0.4,
                "result_match": 0.4,
                "efficiency": 0.1,
            }
            overall_score = (
                weights["syntax"] * syntax_correctness
                + weights["semantic"] * semantic_correctness
                + weights["result_match"] * result_match_score
                + weights["efficiency"] * execution_efficiency
            )

        return QueryEvalMetrics(
            execution_success=execution_success,
            syntax_correctness=syntax_correctness,
            semantic_correctness=semantic_correctness,
            result_match_score=result_match_score,
            execution_efficiency=execution_efficiency,
            overall_score=overall_score,
        )

    def generate_schema_aware_prompt(
        self,
        db_name: str,
        target_tables: List[str],
        question: str,
        include_sample_data: bool = True,
        complexity_estimate: str = "medium",
    ) -> str:
        """
        Generate a schema-aware prompt for SQL generation.

        Args:
            db_name (str): Database name.
            target_tables (List[str]): List of tables relevant to the query.
            question (str): Natural language question to translate to SQL.
            include_sample_data (bool): Whether to include sample data in the prompt.
            complexity_estimate (str): Estimated complexity of the query.

        Returns:
            Formatted prompt for LLM.
        """
        # This would access actual schema information
        # For now, we'll use a template with placeholders

        # Build adaptive schema summary based on complexity
        if complexity_estimate == "simple":
            token_allocation = "brief"
            tables_to_include = target_tables[:2]  # Limit tables for simple queries
        elif complexity_estimate == "medium":
            token_allocation = "moderate"
            tables_to_include = target_tables
        else:  # complex
            token_allocation = "detailed"
            tables_to_include = target_tables + self._get_related_tables(
                db_name, target_tables
            )

        schema_summary = self._generate_schema_summary(
            db_name, tables_to_include, token_allocation, include_sample_data
        )

        # Get common mistakes for these tables/DB
        common_mistakes = self._get_common_mistakes(db_name, tables_to_include)

        # Generate prompt
        prompt = f"""
**Role**: World-class SQL Engineer + Database Architect
**Task**: Solve {db_name} problem using {db_name}'s schema

**Question**:
{question}

**Schema Context**:
{schema_summary}

**Common Mistakes**:
{common_mistakes}

**Reasoning Chain**:
[Tables] → [Filters] → [Aggregation]

**Response Format**:
```sql
/* Explanatory comments */
SELECT ...
```
"""
        return prompt

    def _generate_schema_summary(
        self,
        db_name: str,
        tables: List[str],
        detail_level: str,
        include_sample_data: bool,
    ) -> str:
        """
        Generate a schema summary for the prompt.

        Args:
            db_name (str): Database name.
            tables (List[str]): List of tables to include.
            detail_level (str): Level of detail to include ('brief', 'moderate', 'detailed').
            include_sample_data (bool): Whether to include sample data.

        Returns:
            Schema summary string.
        """
        # This would access actual schema information
        # For now, we'll use a template

        summary = "Tables:\n"

        for table in tables:
            summary += f"- {table}\n"

            # Add column details based on detail level
            if detail_level == "brief":
                summary += "  Columns: id, name, ...\n"
            else:
                # For moderate and detailed, include more column info
                summary += "  Columns:\n"
                summary += "    id (INTEGER, PK)\n"
                summary += "    name (VARCHAR)\n"
                summary += "    created_at (TIMESTAMP)\n"

                if detail_level == "detailed":
                    # For detailed, include constraints and relationships
                    summary += "  Constraints:\n"
                    summary += "    - PK on id\n"
                    summary += f"    - FK from other_table.{table}_id to {table}.id\n"

            # Include sample data if requested
            if include_sample_data and detail_level != "brief":
                summary += "  Sample Data:\n"
                summary += "    (1, 'Example', '2023-01-01')\n"
                summary += "    (2, 'Sample', '2023-01-02')\n"

        if detail_level == "detailed":
            # Add relationship graph for detailed view
            summary += "\nRelationships:\n"
            summary += "users -< posts (users.id = posts.user_id)\n"
            summary += "posts -< comments (posts.id = comments.post_id)\n"

        return summary

    def _get_common_mistakes(self, db_name: str, tables: List[str]) -> str:
        """
        Get common mistakes for the given database and tables.

        Args:
            db_name (str): Database name.
            tables (List[str]): List of tables.

        Returns:
            String describing common mistakes.
        """
        # This would query a knowledge base of common mistakes
        # For now, we'll use a template

        mistakes = "1. Remember to use appropriate JOIN conditions\n"
        mistakes += "2. Be careful with NULL handling in WHERE clauses\n"

        # Add specific mistakes for given tables
        for table in tables:
            mistakes += f"3. For {table} table, check for valid date formats\n"
            break  # Just add one example

        return mistakes

    def _get_related_tables(self, db_name: str, tables: List[str]) -> List[str]:
        """
        Get related tables based on foreign key relationships.

        Args:
            db_name (str): Database name.
            tables (List[str]): List of primary tables.

        Returns:
            List of related tables.
        """
        # This would query the actual database schema
        # For now, we'll return some placeholder related tables

        related = []
        for table in tables:
            if table == "users":
                related.append("profiles")
            elif table == "posts":
                related.append("comments")
                related.append("categories")

        return related


if __name__ == "__main__":
    # Example usage
    trainer = ExecutionAwareTrainer()

    # Example SQL queries
    correct_sql = "SELECT id, name FROM users WHERE age > 18"
    incorrect_sql = "SELECT id, name FROM user WHERE age > 18"  # Table name typo

    # Calculate loss
    loss = trainer.sql_correctness_loss(incorrect_sql, correct_sql, "example_db")
    print(f"SQL Correctness Loss: {loss}")

    # Evaluate query
    metrics = trainer.evaluate_query(incorrect_sql, correct_sql, "example_db")
    print(f"Evaluation Metrics: {metrics}")

    # Generate prompt
    prompt = trainer.generate_schema_aware_prompt(
        "example_db",
        ["users", "orders"],
        "Find all users who placed orders in January 2023",
    )
    print(f"Generated Prompt:\n{prompt}")
