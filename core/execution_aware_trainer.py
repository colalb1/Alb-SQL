# WARNING: This module might be used differently depending on the execution
# mode selected in main.py (e.g., 'eval' vs 'example'). Ensure compatibility
# if making changes related to initialization or external dependencies like
# db_connector.

"""
Execution-Aware Trainer Module

This module implements execution-aware training and validation for SQL
generation, ensuring that the generated SQL queries are not only
syntactically correct but also produce the expected results when executed
against the database.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
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
            execution_timeout (int): Max time (seconds) for query execution.
            max_rows_to_compare (int): Max rows to compare for results.
            plan_weight (float): Weight of execution plan similarity in loss.
            result_weight (float): Weight of result similarity in loss.
        """
        self.db_connector = db_connector
        self.execution_timeout = execution_timeout
        self.max_rows_to_compare = max_rows_to_compare
        self.plan_weight = plan_weight
        self.result_weight = result_weight
        # Using Dict for query history, mapping db_name to list of
        # (query, result) tuples.
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
            logger.warning("No db_connector provided. Cannot execute query.")
            return QueryExecutionResult(
                success=False, error_message="No database connector"
            )

        start_time = time.time()
        try:
            # This would use the actual database connector.
            # For now, we simulate execution results.
            success = True
            rows = [{"id": i, "name": f"Sample_{i}"} for i in range(1, 11)]
            row_count = len(rows)
            execution_plan = None
            if get_execution_plan:
                execution_plan = {"plan_type": "sequential_scan", "cost": 100}
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
        self.query_history.setdefault(db_name, []).append((query, result))

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
            logger.warning("No db_connector provided. Cannot get execution plan.")
            return None

        try:
            # This would use the actual database connector to get the plan.
            # For now, we'll simulate an execution plan.
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
            result1 (List[Dict[str, Any]]): First result set (list of rows).
            result2 (List[Dict[str, Any]]): Second result set (list of rows).

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

        # Check if all rows match exactly (order-sensitive).
        # TODO: Consider order-insensitive comparison (e.g., sets of tuples).
        matching_rows = 0
        num_rows_to_check = min(len(result1), len(result2))
        for i in range(num_rows_to_check):
            # Simple dictionary comparison
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

            # 70% for matching plan type, 30% for cost similarity
            return 0.7 + (0.3 * cost_similarity)
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

        # Convert similarity to loss (higher similarity -> lower loss)
        loss = max(0.0, 1.0 - similarity)

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

        # Semantic correctness - simplified: assumes result match implies semantics
        # TODO: Implement more robust semantic checks (e.g., AST comparison).
        semantic_correctness = result_match_score

        # Execution efficiency score (0-1, higher is better)
        execution_efficiency = 0.0
        if (
            execution_success
            and reference_result.success
            and reference_result.execution_time > 0
        ):
            gen_time = generated_result.execution_time
            ref_time = reference_result.execution_time
            # Normalize score: 1 if gen_time <= ref_time, decreases otherwise.
            # Example: gen_time = 2*ref_time => ratio = 1 => score = 0.5
            # Example: gen_time = 0.5*ref_time => ratio = -0.5 => score = 1.0
            time_ratio = max(0.0, (gen_time - ref_time) / ref_time)
            execution_efficiency = 1.0 / (1.0 + time_ratio)

        # Overall score - weighted average
        overall_score = 0.0
        if execution_success:
            # Define weights for different metrics
            weights = {
                "syntax": 0.1,  # Basic check
                "semantic": 0.4,  # Currently tied to result match
                "result_match": 0.4,  # Core correctness metric
                "efficiency": 0.1,  # Performance aspect
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
            include_sample_data (bool): Whether to include sample data in prompt.
            complexity_estimate (str): Estimated query complexity ('simple', etc.).

        Returns:
            Formatted prompt string for an LLM.
        """
        # This method seems less related to the trainer's core execution-aware
        # logic and more to prompt generation, potentially belonging elsewhere
        # or using components like AdaptiveContextManager.
        # Keeping the placeholder implementation for now.

        # Determine detail level based on complexity estimate
        if complexity_estimate == "simple":
            detail_level = "brief"
            tables_to_include = target_tables[:2]  # Limit tables
        elif complexity_estimate == "medium":
            detail_level = "moderate"
            tables_to_include = target_tables
        else:  # complex
            detail_level = "detailed"
            # Include related tables for complex queries
            related_tables = self._get_related_tables(db_name, target_tables)
            tables_to_include = list(set(target_tables + related_tables))

        # Generate schema summary using the determined parameters
        schema_summary = self._generate_schema_summary(
            db_name, tables_to_include, detail_level, include_sample_data
        )

        # Get common mistakes relevant to the included tables/DB
        common_mistakes = self._get_common_mistakes(db_name, tables_to_include)

        # Construct the prompt string
        prompt = f"""**Role**: World-class SQL Engineer + Database Architect
**Task**: Solve the following problem for the '{db_name}' database.

**Question**:
{question}

**Schema Context**:
{schema_summary}

**Common Mistakes to Avoid**:
{common_mistakes}

**General Reasoning Steps**:
1. Identify relevant tables.
2. Determine JOIN conditions if multiple tables are needed.
3. Apply WHERE clause filters based on the question.
4. SELECT the required columns.
5. Apply aggregation (COUNT, SUM, AVG, etc.) and GROUP BY if needed.
6. Apply ORDER BY and LIMIT if needed.

**Response Format**: Provide only the SQL query, enclosed in ```sql ... ```."""
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
            detail_level (str): Detail level ('brief', 'moderate', 'detailed').
            include_sample_data (bool): Whether to include sample data rows.

        Returns:
            A string summarizing the schema for the specified tables.
        """
        # This is a placeholder. Ideally, use SchemaAnalyzerAgent.
        logger.debug(
            f"Generating mock schema summary for {tables} (detail: {detail_level})"
        )
        summary = f"Schema Summary ({detail_level}):\n"

        for table in tables:
            summary += f"- {table}\n"

            # Add column details based on detail level
            if detail_level == "brief":
                summary += f"  - {table}(id, name, ...)\n"  # Minimal columns
            else:  # moderate or detailed
                # Placeholder columns - replace with actual schema lookup
                cols = ["id (INT, PK)", "name (TEXT)", f"{table}_related_id (INT, FK)"]
                summary += f"  - {table}({', '.join(cols)})\n"
                if detail_level == "detailed":
                    summary += f"    Constraints: PK(id), FK({table}_related_id)\n"

            # Include sample data if requested and detail allows
            if include_sample_data and detail_level != "brief":
                summary += "    Sample: (1, 'Sample Name', 101)\n"

        # Add relationships if detailed
        if detail_level == "detailed":
            # Placeholder relationships - replace with actual schema lookup
            summary += "\nRelationships:\n"
            summary += "  table1.id = table2.table1_id\n"

        return summary

    def _get_common_mistakes(self, db_name: str, tables: List[str]) -> str:
        """
        Get common mistakes for the given database and tables.

        Args:
            db_name (str): Database name.
            tables (List[str]): List of relevant tables.

        Returns:
            A string listing common mistakes related to the tables/DB.
        """
        # Placeholder - replace with actual knowledge base lookup.
        mistakes = [
            "- Ensure correct JOIN conditions.",
            "- Use IS NULL/IS NOT NULL for NULL checks.",
            "- Include non-aggregated columns in GROUP BY.",
        ]
        if "orders" in tables and "users" in tables:
            mistakes.append("- Check join direction between users and orders.")

        return "\n".join(mistakes)

    def _get_related_tables(self, db_name: str, tables: List[str]) -> List[str]:
        """
        Get related tables based on foreign key relationships.

        Args:
            db_name (str): Database name.
            tables (List[str]): List of primary tables.

        Returns:
            A list of table names related to the primary tables via FKs.
        """
        # Placeholder - replace with actual schema introspection.
        related_map = {
            "users": ["orders", "profiles"],
            "orders": ["order_items", "users"],
            "products": ["order_items"],
            "posts": ["comments", "users"],
        }
        related = set()
        for table in tables:
            related.update(related_map.get(table, []))
        # Return only tables not already in the primary list
        return list(related - set(tables))

        return related


if __name__ == "__main__":
    # Example usage
    trainer = ExecutionAwareTrainer()

    logging.basicConfig(level=logging.INFO)
    logger.info("Running ExecutionAwareTrainer example...")

    # Mock DB Connector (replace with actual connector if needed)
    class MockDbConnector:
        def execute(self, db_name, query):
            logger.info(f"Mock executing on {db_name}: {query[:50]}...")
            if "FROM user " in query:  # Simulate error
                raise Exception("Table 'user' not found")
            # Simulate successful execution
            return [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]

        def get_plan(self, db_name, query):
            logger.info(f"Mock getting plan for: {query[:50]}...")
            return {"plan_type": "index_scan", "cost": 50}

    trainer = ExecutionAwareTrainer(db_connector=MockDbConnector())

    # Example evaluation
    correct_sql = "SELECT id, name FROM users WHERE age > 18"
    generated_sql_ok = "SELECT id, name FROM users WHERE age > 18 ORDER BY id"
    generated_sql_bad_table = "SELECT id, name FROM user WHERE age > 18"

    print("\n--- Evaluating OK Query ---")
    metrics_ok = trainer.evaluate_query(generated_sql_ok, correct_sql, "db1")
    print(metrics_ok)

    print("\n--- Evaluating Bad Table Query ---")
    metrics_bad = trainer.evaluate_query(generated_sql_bad_table, correct_sql, "db1")
    print(metrics_bad)

    # Example prompt generation (uses mock schema methods)
    print("\n--- Generating Prompt ---")
    prompt = trainer.generate_schema_aware_prompt(
        db_name="db1",
        target_tables=["users", "orders"],
        question="Show users and their order counts.",
        complexity_estimate="medium",
    )
    print(prompt)
    print("\n--- Example End ---")
