"""
Adaptive Context Manager Module

This module implements dynamic token allocation based on query complexity
to optimize the prompt context within available token limits.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ComplexityLevel(Enum):
    """Query complexity levels."""

    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


@dataclass
class TokenAllocation:
    """Token allocation for different prompt components."""

    schema_summary: int
    examples: int
    query_patterns: int
    reasoning_chain: int
    common_mistakes: int
    type_constraints: int
    total: int


class AdaptiveContextManager:
    """
    A manager that dynamically allocates token budget based on query complexity
    to optimize the prompt context within available token limits.
    """

    def __init__(
        self,
        max_tokens: int = 4096,
        schema_analyzer=None,
        min_schema_tokens: int = 500,
        min_examples_tokens: int = 200,
    ):
        """
        Initialize the AdaptiveContextManager.

        Args:
            max_tokens: Maximum tokens available for context.
            schema_analyzer: Schema analyzer to use for schema information.
            min_schema_tokens: Minimum tokens to allocate for schema information.
            min_examples_tokens: Minimum tokens to allocate for examples.
        """
        self.max_tokens = max_tokens
        self.schema_analyzer = schema_analyzer
        self.min_schema_tokens = min_schema_tokens
        self.min_examples_tokens = min_examples_tokens

        # Default token allocation percentages by complexity
        self.allocation_percentages = {
            ComplexityLevel.SIMPLE: {
                "schema_summary": 40,
                "examples": 25,
                "query_patterns": 15,
                "reasoning_chain": 10,
                "common_mistakes": 5,
                "type_constraints": 5,
            },
            ComplexityLevel.MODERATE: {
                "schema_summary": 50,
                "examples": 20,
                "query_patterns": 10,
                "reasoning_chain": 10,
                "common_mistakes": 5,
                "type_constraints": 5,
            },
            ComplexityLevel.COMPLEX: {
                "schema_summary": 60,
                "examples": 15,
                "query_patterns": 5,
                "reasoning_chain": 10,
                "common_mistakes": 5,
                "type_constraints": 5,
            },
        }

        # Patterns to identify complexity factors
        self.join_pattern = (
            r"\b(join|inner join|left join|right join|full join|cross join)\b"
        )
        self.aggregation_pattern = r"\b(count|sum|avg|min|max|group by|having)\b"
        self.subquery_pattern = r"\(\s*select"
        self.window_pattern = r"\bover\s*\("
        self.union_pattern = r"\bunion\b"
        self.cte_pattern = r"\bwith\s+\w+\s+as\s*\("

    def analyze_query_complexity(
        self,
        query_text: str,
        detected_tables: List[str],
        sql_hint: Optional[str] = None,
    ) -> ComplexityLevel:
        """
        Analyze the complexity of a natural language query.

        Args:
            query_text: Natural language query.
            detected_tables: Tables detected in the query.
            sql_hint: Optional SQL hint if available.

        Returns:
            Query complexity level.
        """
        # Start with a complexity score
        complexity_score = 0

        # Check number of tables mentioned
        if len(detected_tables) == 1:
            complexity_score += 1
        elif len(detected_tables) <= 3:
            complexity_score += 2
        else:
            complexity_score += 3

        # Check for complex keywords in natural language query
        if re.search(
            r"\b(join|connect|relate|linked|associated)\b", query_text, re.IGNORECASE
        ):
            complexity_score += 1

        if re.search(
            r"\b(average|mean|sum|total|count|how many|number of)\b",
            query_text,
            re.IGNORECASE,
        ):
            complexity_score += 1

        if re.search(r"\b(group|categorize|each|per|by)\b", query_text, re.IGNORECASE):
            complexity_score += 1

        if re.search(
            r"\b(compare|ratio|percentage|proportion|relative|normalized)\b",
            query_text,
            re.IGNORECASE,
        ):
            complexity_score += 1

        if re.search(
            r"\b(rank|top|bottom|highest|lowest|best|worst)\b",
            query_text,
            re.IGNORECASE,
        ):
            complexity_score += 1

        if re.search(
            r"\b(before|after|when|date|time|period|duration)\b",
            query_text,
            re.IGNORECASE,
        ):
            complexity_score += 1

        # If we have an SQL hint, analyze it for complexity
        if sql_hint:
            if re.search(self.join_pattern, sql_hint, re.IGNORECASE):
                complexity_score += 1

            if re.search(self.aggregation_pattern, sql_hint, re.IGNORECASE):
                complexity_score += 1

            if re.search(self.subquery_pattern, sql_hint, re.IGNORECASE):
                complexity_score += 2

            if re.search(self.window_pattern, sql_hint, re.IGNORECASE):
                complexity_score += 2

            if re.search(self.union_pattern, sql_hint, re.IGNORECASE):
                complexity_score += 1

            if re.search(self.cte_pattern, sql_hint, re.IGNORECASE):
                complexity_score += 2

        # Determine complexity level based on score
        if complexity_score <= 3:
            return ComplexityLevel.SIMPLE
        elif complexity_score <= 6:
            return ComplexityLevel.MODERATE
        else:
            return ComplexityLevel.COMPLEX

    def allocate_tokens(self, complexity: ComplexityLevel) -> TokenAllocation:
        """
        Allocate tokens based on query complexity.

        Args:
            complexity: Query complexity level.

        Returns:
            TokenAllocation with token allocation for different prompt components.
        """
        percentages = self.allocation_percentages[complexity]

        # Calculate token allocations based on percentages
        schema_summary = max(
            int(self.max_tokens * percentages["schema_summary"] / 100),
            self.min_schema_tokens,
        )
        examples = max(
            int(self.max_tokens * percentages["examples"] / 100),
            self.min_examples_tokens,
        )
        query_patterns = int(self.max_tokens * percentages["query_patterns"] / 100)
        reasoning_chain = int(self.max_tokens * percentages["reasoning_chain"] / 100)
        common_mistakes = int(self.max_tokens * percentages["common_mistakes"] / 100)
        type_constraints = int(self.max_tokens * percentages["type_constraints"] / 100)

        # Calculate total
        total = (
            schema_summary
            + examples
            + query_patterns
            + reasoning_chain
            + common_mistakes
            + type_constraints
        )

        # If total exceeds max_tokens, adjust allocations proportionally
        if total > self.max_tokens:
            excess = total - self.max_tokens
            proportion = self.max_tokens / total

            # Adjust all allocations while respecting minimums
            schema_summary = max(
                int(schema_summary * proportion),
                self.min_schema_tokens,
            )
            examples = max(
                int(examples * proportion),
                self.min_examples_tokens,
            )
            query_patterns = int(query_patterns * proportion)
            reasoning_chain = int(reasoning_chain * proportion)
            common_mistakes = int(common_mistakes * proportion)
            type_constraints = int(type_constraints * proportion)

            # Recalculate total
            total = (
                schema_summary
                + examples
                + query_patterns
                + reasoning_chain
                + common_mistakes
                + type_constraints
            )

        return TokenAllocation(
            schema_summary=schema_summary,
            examples=examples,
            query_patterns=query_patterns,
            reasoning_chain=reasoning_chain,
            common_mistakes=common_mistakes,
            type_constraints=type_constraints,
            total=total,
        )

    def generate_adaptive_schema_summary(
        self,
        db_name: str,
        detected_tables: List[str],
        token_allocation: int,
        include_samples: bool = True,
    ) -> str:
        """
        Generate a schema summary that fits within the allocated token budget.

        Args:
            db_name: Database name.
            detected_tables: Tables detected in the query.
            token_allocation: Token allocation for schema summary.
            include_samples: Whether to include sample data.

        Returns:
            Schema summary string.
        """
        if self.schema_analyzer is None:
            logger.warning(
                "No schema analyzer provided. Cannot generate schema summary."
            )
            return "Schema information not available."

        # Get schema information
        schema_info = self.schema_analyzer.analyze_schema(db_name)

        # Estimate tokens for each table and detail level
        table_token_estimates = self._estimate_table_tokens(
            schema_info, detected_tables
        )

        # Prioritize tables based on detection and complexity
        prioritized_tables = self._prioritize_tables(
            detected_tables, table_token_estimates
        )

        # Determine how much detail we can include based on token allocation
        detail_level, tables_to_include = self._determine_detail_level(
            prioritized_tables, table_token_estimates, token_allocation
        )

        # Generate summary with appropriate detail level
        summary = self.schema_analyzer.generate_schema_summary(
            schema_info,
            tables_to_include,
            detail_level=detail_level,
            include_sample_data=include_samples,
        )

        # If we still have tokens available, add relationships
        estimated_tokens = self._estimate_string_tokens(summary)
        if estimated_tokens < token_allocation and len(tables_to_include) > 1:
            relationships = self._generate_relationship_summary(
                schema_info, tables_to_include, token_allocation - estimated_tokens
            )
            if relationships:
                summary += "\n\n" + relationships

        return summary

    def _estimate_table_tokens(
        self, schema_info, detected_tables: List[str]
    ) -> Dict[str, Dict[str, int]]:
        """
        Estimate token counts for tables at different detail levels.

        Args:
            schema_info: Schema information.
            detected_tables: Tables detected in the query.

        Returns:
            Dictionary mapping table names to detail level token estimates.
        """
        estimates = {}

        for table_name in detected_tables:
            if table_name in schema_info.tables:
                table_info = schema_info.tables[table_name]

                # Estimate tokens for different detail levels
                low_detail = 20 + 5 * len(table_info.columns)  # Basic info
                medium_detail = 40 + 15 * len(table_info.columns)  # More column details
                high_detail = 60 + 30 * len(
                    table_info.columns
                )  # Full details with samples

                estimates[table_name] = {
                    "low": low_detail,
                    "medium": medium_detail,
                    "high": high_detail,
                }

        return estimates

    def _prioritize_tables(
        self, detected_tables: List[str], token_estimates: Dict[str, Dict[str, int]]
    ) -> List[str]:
        """
        Prioritize tables based on detection order and complexity.

        Args:
            detected_tables: Tables detected in the query.
            token_estimates: Token estimates for tables.

        Returns:
            List of prioritized table names.
        """
        # For now, simply use the detection order as the priority
        # In a more sophisticated implementation, we might consider:
        # - Tables mentioned first in the query
        # - Tables with more columns or more complex structure
        # - Tables that are central in the schema (more relationships)
        return detected_tables

    def _determine_detail_level(
        self,
        prioritized_tables: List[str],
        token_estimates: Dict[str, Dict[str, int]],
        token_allocation: int,
    ) -> Tuple[str, List[str]]:
        """
        Determine the appropriate detail level based on token allocation.

        Args:
            prioritized_tables: Prioritized list of tables.
            token_estimates: Token estimates for tables.
            token_allocation: Token allocation for schema summary.

        Returns:
            Tuple of (detail_level, tables_to_include).
        """
        # Start with highest detail level
        detail_level = "high"
        tables_to_include = []
        total_tokens = 0

        # Try to include as many tables as possible at the highest detail level
        for table in prioritized_tables:
            if table in token_estimates:
                if (
                    total_tokens + token_estimates[table][detail_level]
                    <= token_allocation
                ):
                    tables_to_include.append(table)
                    total_tokens += token_estimates[table][detail_level]

        # If we couldn't include all tables, try medium detail
        if len(tables_to_include) < len(prioritized_tables):
            detail_level = "medium"
            tables_to_include = []
            total_tokens = 0

            for table in prioritized_tables:
                if table in token_estimates:
                    if (
                        total_tokens + token_estimates[table][detail_level]
                        <= token_allocation
                    ):
                        tables_to_include.append(table)
                        total_tokens += token_estimates[table][detail_level]

        # If we still couldn't include all tables, try low detail
        if len(tables_to_include) < len(prioritized_tables):
            detail_level = "low"
            tables_to_include = []
            total_tokens = 0

            for table in prioritized_tables:
                if table in token_estimates:
                    if (
                        total_tokens + token_estimates[table][detail_level]
                        <= token_allocation
                    ):
                        tables_to_include.append(table)
                        total_tokens += token_estimates[table][detail_level]

        # If we still couldn't include all tables, prioritize the first ones
        if len(tables_to_include) < len(prioritized_tables):
            tables_to_include = prioritized_tables[
                : max(1, len(tables_to_include))
            ]  # Include at least one table

        # Map our detail levels to schema analyzer detail levels
        detail_level_map = {
            "low": "low",
            "medium": "medium",
            "high": "high",
        }

        return detail_level_map[detail_level], tables_to_include

    def _generate_relationship_summary(
        self, schema_info, tables: List[str], token_allocation: int
    ) -> str:
        """
        Generate a summary of relationships between tables.

        Args:
            schema_info: Schema information.
            tables: Tables to include in the summary.
            token_allocation: Token allocation for relationships.

        Returns:
            Relationship summary string.
        """
        relationships = []

        # Filter relationships involving the specified tables
        for src_table, src_col, dst_table, dst_col in schema_info.relationships:
            if src_table in tables and dst_table in tables:
                relationships.append(f"{src_table}.{src_col} -> {dst_table}.{dst_col}")

        if not relationships:
            return ""

        # Generate summary
        summary = "Relationships:\n"
        summary += "\n".join(f"- {rel}" for rel in relationships)

        # Check if it fits within token allocation
        if self._estimate_string_tokens(summary) > token_allocation:
            # If not, include fewer relationships
            max_relationships = max(
                1, int(token_allocation / 10)
            )  # Rough estimate: 10 tokens per relationship
            relationships = relationships[:max_relationships]
            summary = "Relationships (partial):\n"
            summary += "\n".join(f"- {rel}" for rel in relationships)

        return summary

    def generate_query_patterns(
        self, query_text: str, detected_tables: List[str], token_allocation: int
    ) -> str:
        """
        Generate query patterns that fit within the allocated token budget.

        Args:
            query_text: Natural language query.
            detected_tables: Tables detected in the query.
            token_allocation: Token allocation for query patterns.

        Returns:
            Query patterns string.
        """
        # This would normally load patterns from a knowledge base
        # For now, we'll return a simple template
        patterns = []

        if "count" in query_text.lower() or "how many" in query_text.lower():
            patterns.append("COUNT Pattern: SELECT COUNT(*) FROM table WHERE condition")

        if "average" in query_text.lower() or "mean" in query_text.lower():
            patterns.append(
                "AVG Pattern: SELECT AVG(column) FROM table WHERE condition"
            )

        if "join" in query_text.lower() or len(detected_tables) > 1:
            patterns.append(
                "JOIN Pattern: SELECT columns FROM table1 JOIN table2 ON table1.id = table2.table1_id"
            )

        if "group" in query_text.lower():
            patterns.append(
                "GROUP BY Pattern: SELECT column, COUNT(*) FROM table GROUP BY column"
            )

        # Limit patterns to fit token allocation
        patterns_text = "Relevant Query Patterns:\n" + "\n".join(
            f"- {p}" for p in patterns
        )

        while (
            self._estimate_string_tokens(patterns_text) > token_allocation and patterns
        ):
            patterns.pop()  # Remove one pattern
            patterns_text = "Relevant Query Patterns:\n" + "\n".join(
                f"- {p}" for p in patterns
            )

        return patterns_text

    def generate_common_mistakes(
        self, query_text: str, detected_tables: List[str], token_allocation: int
    ) -> str:
        """
        Generate common mistakes that fit within the allocated token budget.

        Args:
            query_text: Natural language query.
            detected_tables: Tables detected in the query.
            token_allocation: Token allocation for common mistakes.

        Returns:
            Common mistakes string.
        """
        # This would normally load common mistakes from a knowledge base
        # For now, we'll return a simple template
        mistakes = [
            "Remember to use appropriate JOIN conditions between tables",
            "Be careful with NULL handling in WHERE clauses",
            "Avoid cartesian products by specifying JOIN conditions",
        ]

        if len(detected_tables) > 1:
            mistakes.append(
                f"Don't confuse the primary keys between {' and '.join(detected_tables)}"
            )

        if "group" in query_text.lower():
            mistakes.append("Ensure non-aggregated columns appear in GROUP BY clause")

        if "where" in query_text.lower() and "null" in query_text.lower():
            mistakes.append("Use IS NULL instead of = NULL for NULL comparisons")

        # Limit mistakes to fit token allocation
        mistakes_text = "Common Mistakes:\n" + "\n".join(f"- {m}" for m in mistakes)

        while (
            self._estimate_string_tokens(mistakes_text) > token_allocation and mistakes
        ):
            mistakes.pop()  # Remove one mistake
            mistakes_text = "Common Mistakes:\n" + "\n".join(f"- {m}" for m in mistakes)

        return mistakes_text

    def generate_type_constraints(
        self, db_name: str, detected_tables: List[str], token_allocation: int
    ) -> str:
        """
        Generate type constraints that fit within the allocated token budget.

        Args:
            db_name: Database name.
            detected_tables: Tables detected in the query.
            token_allocation: Token allocation for type constraints.

        Returns:
            Type constraints string.
        """
        if self.schema_analyzer is None:
            logger.warning(
                "No schema analyzer provided. Cannot generate type constraints."
            )
            return ""

        # Get column constraints
        constraints = self.schema_analyzer.infer_column_constraints(
            self.schema_analyzer.analyze_schema(db_name)
        )

        # Filter for detected tables
        relevant_constraints = []
        for table_name in detected_tables:
            if table_name in constraints:
                for column_name, constraint in constraints[table_name].items():
                    relevant_constraints.append(
                        f"{table_name}.{column_name}: {constraint}"
                    )

        # Limit constraints to fit token allocation
        constraints_text = "Type Constraints:\n" + "\n".join(
            f"- {c}" for c in relevant_constraints
        )

        while (
            self._estimate_string_tokens(constraints_text) > token_allocation
            and relevant_constraints
        ):
            relevant_constraints.pop()  # Remove one constraint
            constraints_text = "Type Constraints:\n" + "\n".join(
                f"- {c}" for c in relevant_constraints
            )

        return constraints_text

    def generate_examples(
        self, db_name: str, detected_tables: List[str], token_allocation: int
    ) -> str:
        """
        Generate examples that fit within the allocated token budget.

        Args:
            db_name: Database name.
            detected_tables: Tables detected in the query.
            token_allocation: Token allocation for examples.

        Returns:
            Examples string.
        """
        # This would normally load examples from a knowledge base
        # For now, we'll return a simple template
        examples = []

        if len(detected_tables) == 1:
            table = detected_tables[0]
            examples.append(f"Example 1: SELECT * FROM {table} WHERE id = 1")
            examples.append(
                f"Example 2: SELECT name, created_at FROM {table} ORDER BY created_at DESC LIMIT 10"
            )
        elif len(detected_tables) > 1:
            table1 = detected_tables[0]
            table2 = detected_tables[1]
            examples.append(
                f"Example 1: SELECT {table1}.name, {table2}.name FROM {table1} JOIN {table2} ON {table1}.id = {table2}.{table1[:-1]}_id"
            )

            if len(detected_tables) > 2:
                table3 = detected_tables[2]
                examples.append(
                    f"Example 2: SELECT {table1}.name, COUNT({table3}.id) FROM {table1} JOIN {table2} ON {table1}.id = {table2}.{table1[:-1]}_id JOIN {table3} ON {table2}.id = {table3}.{table2[:-1]}_id GROUP BY {table1}.id"
                )

        # Limit examples to fit token allocation
        examples_text = "Examples:\n" + "\n".join(f"{e}" for e in examples)

        while (
            self._estimate_string_tokens(examples_text) > token_allocation and examples
        ):
            examples.pop()  # Remove one example
            examples_text = "Examples:\n" + "\n".join(f"{e}" for e in examples)

        return examples_text

    def generate_reasoning_chain(
        self, detected_tables: List[str], token_allocation: int
    ) -> str:
        """
        Generate reasoning chain that fits within the allocated token budget.

        Args:
            detected_tables: Tables detected in the query.
            token_allocation: Token allocation for reasoning chain.

        Returns:
            Reasoning chain string.
        """
        # Define reasoning steps
        steps = [
            "1. Identify tables needed",
            "2. Determine join conditions",
            "3. Select appropriate columns",
            "4. Apply filter conditions",
            "5. Determine if aggregation is needed",
            "6. Apply ordering and limits",
        ]

        # Add table-specific reasoning
        for i, table in enumerate(detected_tables[:3]):  # Limit to first 3 tables
            steps.append(
                f"Table {i + 1} ({table}): Identify relevant columns and constraints"
            )

        # Limit steps to fit token allocation
        chain_text = "Reasoning Chain:\n" + "\n".join(steps)

        while (
            self._estimate_string_tokens(chain_text) > token_allocation
            and len(steps) > 1
        ):
            steps.pop()  # Remove one step
            chain_text = "Reasoning Chain:\n" + "\n".join(steps)

        return chain_text

    def generate_adaptive_context(
        self,
        query_text: str,
        db_name: str,
        detected_tables: List[str],
        sql_hint: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Generate a complete adaptive context for the prompt.

        Args:
            query_text: Natural language query.
            db_name: Database name.
            detected_tables: Tables detected in the query.
            sql_hint: Optional SQL hint if available.

        Returns:
            Dictionary with context components.
        """
        # Analyze query complexity
        complexity = self.analyze_query_complexity(
            query_text, detected_tables, sql_hint
        )
        logger.info(f"Query complexity: {complexity.value}")

        # Allocate tokens
        allocation = self.allocate_tokens(complexity)
        logger.info(f"Token allocation: {allocation}")

        # Generate context components
        schema_summary = self.generate_adaptive_schema_summary(
            db_name, detected_tables, allocation.schema_summary
        )

        examples = self.generate_examples(db_name, detected_tables, allocation.examples)

        query_patterns = self.generate_query_patterns(
            query_text, detected_tables, allocation.query_patterns
        )

        reasoning_chain = self.generate_reasoning_chain(
            detected_tables, allocation.reasoning_chain
        )

        common_mistakes = self.generate_common_mistakes(
            query_text, detected_tables, allocation.common_mistakes
        )

        type_constraints = self.generate_type_constraints(
            db_name, detected_tables, allocation.type_constraints
        )

        # Combine into context
        context = {
            "schema_summary": schema_summary,
            "examples": examples,
            "query_patterns": query_patterns,
            "reasoning_chain": reasoning_chain,
            "common_mistakes": common_mistakes,
            "type_constraints": type_constraints,
            "complexity": complexity.value,
            "token_allocation": {
                "schema_summary": allocation.schema_summary,
                "examples": allocation.examples,
                "query_patterns": allocation.query_patterns,
                "reasoning_chain": allocation.reasoning_chain,
                "common_mistakes": allocation.common_mistakes,
                "type_constraints": allocation.type_constraints,
                "total": allocation.total,
            },
        }

        return context

    def _estimate_string_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a string.

        Args:
            text: String to estimate tokens for.

        Returns:
            Estimated token count.
        """
        # Simple heuristic: ~4 characters per token for English text
        return len(text) // 4 + 1


if __name__ == "__main__":
    # Example usage
    context_manager = AdaptiveContextManager()

    # Example query
    query = "Find all orders placed by users in New York in the last month, along with the products they ordered"
    detected_tables = ["users", "orders", "order_items", "products"]

    # Analyze complexity
    complexity = context_manager.analyze_query_complexity(query, detected_tables)
    print(f"Query complexity: {complexity.value}")

    # Allocate tokens
    allocation = context_manager.allocate_tokens(complexity)
    print(f"Token allocation: {allocation}")

    # Mocked example (in real usage, you would pass a schema analyzer)
    context = {
        "schema_summary": f"Token budget: {allocation.schema_summary}",
        "examples": f"Token budget: {allocation.examples}",
        "query_patterns": f"Token budget: {allocation.query_patterns}",
        "reasoning_chain": f"Token budget: {allocation.reasoning_chain}",
        "common_mistakes": f"Token budget: {allocation.common_mistakes}",
        "type_constraints": f"Token budget: {allocation.type_constraints}",
    }

    print("\nGenerated Context Components:")
    for key, value in context.items():
        print(f"{key}: {value}")
