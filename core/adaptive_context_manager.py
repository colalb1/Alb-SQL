# WARNING: This module might be used differently depending on the execution mode
# selected in main.py (e.g., 'eval' vs 'example'). Ensure compatibility
# if making changes related to initialization or external dependencies.

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
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
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
            max_tokens (int): Maximum tokens available for context.
            schema_analyzer: Schema analyzer for schema information.
            min_schema_tokens (int): Min tokens for schema info.
            min_examples_tokens (int): Min tokens for examples.
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
        join_kw = r"join|inner join|left join|right join|full join|cross join"
        agg_kw = r"count|sum|avg|min|max|group by|having"
        self.join_pattern = rf"\b({join_kw})\b"
        self.aggregation_pattern = rf"\b({agg_kw})\b"
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
            query_text (str): Natural language query.
            detected_tables (List[str]): Tables detected in the query.
            sql_hint (Optional[str]): Optional SQL hint if available.

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
        nl_patterns = [
            r"\b(join|connect|relate|linked|associated)\b",
            r"\b(average|mean|sum|total|count|how many|number of)\b",
            r"\b(group|categorize|each|per|by)\b",
            r"\b(compare|ratio|percentage|proportion|relative|normalized)\b",
            r"\b(rank|top|bottom|highest|lowest|best|worst)\b",
            r"\b(before|after|when|date|time|period|duration)\b",
        ]
        for pattern in nl_patterns:
            if re.search(pattern, query_text, re.IGNORECASE):
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
            complexity (ComplexityLevel): Query complexity level.

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
            # excess = total - self.max_tokens # Unused variable
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
            db_name (str): Database name.
            detected_tables (List[str]): Tables detected in the query.
            token_allocation (int): Token allocation for schema summary.
            include_samples (bool): Whether to include sample data.

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

        # If we still have tokens available, add relationships summary
        estimated_summary_tokens = self._estimate_string_tokens(summary)
        remaining_tokens = token_allocation - estimated_summary_tokens
        if remaining_tokens > 20 and len(tables_to_include) > 1:  # Min threshold
            relationships_summary = self._generate_relationship_summary(
                schema_info, tables_to_include, remaining_tokens
            )
            if relationships_summary:
                summary += "\n\n" + relationships_summary

        return summary

    def _estimate_table_tokens(
        self, schema_info, detected_tables: List[str]
    ) -> Dict[str, Dict[str, int]]:
        """
        Estimate token counts for tables at different detail levels.

        Args:
            schema_info: Schema information.
            detected_tables (List[str]): Tables detected in the query.

        Returns:
            Dictionary mapping table names to detail level token estimates.
        """
        estimates = {}

        for table_name in detected_tables:
            if table_name in schema_info.tables:
                table_info = schema_info.tables[table_name]

                # Estimate tokens for different detail levels
                num_cols = len(table_info.columns)
                low_detail = 20 + 5 * num_cols  # Basic info
                medium_detail = 40 + 15 * num_cols  # More column details
                # Full details with samples
                high_detail = 60 + 30 * num_cols

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
            detected_tables (List[str]): Tables detected in the query.
            token_estimates (Dict[str, Dict[str, int]]): Token estimates for tables.

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
            prioritized_tables (List[str]): Prioritized list of tables.
            token_estimates (Dict[str, Dict[str, int]]): Token estimates for tables.
            token_allocation (int): Token allocation for schema summary.

        Returns:
            Tuple of (detail_level, tables_to_include).
        """
        # Start with highest detail level
        detail_level = "high"
        tables_to_include = []
        total_tokens = 0

        # Iterate through detail levels (high, medium, low)
        for current_detail in ["high", "medium", "low"]:
            tables_can_include = []
            current_total_tokens = 0
            for table in prioritized_tables:
                if table in token_estimates:
                    cost = token_estimates[table][current_detail]
                    if current_total_tokens + cost <= token_allocation:
                        tables_can_include.append(table)
                        current_total_tokens += cost
                    else:
                        # Cannot add this table at this detail level, break inner loop
                        break  # Move to next table for this detail level

            # If this detail level allows including all tables, use it
            if len(tables_can_include) == len(prioritized_tables):
                detail_level = current_detail
                tables_to_include = tables_can_include
                break  # Found the best fit for all tables

            # If this is the first successful level, store it as a fallback
            if not tables_to_include:
                detail_level = current_detail
                tables_to_include = tables_can_include

            # If medium/low allows more tables than high, update
            elif len(tables_can_include) > len(tables_to_include):
                detail_level = current_detail
                tables_to_include = tables_can_include

        # If even low detail couldn't include any table, include at least the first one
        if not tables_to_include and prioritized_tables:
            first_table = prioritized_tables[0]
            if (
                first_table in token_estimates
                and token_estimates[first_table]["low"] <= token_allocation
            ):
                tables_to_include = [first_table]
                detail_level = "low"
            else:
                # Cannot even fit the first table at low detail
                logger.warning(
                    f"Cannot fit even the first table '{first_table}' within token budget {token_allocation}"
                )
                tables_to_include = []
                detail_level = "low"  # Default, though nothing will be shown

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
            tables (List[str]): Tables to include in the summary.
            token_allocation (int): Token allocation for relationships.

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

        # Check if summary fits within token allocation, truncate if needed
        summary_tokens = self._estimate_string_tokens(summary)
        if summary_tokens > token_allocation:
            # Simple truncation strategy: keep removing last line until it fits
            lines = summary.split("\n")
            while (
                self._estimate_string_tokens("\n".join(lines)) > token_allocation
                and len(lines) > 1
            ):
                lines.pop()
            summary = "\n".join(lines)
            # Add indication of truncation if lines were removed
            if not summary.endswith("..."):
                summary += "\n..."  # Indicate truncation

        return summary

    def generate_query_patterns(
        self, query_text: str, detected_tables: List[str], token_allocation: int
    ) -> str:
        """
        Generate query patterns that fit within the allocated token budget.

        Args:
            query_text (str): Natural language query.
            detected_tables (List[str]): Tables detected in the query.
            token_allocation (int): Token allocation for query patterns.

        Returns:
            Query patterns string.
        """
        # This would normally load patterns from a knowledge base
        # For now, we'll return a simple template
        patterns = []

        # Simplified pattern generation for brevity
        if "count" in query_text.lower() or "how many" in query_text.lower():
            patterns.append("COUNT(*): SELECT COUNT(*) FROM ... WHERE ...")
        if "average" in query_text.lower() or "mean" in query_text.lower():
            patterns.append("AVG(col): SELECT AVG(column) FROM ... WHERE ...")
        if "join" in query_text.lower() or len(detected_tables) > 1:
            patterns.append("JOIN: SELECT ... FROM t1 JOIN t2 ON t1.id = t2.t1_id")
        if "group" in query_text.lower():
            patterns.append("GROUP BY: SELECT col, COUNT(*) FROM ... GROUP BY col")

        # Limit patterns to fit token allocation
        patterns_text = "Relevant Query Patterns:\n" + "\n".join(
            f"- {p}" for p in patterns
        )
        patterns_text = self._truncate_text_by_tokens(patterns_text, token_allocation)

        return patterns_text

    def generate_common_mistakes(
        self, query_text: str, detected_tables: List[str], token_allocation: int
    ) -> str:
        """
        Generate common mistakes that fit within the allocated token budget.

        Args:
            query_text (str): Natural language query.
            detected_tables (List[str]): Tables detected in the query.
            token_allocation (int): Token allocation for common mistakes.

        Returns:
            Common mistakes string.
        """
        # Simplified mistakes for brevity
        mistakes = [
            "JOIN conditions: Ensure correct columns are used.",
            "NULL handling: Use IS NULL / IS NOT NULL.",
            "GROUP BY: Include all non-aggregated SELECT columns.",
            "Aliases: Use table aliases in complex joins.",
        ]
        if len(detected_tables) > 1:
            mistakes.append(f"PK/FK: Check keys for {'/'.join(detected_tables)}.")

        # Limit mistakes to fit token allocation
        mistakes_text = "Common Mistakes:\n" + "\n".join(f"- {m}" for m in mistakes)
        mistakes_text = self._truncate_text_by_tokens(mistakes_text, token_allocation)

        return mistakes_text

    def generate_type_constraints(
        self, db_name: str, detected_tables: List[str], token_allocation: int
    ) -> str:
        """
        Generate type constraints that fit within the allocated token budget.

        Args:
            db_name (str): Database name.
            detected_tables (List[str]): Tables detected in the query.
            token_allocation (int): Token allocation for type constraints.

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
        constraints_text = self._truncate_text_by_tokens(
            constraints_text, token_allocation
        )

        return constraints_text

    def generate_examples(
        self, db_name: str, detected_tables: List[str], token_allocation: int
    ) -> str:
        """
        Generate examples that fit within the allocated token budget.

        Args:
            db_name (str): Database name.
            detected_tables (List[str]): Tables detected in the query.
            token_allocation (int): Token allocation for examples.

        Returns:
            Examples string.
        """
        # Simplified examples for brevity
        examples = []
        if detected_tables:
            t1 = detected_tables[0]
            examples.append(f"Ex 1 (Filter): SELECT name FROM {t1} WHERE id = ?")
            if len(detected_tables) > 1:
                t2 = detected_tables[1]
                # Assume simple FK naming convention for example
                fk = f"{t1}_id" if not t1.endswith("s") else f"{t1[:-1]}_id"
                examples.append(
                    f"Ex 2 (Join): SELECT t1.*, t2.* FROM {t1} t1 JOIN {t2} t2 ON t1.id = t2.{fk}"
                )

        # Limit examples to fit token allocation
        examples_text = "Examples:\n" + "\n".join(examples)
        examples_text = self._truncate_text_by_tokens(examples_text, token_allocation)

        return examples_text

    def generate_reasoning_chain(
        self, detected_tables: List[str], token_allocation: int
    ) -> str:
        """
        Generate reasoning chain that fits within the allocated token budget.

        Args:
            detected_tables (List[str]): Tables detected in the query.
            token_allocation (int): Token allocation for reasoning chain.

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
        chain_text = self._truncate_text_by_tokens(chain_text, token_allocation)

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
            query_text (str): Natural language query.
            db_name (str): Database name.
            detected_tables (List[str]): Tables detected in the query.
            sql_hint (Optional[str]): Optional SQL hint if available.

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

        # Generate context components within allocated budgets
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
            text (str): String to estimate tokens for.

        Returns:
            Estimated token count.
        """
        # Simple heuristic: ~4 characters per token for English text.
        # Add 1 to avoid division by zero and account for small strings.
        return (len(text) + 3) // 4

    def _truncate_text_by_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within a token budget."""
        estimated_tokens = self._estimate_string_tokens(text)
        if estimated_tokens <= max_tokens:
            return text

        # Estimate target character count (leave some buffer)
        target_chars = max(0, max_tokens * 4 - 20)
        truncated_text = text[:target_chars]

        # Try to truncate at a newline boundary if possible
        last_newline = truncated_text.rfind("\n")
        if last_newline > 0:
            truncated_text = truncated_text[:last_newline]

        return truncated_text + "\n..."  # Indicate truncation


if __name__ == "__main__":
    # Example usage (requires a mock schema analyzer or actual setup)
    logging.basicConfig(level=logging.INFO)
    logger.info("Running AdaptiveContextManager example...")

    # Mock Schema Analyzer (replace with actual implementation)
    class MockSchemaAnalyzer:
        def analyze_schema(self, db_name):
            # Return minimal mock schema info
            from collections import namedtuple

            TableInfo = namedtuple("TableInfo", ["columns"])
            ColumnInfo = namedtuple("ColumnInfo", ["name", "type"])
            SchemaInfo = namedtuple("SchemaInfo", ["tables", "relationships"])
            return SchemaInfo(
                tables={
                    "users": TableInfo(
                        columns=[ColumnInfo("id", "INT"), ColumnInfo("name", "TEXT")]
                    ),
                    "orders": TableInfo(
                        columns=[ColumnInfo("id", "INT"), ColumnInfo("user_id", "INT")]
                    ),
                    "products": TableInfo(
                        columns=[ColumnInfo("id", "INT"), ColumnInfo("desc", "TEXT")]
                    ),
                },
                relationships=[("orders", "user_id", "users", "id")],
            )

        def generate_schema_summary(
            self, schema_info, tables, detail_level, include_sample_data
        ):
            return f"Mock Summary ({detail_level}) for: {', '.join(tables)}"

        def infer_column_constraints(self, schema_info):
            return {"users": {"id": "PK", "name": "TEXT"}}

    context_manager = AdaptiveContextManager(schema_analyzer=MockSchemaAnalyzer())

    query = "Find orders for user 'Alice'"
    detected = ["users", "orders"]
    context = context_manager.generate_adaptive_context(query, "mock_db", detected)

    print("\n--- Adaptive Context Example ---")
    print(f"Query: {query}")
    print(f"Complexity: {context.get('complexity')}")
    print(f"Token Allocation: {context.get('token_allocation')}")
    print("\nGenerated Context:")
    print(f"Schema Summary:\n{context.get('schema_summary')}\n")
    print(f"Examples:\n{context.get('examples')}\n")
    print(f"Query Patterns:\n{context.get('query_patterns')}\n")
    print(f"Reasoning Chain:\n{context.get('reasoning_chain')}\n")
    print(f"Common Mistakes:\n{context.get('common_mistakes')}\n")
    print(f"Type Constraints:\n{context.get('type_constraints')}\n")
    print("--- End Example ---")
