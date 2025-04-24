"""
Ambiguity Resolver Agent Module

This module implements an agent that identifies and resolves ambiguities
in natural language queries, helping to generate more accurate SQL.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AmbiguityType(Enum):
    """Types of ambiguities that can occur in queries."""

    COLUMN_REFERENCE = "column_reference"  # Which column in which table?
    TABLE_REFERENCE = "table_reference"  # Which table?
    AGGREGATION = "aggregation"  # How to aggregate? (COUNT, SUM, AVG, etc.)
    FILTER_VALUE = "filter_value"  # What value to filter on?
    JOIN_PATH = "join_path"  # What join path to use between tables?
    LIMIT = "limit"  # Should results be limited? How many?
    ORDER = "order"  # How to order results?
    GROUPING = "grouping"  # How to group results?
    SUBQUERY = "subquery"  # Is a subquery needed?
    TEMPORAL = "temporal"  # Time range ambiguity


@dataclass
class Ambiguity:
    """Represents an ambiguity in a query."""

    type: AmbiguityType
    description: str
    context: str
    options: List[str] = field(default_factory=list)
    resolved_option: Optional[str] = None
    confidence: float = 0.0
    impact: float = 0.0  # How much resolving this impacts the query correctness


@dataclass
class QueryClarification:
    """Represents a clarification request for a query."""

    question: str
    ambiguity: Ambiguity
    suggested_options: List[str] = field(default_factory=list)
    clarification_response: Optional[str] = None


class AmbiguityResolverAgent:
    """
    An agent that identifies and resolves ambiguities in natural language queries.
    """

    def __init__(
        self,
        schema_analyzer=None,
        confidence_threshold: float = 0.8,
        max_clarifications: int = 3,
    ):
        """
        Initialize the AmbiguityResolverAgent.

        Args:
            schema_analyzer: Schema analyzer to use for schema information.
            confidence_threshold (float): Threshold for confidence to avoid clarification.
            max_clarifications (int): Maximum number of clarifications to ask.
        """
        self.schema_analyzer = schema_analyzer
        self.confidence_threshold = confidence_threshold
        self.max_clarifications = max_clarifications
        # Common date/time terms and their standard formats
        self.time_formats = {
            "today": "CURRENT_DATE",
            "yesterday": "CURRENT_DATE - INTERVAL '1 day'",
            "tomorrow": "CURRENT_DATE + INTERVAL '1 day'",
            "last week": "CURRENT_DATE - INTERVAL '7 days'",
            "next week": "CURRENT_DATE + INTERVAL '7 days'",
            "last month": "CURRENT_DATE - INTERVAL '1 month'",
            "next month": "CURRENT_DATE + INTERVAL '1 month'",
            "last year": "CURRENT_DATE - INTERVAL '1 year'",
            "next year": "CURRENT_DATE + INTERVAL '1 year'",
            "this week": "DATE_TRUNC('week', CURRENT_DATE)",
            "this month": "DATE_TRUNC('month', CURRENT_DATE)",
            "this year": "DATE_TRUNC('year', CURRENT_DATE)",
        }
        # Common aggregate functions
        self.aggregation_functions = [
            "COUNT",
            "SUM",
            "AVG",
            "MIN",
            "MAX",
            "MEDIAN",
            "STDDEV",
        ]

    def identify_ambiguities(
        self, query_text: str, db_name: str, detected_tables: List[str]
    ) -> List[Ambiguity]:
        """
        Identify ambiguities in a natural language query.

        Args:
            query_text (str): Natural language query.
            db_name (str): Database name.
            detected_tables (List[str]): Tables detected in the query.

        Returns:
            List of identified ambiguities.
        """
        ambiguities = []

        # Check if schema analyzer is available
        if self.schema_analyzer is None:
            logger.warning("No schema analyzer provided. Limited ambiguity detection.")
            return ambiguities

        # Get schema information
        schema_info = self.schema_analyzer.analyze_schema(db_name)

        # 1. Check for column reference ambiguities
        column_ambiguities = self._identify_column_ambiguities(
            query_text, schema_info, detected_tables
        )
        ambiguities.extend(column_ambiguities)

        # 2. Check for join path ambiguities
        if len(detected_tables) > 1:
            join_ambiguities = self._identify_join_ambiguities(
                query_text, schema_info, detected_tables
            )
            ambiguities.extend(join_ambiguities)

        # 3. Check for aggregation ambiguities
        aggregation_ambiguities = self._identify_aggregation_ambiguities(
            query_text, schema_info
        )
        ambiguities.extend(aggregation_ambiguities)

        # 4. Check for temporal ambiguities
        temporal_ambiguities = self._identify_temporal_ambiguities(query_text)
        ambiguities.extend(temporal_ambiguities)

        # 5. Check for ordering/grouping ambiguities
        ordering_ambiguities = self._identify_ordering_ambiguities(
            query_text, schema_info, detected_tables
        )
        ambiguities.extend(ordering_ambiguities)

        return ambiguities

    def _identify_column_ambiguities(
        self, query_text: str, schema_info, detected_tables: List[str]
    ) -> List[Ambiguity]:
        """
        Identify column reference ambiguities.

        Args:
            query_text (str): Natural language query.
            schema_info: Schema information.
            detected_tables (List[str]): Tables detected in the query.

        Returns:
            List of column reference ambiguities.
        """
        ambiguities = []

        # Get all column names across detected tables
        column_names = set()
        columns_by_name = {}
        for table_name in detected_tables:
            if table_name in schema_info.tables:
                for column_name in schema_info.tables[table_name].columns:
                    column_names.add(column_name)
                    if column_name not in columns_by_name:
                        columns_by_name[column_name] = []
                    columns_by_name[column_name].append(table_name)

        # Look for common attributes that might be ambiguous
        common_attributes = ["id", "name", "date", "status", "type", "amount"]
        for attr in common_attributes:
            matching_tables = []
            for table_name in detected_tables:
                if (
                    table_name in schema_info.tables
                    and attr in schema_info.tables[table_name].columns
                ):
                    matching_tables.append(table_name)

            if len(matching_tables) > 1 and attr.lower() in query_text.lower():
                options = [f"{table}.{attr}" for table in matching_tables]
                ambiguities.append(
                    Ambiguity(
                        type=AmbiguityType.COLUMN_REFERENCE,
                        description=f"The column '{attr}' exists in multiple tables",
                        context=f"In query: '{query_text}'",
                        options=options,
                        confidence=0.5,
                        impact=0.8,
                    )
                )

        # Check for columns with same name in different tables
        for column_name, tables in columns_by_name.items():
            if len(tables) > 1 and column_name.lower() in query_text.lower():
                options = [f"{table}.{column_name}" for table in tables]
                ambiguities.append(
                    Ambiguity(
                        type=AmbiguityType.COLUMN_REFERENCE,
                        description=f"The column '{column_name}' exists in multiple tables",
                        context=f"In query: '{query_text}'",
                        options=options,
                        confidence=0.5,
                        impact=0.8,
                    )
                )

        return ambiguities

    def _identify_join_ambiguities(
        self, query_text: str, schema_info, detected_tables: List[str]
    ) -> List[Ambiguity]:
        """
        Identify join path ambiguities.

        Args:
            query_text (str): Natural language query.
            schema_info: Schema information.
            detected_tables (List[str]): Tables detected in the query.

        Returns:
            List of join path ambiguities.
        """
        ambiguities = []

        # Check if there are multiple possible join paths
        if len(detected_tables) > 2:
            # This would use schema information to find different join paths
            # For now, we'll simulate a join ambiguity
            options = [
                "Join path 1: Table1-Table2-Table3",
                "Join path 2: Table1-Table3-Table2",
            ]
            ambiguities.append(
                Ambiguity(
                    type=AmbiguityType.JOIN_PATH,
                    description="Multiple possible join paths detected",
                    context=f"For tables: {', '.join(detected_tables)}",
                    options=options,
                    confidence=0.6,
                    impact=0.9,
                )
            )

        return ambiguities

    def _identify_aggregation_ambiguities(
        self, query_text: str, schema_info
    ) -> List[Ambiguity]:
        """
        Identify aggregation ambiguities.

        Args:
            query_text (str): Natural language query.
            schema_info: Schema information.

        Returns:
            List of aggregation ambiguities.
        """
        ambiguities = []

        # Check for aggregation terms
        aggregation_terms = {
            "average": "AVG",
            "mean": "AVG",
            "total": "SUM",
            "sum": "SUM",
            "count": "COUNT",
            "how many": "COUNT",
            "maximum": "MAX",
            "highest": "MAX",
            "minimum": "MIN",
            "lowest": "MIN",
        }

        for term, agg_func in aggregation_terms.items():
            if term.lower() in query_text.lower():
                # Check if the aggregation target is ambiguous
                if "of" in query_text.lower() or "for" in query_text.lower():
                    # The target might be specified
                    confidence = 0.7
                else:
                    # No clear target for aggregation
                    confidence = 0.4
                    ambiguities.append(
                        Ambiguity(
                            type=AmbiguityType.AGGREGATION,
                            description=f"Ambiguous target for {agg_func} aggregation",
                            context=f"In query: '{query_text}'",
                            options=self.aggregation_functions,
                            confidence=confidence,
                            impact=0.7,
                        )
                    )

        return ambiguities

    def _identify_temporal_ambiguities(self, query_text: str) -> List[Ambiguity]:
        """
        Identify temporal ambiguities.

        Args:
            query_text (str): Natural language query.

        Returns:
            List of temporal ambiguities.
        """
        ambiguities = []

        # Check for relative time references
        relative_time_patterns = [
            r"recent(?:ly)?",
            r"last (?:few|couple of|several)?",
            r"past",
            r"previous",
            r"upcoming",
            r"next",
            r"soon",
            r"current(?:ly)?",
        ]

        for pattern in relative_time_patterns:
            if re.search(pattern, query_text, re.IGNORECASE):
                options = [
                    "Last 7 days",
                    "Last 30 days",
                    "Last 90 days",
                    "Last 365 days",
                    "Current month",
                    "Current year",
                ]
                ambiguities.append(
                    Ambiguity(
                        type=AmbiguityType.TEMPORAL,
                        description="Ambiguous time reference detected",
                        context=f"Matched pattern: '{pattern}' in query: '{query_text}'",
                        options=options,
                        confidence=0.3,
                        impact=0.8,
                    )
                )
                break  # One temporal ambiguity is enough

        return ambiguities

    def _identify_ordering_ambiguities(
        self, query_text: str, schema_info, detected_tables: List[str]
    ) -> List[Ambiguity]:
        """
        Identify ordering/grouping ambiguities.

        Args:
            query_text (str): Natural language query.
            schema_info: Schema information.
            detected_tables (List[str]): Tables detected in the query.

        Returns:
            List of ordering/grouping ambiguities.
        """
        ambiguities = []

        # Check for ordering terms
        ordering_terms = [
            "sort",
            "order",
            "arrange",
            "rank",
            "top",
            "bottom",
            "highest",
            "lowest",
            "best",
            "worst",
        ]

        for term in ordering_terms:
            if term.lower() in query_text.lower():
                # Check if direction is specified
                if any(
                    dir_term in query_text.lower()
                    for dir_term in ["ascending", "descending", "asc", "desc"]
                ):
                    # Direction is specified
                    confidence = 0.8
                else:
                    # No clear direction
                    ambiguities.append(
                        Ambiguity(
                            type=AmbiguityType.ORDER,
                            description="Ambiguous ordering direction",
                            context=f"Ordering term: '{term}' in query: '{query_text}'",
                            options=["ASC (ascending)", "DESC (descending)"],
                            confidence=0.5,
                            impact=0.5,
                        )
                    )

                # Check if ordering column is clear
                column_mentioned = False
                for table_name in detected_tables:
                    if table_name in schema_info.tables:
                        for column_name in schema_info.tables[table_name].columns:
                            if column_name.lower() in query_text.lower():
                                # Ordering column might be specified
                                column_mentioned = True
                                break

                if not column_mentioned:
                    # No clear column for ordering
                    # Suggest common ordering columns based on detected tables
                    options = []
                    for table_name in detected_tables:
                        if table_name in schema_info.tables:
                            for column_name in schema_info.tables[table_name].columns:
                                if any(
                                    pattern in column_name.lower()
                                    for pattern in [
                                        "date",
                                        "time",
                                        "id",
                                        "name",
                                        "price",
                                        "amount",
                                        "score",
                                        "rating",
                                    ]
                                ):
                                    options.append(f"{table_name}.{column_name}")

                    if options:
                        ambiguities.append(
                            Ambiguity(
                                type=AmbiguityType.ORDER,
                                description="Ambiguous ordering column",
                                context=f"Ordering term: '{term}' in query: '{query_text}'",
                                options=options[:5],  # Limit to 5 suggestions
                                confidence=0.4,
                                impact=0.6,
                            )
                        )

                break  # One ordering ambiguity check is enough

        return ambiguities

    def resolve_ambiguities(
        self, ambiguities: List[Ambiguity], context: Dict[str, any] = None
    ) -> List[Ambiguity]:
        """
        Attempt to resolve ambiguities automatically based on context.

        Args:
            ambiguities (List[Ambiguity]): List of ambiguities to resolve.
            context (Dict[str, any]): Additional context to help resolve ambiguities.

        Returns:
            Updated list of ambiguities with resolutions when possible.
        """
        if context is None:
            context = {}

        for ambiguity in ambiguities:
            # Skip already resolved ambiguities
            if ambiguity.resolved_option is not None:
                continue

            # Try to resolve based on ambiguity type
            if ambiguity.type == AmbiguityType.COLUMN_REFERENCE:
                self._resolve_column_ambiguity(ambiguity, context)
            elif ambiguity.type == AmbiguityType.JOIN_PATH:
                self._resolve_join_ambiguity(ambiguity, context)
            elif ambiguity.type == AmbiguityType.AGGREGATION:
                self._resolve_aggregation_ambiguity(ambiguity, context)
            elif ambiguity.type == AmbiguityType.TEMPORAL:
                self._resolve_temporal_ambiguity(ambiguity, context)
            elif ambiguity.type == AmbiguityType.ORDER:
                self._resolve_ordering_ambiguity(ambiguity, context)

        return ambiguities

    def _resolve_column_ambiguity(
        self, ambiguity: Ambiguity, context: Dict[str, any]
    ) -> None:
        """
        Attempt to resolve column reference ambiguity.

        Args:
            ambiguity (Ambiguity): Column reference ambiguity.
            context (Dict[str, any]): Additional context.
        """
        # Check if we have previous query patterns
        query_patterns = context.get("query_patterns", [])
        if query_patterns:
            # Look for common column usage in similar queries
            column_usage = {}
            for pattern in query_patterns:
                for option in ambiguity.options:
                    if option in pattern:
                        column_usage[option] = column_usage.get(option, 0) + 1

            if column_usage:
                # Choose most common option
                most_common = max(column_usage.items(), key=lambda x: x[1])
                ambiguity.resolved_option = most_common[0]
                ambiguity.confidence = min(0.7 + (most_common[1] / 10), 0.9)
                return

        # Check if there's a primary table
        primary_table = context.get("primary_table")
        if primary_table:
            for option in ambiguity.options:
                if option.startswith(f"{primary_table}."):
                    ambiguity.resolved_option = option
                    ambiguity.confidence = 0.8
                    return

        # Fallback: choose option based on table centrality in the schema
        # (If we had full schema information, we could do better here)
        if ambiguity.options:
            # For now, just choose the first option
            ambiguity.resolved_option = ambiguity.options[0]
            ambiguity.confidence = 0.6

    def _resolve_join_ambiguity(
        self, ambiguity: Ambiguity, context: Dict[str, any]
    ) -> None:
        """
        Attempt to resolve join path ambiguity.

        Args:
            ambiguity (Ambiguity): Join path ambiguity.
            context (Dict[str, any]): Additional context.
        """
        # For now, prefer direct joins over complex paths
        for option in ambiguity.options:
            if "Join path" in option and "1" in option:  # Choose first join path
                ambiguity.resolved_option = option
                ambiguity.confidence = 0.7
                return

        # Fallback
        if ambiguity.options:
            ambiguity.resolved_option = ambiguity.options[0]
            ambiguity.confidence = 0.6

    def _resolve_aggregation_ambiguity(
        self, ambiguity: Ambiguity, context: Dict[str, any]
    ) -> None:
        """
        Attempt to resolve aggregation ambiguity.

        Args:
            ambiguity (Ambiguity): Aggregation ambiguity.
            context (Dict[str, any]): Additional context.
        """
        query_text = context.get("query_text", "")

        # Try to infer from query text
        if "average" in query_text.lower() or "mean" in query_text.lower():
            ambiguity.resolved_option = "AVG"
            ambiguity.confidence = 0.8
        elif "total" in query_text.lower() or "sum" in query_text.lower():
            ambiguity.resolved_option = "SUM"
            ambiguity.confidence = 0.8
        elif (
            "count" in query_text.lower()
            or "how many" in query_text.lower()
            or "number of" in query_text.lower()
        ):
            ambiguity.resolved_option = "COUNT"
            ambiguity.confidence = 0.8
        elif (
            "maximum" in query_text.lower()
            or "highest" in query_text.lower()
            or "largest" in query_text.lower()
        ):
            ambiguity.resolved_option = "MAX"
            ambiguity.confidence = 0.8
        elif (
            "minimum" in query_text.lower()
            or "lowest" in query_text.lower()
            or "smallest" in query_text.lower()
        ):
            ambiguity.resolved_option = "MIN"
            ambiguity.confidence = 0.8
        else:
            # Default to COUNT for most scenarios
            ambiguity.resolved_option = "COUNT"
            ambiguity.confidence = 0.6

    def _resolve_temporal_ambiguity(
        self, ambiguity: Ambiguity, context: Dict[str, any]
    ) -> None:
        """
        Attempt to resolve temporal ambiguity.

        Args:
            ambiguity (Ambiguity): Temporal ambiguity.
            context (Dict[str, any]): Additional context.
        """
        query_text = context.get("query_text", "")

        # Try to infer from query text
        if "recent" in query_text.lower() or "recently" in query_text.lower():
            ambiguity.resolved_option = "Last 7 days"
            ambiguity.confidence = 0.7
        elif "last week" in query_text.lower():
            ambiguity.resolved_option = "Last 7 days"
            ambiguity.confidence = 0.9
        elif "last month" in query_text.lower():
            ambiguity.resolved_option = "Last 30 days"
            ambiguity.confidence = 0.9
        elif "last year" in query_text.lower():
            ambiguity.resolved_option = "Last 365 days"
            ambiguity.confidence = 0.9
        elif (
            "this month" in query_text.lower() or "current month" in query_text.lower()
        ):
            ambiguity.resolved_option = "Current month"
            ambiguity.confidence = 0.9
        elif "this year" in query_text.lower() or "current year" in query_text.lower():
            ambiguity.resolved_option = "Current year"
            ambiguity.confidence = 0.9
        else:
            # Default to last 30 days for most temporal queries
            ambiguity.resolved_option = "Last 30 days"
            ambiguity.confidence = 0.6

    def _resolve_ordering_ambiguity(
        self, ambiguity: Ambiguity, context: Dict[str, any]
    ) -> None:
        """
        Attempt to resolve ordering ambiguity.

        Args:
            ambiguity (Ambiguity): Ordering ambiguity.
            context (Dict[str, any]): Additional context.
        """
        query_text = context.get("query_text", "")

        # Check if this is a direction ambiguity
        if "direction" in ambiguity.description.lower():
            if (
                "ascending" in query_text.lower()
                or "increasing" in query_text.lower()
                or "lowest to highest" in query_text.lower()
            ):
                ambiguity.resolved_option = "ASC (ascending)"
                ambiguity.confidence = 0.9
            elif (
                "descending" in query_text.lower()
                or "decreasing" in query_text.lower()
                or "highest to lowest" in query_text.lower()
            ):
                ambiguity.resolved_option = "DESC (descending)"
                ambiguity.confidence = 0.9
            else:
                # Default to descending for most ranking queries
                ambiguity.resolved_option = "DESC (descending)"
                ambiguity.confidence = 0.7
        # Check if this is a column ambiguity
        elif "column" in ambiguity.description.lower():
            # Try to infer from query text and options
            for option in ambiguity.options:
                if "date" in option.lower() and any(
                    term in query_text.lower()
                    for term in ["recent", "latest", "newest", "oldest"]
                ):
                    ambiguity.resolved_option = option
                    ambiguity.confidence = 0.8
                    return
                elif "price" in option.lower() and any(
                    term in query_text.lower()
                    for term in ["expensive", "cheap", "cost", "price"]
                ):
                    ambiguity.resolved_option = option
                    ambiguity.confidence = 0.8
                    return
                elif "name" in option.lower() and any(
                    term in query_text.lower()
                    for term in ["alphabetical", "name", "called"]
                ):
                    ambiguity.resolved_option = option
                    ambiguity.confidence = 0.8
                    return

            # No clear match, choose a reasonable default
            for option in ambiguity.options:
                if "date" in option.lower():
                    ambiguity.resolved_option = option
                    ambiguity.confidence = 0.7
                    return

            # Fallback
            if ambiguity.options:
                ambiguity.resolved_option = ambiguity.options[0]
                ambiguity.confidence = 0.6

    def generate_clarification_questions(
        self, ambiguities: List[Ambiguity], max_questions: int = 3
    ) -> List[QueryClarification]:
        """
        Generate clarification questions for unresolved ambiguities.

        Args:
            ambiguities (List[Ambiguity]): List of ambiguities.
            max_questions (int): Maximum number of questions to generate.

        Returns:
            List of clarification questions.
        """
        # Filter ambiguities that need clarification
        need_clarification = [
            a
            for a in ambiguities
            if a.confidence < self.confidence_threshold and a.impact >= 0.5
        ]

        # Sort by impact (highest first)
        need_clarification.sort(key=lambda a: a.impact, reverse=True)

        # Generate clarification questions
        clarifications = []
        for ambiguity in need_clarification[:max_questions]:
            question = self._generate_question_for_ambiguity(ambiguity)
            clarifications.append(
                QueryClarification(
                    question=question,
                    ambiguity=ambiguity,
                    suggested_options=ambiguity.options[:5],  # Limit to 5 options
                )
            )

        return clarifications

    def _generate_question_for_ambiguity(self, ambiguity: Ambiguity) -> str:
        """
        Generate a human-readable question for an ambiguity.

        Args:
            ambiguity (Ambiguity): Ambiguity to generate question for.

        Returns:
            Human-readable question.
        """
        if ambiguity.type == AmbiguityType.COLUMN_REFERENCE:
            return "I found multiple columns named similarly to what you referenced. Which one did you mean?"

        elif ambiguity.type == AmbiguityType.JOIN_PATH:
            return "There are multiple ways to join the tables you mentioned. Which join path would be most appropriate?"

        elif ambiguity.type == AmbiguityType.AGGREGATION:
            return "How would you like to aggregate the data? (e.g., COUNT, SUM, AVG)"

        elif ambiguity.type == AmbiguityType.TEMPORAL:
            return "What time period are you interested in?"

        elif ambiguity.type == AmbiguityType.ORDER:
            if "direction" in ambiguity.description.lower():
                return (
                    "How would you like the results ordered? (ascending or descending)"
                )
            else:
                return "Which field would you like to order the results by?"

        else:
            return f"Could you clarify the following: {ambiguity.description}?"

    def update_from_clarification(
        self, clarification: QueryClarification, response: str
    ) -> None:
        """
        Update ambiguity resolution based on user clarification.

        Args:
            clarification (QueryClarification): QueryClarification containing the ambiguity.
            response (str): User's response to clarification.
        """
        clarification.clarification_response = response
        ambiguity = clarification.ambiguity

        # Check if response matches any of the suggested options
        for option in ambiguity.options:
            if option.lower() in response.lower():
                ambiguity.resolved_option = option
                ambiguity.confidence = 1.0
                return

        # No direct match, try to interpret the response
        ambiguity.resolved_option = response  # Use response directly
        ambiguity.confidence = 0.8  # Assume fairly high confidence since user responded

    def apply_resolved_ambiguities(
        self, sql_template: str, ambiguities: List[Ambiguity]
    ) -> str:
        """
        Apply resolved ambiguities to a SQL template.

        Args:
            sql_template (str): SQL template with placeholders.
            ambiguities (List[Ambiguity]): List of resolved ambiguities.

        Returns:
            Updated SQL with ambiguities resolved.
        """
        updated_sql = sql_template

        # Replace placeholders based on ambiguity type
        for ambiguity in ambiguities:
            if ambiguity.resolved_option is None:
                continue

            if ambiguity.type == AmbiguityType.COLUMN_REFERENCE:
                # Example placeholder: {{COLUMN:column_name}}
                placeholder = f"{{{{COLUMN:{ambiguity.description}}}}}"
                updated_sql = updated_sql.replace(
                    placeholder, ambiguity.resolved_option
                )

            elif ambiguity.type == AmbiguityType.JOIN_PATH:
                # Example placeholder: {{JOIN_PATH:tables}}
                placeholder = f"{{{{JOIN_PATH:{ambiguity.description}}}}}"
                # Extract the actual join condition from the option
                join_condition = ambiguity.resolved_option.split(": ", 1)[1]
                updated_sql = updated_sql.replace(placeholder, join_condition)

            elif ambiguity.type == AmbiguityType.AGGREGATION:
                # Example placeholder: {{AGGREGATION:target}}
                placeholder = f"{{{{AGGREGATION:{ambiguity.description}}}}}"
                updated_sql = updated_sql.replace(
                    placeholder, ambiguity.resolved_option
                )

            elif ambiguity.type == AmbiguityType.TEMPORAL:
                # Example placeholder: {{TEMPORAL:reference}}
                placeholder = f"{{{{TEMPORAL:{ambiguity.description}}}}}"
                # Convert friendly terms to SQL
                sql_date = self._convert_temporal_to_sql(ambiguity.resolved_option)
                updated_sql = updated_sql.replace(placeholder, sql_date)

            elif ambiguity.type == AmbiguityType.ORDER:
                if "direction" in ambiguity.description.lower():
                    # Example placeholder: {{ORDER_DIRECTION:column}}
                    placeholder = f"{{{{ORDER_DIRECTION:{ambiguity.description}}}}}"
                    # Extract the direction (ASC/DESC)
                    direction = "ASC" if "ASC" in ambiguity.resolved_option else "DESC"
                    updated_sql = updated_sql.replace(placeholder, direction)
                else:
                    # Example placeholder: {{ORDER_COLUMN:description}}
                    placeholder = f"{{{{ORDER_COLUMN:{ambiguity.description}}}}}"
                    updated_sql = updated_sql.replace(
                        placeholder, ambiguity.resolved_option
                    )

        return updated_sql

    def _convert_temporal_to_sql(self, temporal_option: str) -> str:
        """
        Convert a temporal option to a SQL expression.

        Args:
            temporal_option (str): Temporal option (e.g., 'Last 7 days').

        Returns:
            SQL expression for the temporal option.
        """
        # Check if it's in our predefined formats
        if temporal_option.lower() in self.time_formats:
            return self.time_formats[temporal_option.lower()]

        # Handle other common formats
        if "last" in temporal_option.lower():
            if "7 days" in temporal_option.lower() or "week" in temporal_option.lower():
                return "CURRENT_DATE - INTERVAL '7 days'"
            elif (
                "30 days" in temporal_option.lower()
                or "month" in temporal_option.lower()
            ):
                return "CURRENT_DATE - INTERVAL '30 days'"
            elif (
                "90 days" in temporal_option.lower()
                or "3 months" in temporal_option.lower()
            ):
                return "CURRENT_DATE - INTERVAL '90 days'"
            elif (
                "365 days" in temporal_option.lower()
                or "year" in temporal_option.lower()
            ):
                return "CURRENT_DATE - INTERVAL '365 days'"
        elif "current" in temporal_option.lower():
            if "month" in temporal_option.lower():
                return "DATE_TRUNC('month', CURRENT_DATE)"
            elif "year" in temporal_option.lower():
                return "DATE_TRUNC('year', CURRENT_DATE)"

        # Default to current date
        return "CURRENT_DATE"


if __name__ == "__main__":
    # Example usage
    resolver = AmbiguityResolverAgent()

    # Sample query
    query = "Show me the average order amount for recent orders"
    detected_tables = ["orders", "order_items", "customers"]

    # Identify ambiguities
    ambiguities = resolver.identify_ambiguities(query, "e_commerce", detected_tables)

    # Print identified ambiguities
    print(f"Identified {len(ambiguities)} ambiguities:")
    for a in ambiguities:
        print(
            f"- {a.type.value}: {a.description} (confidence: {a.confidence}, impact: {a.impact})"
        )
        if a.options:
            print(f"  Options: {', '.join(a.options[:3])}...")

    # Resolve ambiguities
    context = {"query_text": query, "primary_table": "orders"}
    resolved = resolver.resolve_ambiguities(ambiguities, context)

    # Print resolved ambiguities
    print("\nResolved ambiguities:")
    for a in resolved:
        if a.resolved_option:
            print(f"- {a.type.value}: {a.resolved_option} (confidence: {a.confidence})")

    # Generate clarification questions for low-confidence resolutions
    clarifications = resolver.generate_clarification_questions(resolved)

    # Print clarification questions
    if clarifications:
        print("\nClarification questions:")
        for c in clarifications:
            print(f"- {c.question}")
            if c.suggested_options:
                print(f"  Suggested options: {', '.join(c.suggested_options)}")
