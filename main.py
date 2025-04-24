"""
Alb-SQL: Cross-Attention SQL Fabric

A schema-aware LLM prompting system with database execution semantics
for generating high-quality SQL from natural language queries.
"""

import json
import logging
import os
import pickle
import re
from typing import Any, Dict, List, Optional, Tuple

from agents.ambiguity_resolver import AmbiguityResolverAgent
from agents.schema_analyzer_agent import SchemaAnalyzerAgent
from core.adaptive_context_manager import AdaptiveContextManager
from core.execution_aware_trainer import ExecutionAwareTrainer
from core.schema_analogizer import SchemaAnalogizer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AlbSQL:
    """
    Main class for the Alb-SQL system. Integrates all components and provides
    a unified interface for generating SQL from natural language queries.
    """

    def __init__(
        self,
        model_name: str = "gpt-4",
        max_tokens: int = 4096,
        cache_dir: str = "neural_cache",
        db_connector=None,
    ):
        """
        Initialize the Alb-SQL system.

        Args:
            model_name: Name of the LLM model to use.
            max_tokens: Maximum tokens for context.
            cache_dir: Directory for caching data.
            db_connector: Database connector for executing queries.
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.cache_dir = cache_dir
        self.db_connector = db_connector

        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)

        # Initialize components
        self.schema_analyzer = SchemaAnalyzerAgent(
            db_connector=db_connector, cache_dir=cache_dir
        )

        self.schema_analogizer = SchemaAnalogizer()

        self.ambiguity_resolver = AmbiguityResolverAgent(
            schema_analyzer=self.schema_analyzer,
            confidence_threshold=0.75,
            max_clarifications=3,
        )

        self.context_manager = AdaptiveContextManager(
            max_tokens=max_tokens,
            schema_analyzer=self.schema_analyzer,
            min_schema_tokens=500,
            min_examples_tokens=200,
        )

        self.execution_trainer = ExecutionAwareTrainer(
            db_connector=db_connector,
            execution_timeout=30,
            max_rows_to_compare=1000,
        )

        # Load query patterns
        self.query_patterns = self._load_query_patterns()

        # Cache for schema embeddings
        self.schema_embeddings_cache = {}

        # Try to load schema embeddings if they exist
        self._load_schema_embeddings()

    def generate_sql(
        self,
        query_text: str,
        db_name: str,
        domain: Optional[str] = None,
        clarify_ambiguities: bool = True,
        execution_aware: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate SQL from a natural language query.

        Args:
            query_text: Natural language query.
            db_name: Database name.
            domain: Optional domain for domain-specific handling.
            clarify_ambiguities: Whether to clarify ambiguities.
            execution_aware: Whether to use execution-aware validation.

        Returns:
            Dictionary with generated SQL and additional information.
        """
        logger.info(f"Generating SQL for query: {query_text}")

        # 1. Detect tables mentioned in the query
        detected_tables = self._detect_tables(query_text, db_name, domain)
        logger.info(f"Detected tables: {detected_tables}")

        # 2. Identify and resolve ambiguities
        ambiguities = []
        if clarify_ambiguities:
            ambiguities = self.ambiguity_resolver.identify_ambiguities(
                query_text, db_name, detected_tables
            )

            # Attempt to resolve ambiguities automatically
            context = {
                "query_text": query_text,
                "primary_table": detected_tables[0] if detected_tables else None,
            }
            ambiguities = self.ambiguity_resolver.resolve_ambiguities(
                ambiguities, context
            )

            # Log ambiguities
            for ambiguity in ambiguities:
                if ambiguity.resolved_option:
                    logger.info(
                        f"Resolved ambiguity: {ambiguity.type.value} -> {ambiguity.resolved_option} (confidence: {ambiguity.confidence})"
                    )
                else:
                    logger.warning(
                        f"Unresolved ambiguity: {ambiguity.type.value} (impact: {ambiguity.impact})"
                    )

        # 3. Generate adaptive context
        adaptive_context = self.context_manager.generate_adaptive_context(
            query_text, db_name, detected_tables
        )

        # 4. Add domain-specific patterns if available
        if domain and domain in self.query_patterns.get("domain_specific_patterns", {}):
            domain_patterns = self.query_patterns["domain_specific_patterns"][domain]
            adaptive_context["domain_patterns"] = domain_patterns

        # 5. Generate schema-aware prompt
        prompt = self._generate_prompt(
            query_text, db_name, adaptive_context, ambiguities, domain
        )

        # 6. Generate SQL using the LLM
        sql_candidates = self._generate_sql_candidates(
            prompt, 3
        )  # Generate 3 candidates

        # 7. Validate and pick the best candidate
        if execution_aware and self.db_connector is not None:
            best_candidate, metrics = self._validate_candidates(sql_candidates, db_name)
            validation_info = {
                "candidates": len(sql_candidates),
                "metrics": metrics,
            }
        else:
            best_candidate = sql_candidates[0] if sql_candidates else ""
            validation_info = {"candidates": len(sql_candidates), "metrics": None}

        # 8. Return the result
        result = {
            "sql": best_candidate,
            "tables": detected_tables,
            "ambiguities": [
                {
                    "type": a.type.value,
                    "description": a.description,
                    "resolved": a.resolved_option,
                    "confidence": a.confidence,
                }
                for a in ambiguities
            ],
            "complexity": adaptive_context.get("complexity", "unknown"),
            "validation_info": validation_info,
            "domain": domain,
        }

        return result

    def _detect_tables(
        self, query_text: str, db_name: str, domain: Optional[str] = None
    ) -> List[str]:
        """
        Detect tables mentioned in the query.

        Args:
            query_text: Natural language query.
            db_name: Database name.
            domain: Optional domain for domain-specific handling.

        Returns:
            List of detected table names.
        """
        # Get schema information
        schema_info = self.schema_analyzer.analyze_schema(db_name)

        # Get all tables in the schema
        all_tables = list(schema_info.tables.keys())

        # Simple detection: check if table names appear in the query
        detected = []
        for table in all_tables:
            # Check for exact match or plural form
            if re.search(
                rf"\b{re.escape(table)}\b", query_text, re.IGNORECASE
            ) or re.search(rf"\b{re.escape(table)}s\b", query_text, re.IGNORECASE):
                detected.append(table)

        # If no tables detected explicitly, try to infer from domain knowledge
        if not detected and domain:
            # This would use domain knowledge to infer relevant tables
            # For now, we'll return a simple placeholder
            domain_tables = self._get_domain_tables(domain, all_tables)
            detected = domain_tables[:2]  # Limit to 2 tables

        # If still no tables detected, use the most connected tables
        if not detected:
            # Get table connectivity (number of relationships)
            connectivity = self._get_table_connectivity(schema_info)
            # Sort by connectivity (most connected first)
            sorted_tables = sorted(
                connectivity.items(), key=lambda x: x[1], reverse=True
            )
            # Take the most connected tables (up to 2)
            detected = [t[0] for t in sorted_tables[:2]]

        return detected

    def _get_domain_tables(self, domain: str, all_tables: List[str]) -> List[str]:
        """
        Get relevant tables for a domain.

        Args:
            domain: Domain name.
            all_tables: All available tables.

        Returns:
            List of relevant tables for the domain.
        """
        # This would use domain knowledge to determine relevant tables
        # For now, we'll use a simple mapping
        domain_table_mapping = {
            "e_commerce": ["users", "orders", "products", "categories"],
            "healthcare": ["patients", "doctors", "appointments", "medications"],
            "finance": ["accounts", "transactions", "customers", "loans"],
            "education": ["students", "courses", "enrollments", "grades"],
            "social_media": ["users", "posts", "comments", "likes"],
        }

        if domain in domain_table_mapping:
            # Filter available tables by domain relevance
            return [t for t in domain_table_mapping[domain] if t in all_tables]

        return []

    def _get_table_connectivity(self, schema_info) -> Dict[str, int]:
        """
        Calculate table connectivity based on relationships.

        Args:
            schema_info: Schema information.

        Returns:
            Dictionary mapping table names to connectivity scores.
        """
        connectivity = {}
        for table in schema_info.tables:
            connectivity[table] = 0

        # Count relationships for each table
        for src_table, src_col, dst_table, dst_col in schema_info.relationships:
            connectivity[src_table] = connectivity.get(src_table, 0) + 1
            connectivity[dst_table] = connectivity.get(dst_table, 0) + 1

        return connectivity

    def _generate_prompt(
        self,
        query_text: str,
        db_name: str,
        adaptive_context: Dict[str, Any],
        ambiguities: List[Any],
        domain: Optional[str] = None,
    ) -> str:
        """
        Generate a schema-aware prompt for the LLM.

        Args:
            query_text: Natural language query.
            db_name: Database name.
            adaptive_context: Adaptive context components.
            ambiguities: List of ambiguities.
            domain: Optional domain for domain-specific handling.

        Returns:
            Prompt string for the LLM.
        """
        # Build the prompt
        prompt_parts = [
            "**Role**: World-class SQL Engineer + Database Architect",
            f"**Task**: Solve {domain if domain else db_name} problem using {db_name}'s schema",
            f"\n**Question**:\n{query_text}",
            f"\n**Schema Context**:\n{adaptive_context.get('schema_summary', '')}",
        ]

        # Add resolved ambiguities if any
        resolved_ambiguities = [a for a in ambiguities if a.resolved_option is not None]
        if resolved_ambiguities:
            ambiguity_info = "\n**Resolved Ambiguities**:\n"
            for a in resolved_ambiguities:
                ambiguity_info += f"- {a.description}: {a.resolved_option}\n"
            prompt_parts.append(ambiguity_info)

        # Add type constraints
        if adaptive_context.get("type_constraints"):
            prompt_parts.append(
                f"\n**Type Constraints**:\n{adaptive_context['type_constraints']}"
            )

        # Add common mistakes
        if adaptive_context.get("common_mistakes"):
            prompt_parts.append(
                f"\n**Common Mistakes**:\n{adaptive_context['common_mistakes']}"
            )

        # Add examples
        if adaptive_context.get("examples"):
            prompt_parts.append(f"\n**Examples**:\n{adaptive_context['examples']}")

        # Add query patterns
        if adaptive_context.get("query_patterns"):
            prompt_parts.append(
                f"\n**Query Patterns**:\n{adaptive_context['query_patterns']}"
            )

        # Add reasoning chain
        if adaptive_context.get("reasoning_chain"):
            prompt_parts.append(
                f"\n**Reasoning Chain**:\n{adaptive_context['reasoning_chain']}"
            )

        # Add domain-specific patterns if available
        if adaptive_context.get("domain_patterns"):
            domain_info = "\n**Domain-Specific Patterns**:\n"
            for pattern in adaptive_context["domain_patterns"]:
                domain_info += (
                    f"- {pattern['description']}: {pattern['template_sql']}\n"
                )
            prompt_parts.append(domain_info)

        # Add response format
        prompt_parts.append(
            "\n**Response Format**:\n```sql\n/* Explanatory comments */\nSELECT ...\n```"
        )

        # Combine all parts
        prompt = "\n".join(prompt_parts)

        return prompt

    def _generate_sql_candidates(
        self, prompt: str, num_candidates: int = 3
    ) -> List[str]:
        """
        Generate SQL candidates from the prompt using the LLM.

        Args:
            prompt: Prompt for the LLM.
            num_candidates: Number of candidates to generate.

        Returns:
            List of SQL candidate strings.
        """
        # In a real implementation, this would call an LLM API
        # For now, we'll return a placeholder SQL query

        # Simulate generating multiple candidates
        candidates = []
        for i in range(num_candidates):
            # Generate increasingly complex candidates (for demonstration)
            if i == 0:
                # Simple candidate
                sql = "SELECT * FROM users WHERE id = 1"
            elif i == 1:
                # Medium complexity candidate
                sql = "SELECT u.name, o.order_date FROM users u JOIN orders o ON u.id = o.user_id WHERE u.id = 1"
            else:
                # Complex candidate
                sql = "SELECT u.name, COUNT(o.id) as order_count FROM users u JOIN orders o ON u.id = o.user_id GROUP BY u.name HAVING COUNT(o.id) > 5 ORDER BY order_count DESC"

            candidates.append(sql)

        return candidates

    def _validate_candidates(
        self, sql_candidates: List[str], db_name: str
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Validate SQL candidates and pick the best one.

        Args:
            sql_candidates: List of SQL candidate strings.
            db_name: Database name.

        Returns:
            Tuple of (best_candidate, metrics).
        """
        if not sql_candidates:
            return "", {}

        # If we don't have a database connector for execution-aware validation
        if self.db_connector is None:
            return sql_candidates[0], {}

        # Track metrics for each candidate
        metrics = []

        try:
            # For each candidate, execute and collect metrics
            for i, sql in enumerate(sql_candidates):
                # Run evaluation with the execution trainer
                reference_sql = sql_candidates[0]  # Use first candidate as reference
                eval_metrics = self.execution_trainer.evaluate_query(
                    sql, reference_sql, db_name
                )

                # Store metrics
                metrics.append(
                    {
                        "candidate_id": i,
                        "sql": sql,
                        "success": eval_metrics.execution_success,
                        "syntax_correctness": eval_metrics.syntax_correctness,
                        "semantic_correctness": eval_metrics.semantic_correctness,
                        "result_match": eval_metrics.result_match_score,
                        "efficiency": eval_metrics.execution_efficiency,
                        "overall_score": eval_metrics.overall_score,
                    }
                )

            # Sort by overall score (highest first)
            metrics.sort(key=lambda x: x["overall_score"], reverse=True)

            # Select the best candidate
            best_candidate = metrics[0]["sql"] if metrics else sql_candidates[0]

            return best_candidate, {"candidate_metrics": metrics}

        except Exception as e:
            logger.error(f"Error during SQL validation: {e}")
            return sql_candidates[0], {"error": str(e)}

    def _load_query_patterns(self) -> Dict[str, Any]:
        """
        Load query patterns from JSON file.

        Returns:
            Dictionary containing query patterns.
        """
        patterns_path = os.path.join(self.cache_dir, "query_patterns.json")

        try:
            if os.path.exists(patterns_path):
                with open(patterns_path, "r") as f:
                    patterns = json.load(f)
                logger.info(f"Loaded query patterns from {patterns_path}")
                return patterns
            else:
                logger.warning(f"Query patterns file not found: {patterns_path}")
                return {}
        except Exception as e:
            logger.error(f"Error loading query patterns: {e}")
            return {}

    def _load_schema_embeddings(self) -> None:
        """
        Load schema embeddings from cache.
        """
        embeddings_path = os.path.join(self.cache_dir, "schema_embeddings.pkl")

        try:
            if os.path.exists(embeddings_path):
                with open(embeddings_path, "rb") as f:
                    self.schema_embeddings_cache = pickle.load(f)
                # Pass embeddings to the schema analogizer
                self.schema_analogizer.schema_embeddings = self.schema_embeddings_cache
                logger.info(f"Loaded schema embeddings from {embeddings_path}")
            else:
                logger.info(f"Schema embeddings file not found: {embeddings_path}")
        except Exception as e:
            logger.error(f"Error loading schema embeddings: {e}")


def main():
    """
    Main function for standalone usage.
    """
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create AlbSQL instance
    alb_sql = AlbSQL()

    # Example query
    query = "Find users who placed orders in the last month"
    db_name = "e_commerce"

    # Generate SQL
    result = alb_sql.generate_sql(query, db_name)

    # Print result
    print("\nGenerated SQL:")
    print(result["sql"])

    print("\nDetected tables:", result["tables"])
    print("\nComplexity:", result["complexity"])


if __name__ == "__main__":
    main()
