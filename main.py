"""
Alb-SQL: Cross-Attention SQL Fabric

A schema-aware LLM prompting system with database execution semantics
for generating high-quality SQL from natural language queries.
"""

import argparse
import json
import logging
import os
import pickle
import re
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple

from agents.ambiguity_resolver import AmbiguityResolverAgent
from agents.schema_analyzer_agent import SchemaAnalyzerAgent
from core.adaptive_context_manager import AdaptiveContextManager
from core.execution_aware_trainer import ExecutionAwareTrainer
from core.llm_sql_generator import generate_sql_from_llm
from core.schema_analogizer import SchemaAnalogizer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# --- Helper Functions for BIRD Evaluation ---


def normalize_sql(sql: str) -> str:
    """Normalize SQL query for comparison (lowercase, remove comments, strip whitespace)."""
    if not isinstance(sql, str):
        return ""
    # Remove SQL comments
    sql = re.sub(r"--.*", "", sql)
    sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
    # Lowercase and strip whitespace
    return sql.lower().strip()


def execute_sql(db_path: str, sql: str) -> Tuple[Optional[List[Tuple]], Optional[str]]:
    """Execute SQL query on a SQLite database and return results or error."""
    if not sql:
        return None, "Empty SQL query"
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            results = cursor.fetchall()
            # Sort results for consistent comparison if results exist
            if results:
                try:
                    # Attempt to sort; handle potential unorderable types
                    results.sort()
                except TypeError:
                    logger.debug(f"Could not sort results for query: {sql[:100]}...")
            return results, None
    except sqlite3.Error as e:
        logger.debug(
            f"SQL Execution Error in {db_path} for query '{sql[:100]}...': {e}"
        )
        return None, str(e)
    except Exception as e:  # Catch other potential errors
        logger.error(f"Unexpected Error during SQL Execution in {db_path}: {e}")
        return None, str(e)


# --- End Helper Functions ---


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
            model_name (str): Name of the LLM model to use.
            max_tokens (int): Maximum tokens for context.
            cache_dir (str): Directory for caching data.
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

        # TODO: Consider adding a parameter to specify a BIRD-optimized model variant
        # e.g., self.model_name = model_name_for_bird if use_bird_config else model_name

    def generate_sql(  # noqa: C901 (Function too complex) - Consider refactoring
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
            query_text (str): Natural language query.
            db_name (str): Database name.
            domain (Optional[str]): Optional domain for domain-specific handling.
            clarify_ambiguities (bool): Whether to clarify ambiguities.
            execution_aware (bool): Whether to use execution-aware validation.

        Returns:
            Dictionary with generated SQL and additional information.
        """
        logger.info(
            f"Generating SQL for query: {query_text[:100]}..."
        )  # Limit log length

        # 1. Detect tables mentioned in the query
        detected_tables = self._detect_tables(query_text, db_name, domain)
        logger.info(f"Detected tables: {detected_tables}")

        # 2. Identify and resolve ambiguities
        ambiguities = []
        if clarify_ambiguities:
            try:
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
                            f"Resolved ambiguity: {ambiguity.type.value} -> "
                            f"{ambiguity.resolved_option} "
                            f"(confidence: {ambiguity.confidence:.2f})"
                        )
                    else:
                        logger.warning(
                            f"Unresolved ambiguity: {ambiguity.type.value} "
                            f"(impact: {ambiguity.impact})"
                        )
            except Exception as e:
                logger.error(f"Error during ambiguity resolution: {e}")
                ambiguities = []  # Continue without resolution if error occurs

        # 3. Generate adaptive context
        try:
            adaptive_context = self.context_manager.generate_adaptive_context(
                query_text, db_name, detected_tables
            )
        except Exception as e:
            logger.error(f"Error generating adaptive context: {e}")
            adaptive_context = {}  # Use empty context if error occurs

        # 4. Add domain-specific patterns if available
        domain_patterns_data = self.query_patterns.get("domain_specific_patterns", {})
        if domain and domain in domain_patterns_data:
            domain_patterns = domain_patterns_data[domain]
            adaptive_context["domain_patterns"] = domain_patterns

        # 5. Generate schema-aware prompt
        prompt = self._generate_prompt(
            query_text, db_name, adaptive_context, ambiguities, domain
        )

        # 6. Generate SQL using the LLM
        sql_candidates = self._generate_sql_candidates(
            prompt, query_text, 3
        )  # Generate 3 candidates, pass query_text for fallback

        # 7. Validate and pick the best candidate
        best_candidate = sql_candidates[0] if sql_candidates else ""
        validation_info = {"candidates": len(sql_candidates), "metrics": None}
        if execution_aware and self.db_connector is not None:
            try:
                best_candidate, metrics = self._validate_candidates(
                    sql_candidates, db_name
                )
                validation_info["metrics"] = metrics
            except Exception as e:
                logger.error(f"Error during candidate validation: {e}")
                # Keep the first candidate but note the validation error
                validation_info["error"] = f"Validation failed: {e}"

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
            query_text (str): Natural language query.
            db_name (str): Database name.
            domain (Optional[str]): Optional domain for domain-specific handling.

        Returns:
            List of detected table names.
        """
        # Get schema information
        try:
            schema_info = self.schema_analyzer.analyze_schema(db_name)
            if not schema_info or not schema_info.tables:
                logger.warning(
                    f"No schema info found or empty schema for DB: {db_name}"
                )
                return []
            all_tables = list(schema_info.tables.keys())
        except Exception as e:
            logger.error(f"Error analyzing schema for {db_name}: {e}")
            return []  # Return empty list if schema analysis fails

        # Simple detection: check if table names appear in the query
        detected = []
        for table in all_tables:
            # Check for exact match or plural form (case-insensitive)
            # Use word boundaries to avoid partial matches
            pattern = rf"\b({re.escape(table)}|{re.escape(table)}s)\b"
            if re.search(pattern, query_text, re.IGNORECASE):
                detected.append(table)

        # If no tables detected explicitly, try to infer from domain knowledge
        if not detected and domain:
            # This would use domain knowledge to infer relevant tables
            # For now, we'll return a simple placeholder
            domain_tables = self._get_domain_tables(domain, all_tables)
            detected = domain_tables[:2]  # Limit to 2 tables

        # If still no tables detected, use the most connected tables (if schema available)
        if not detected and schema_info:
            try:
                # Get table connectivity (number of relationships)
                connectivity = self._get_table_connectivity(schema_info)
                # Sort by connectivity (most connected first)
                sorted_tables = sorted(
                    connectivity.items(), key=lambda x: x[1], reverse=True
                )
                # Take the most connected tables (up to 2)
                detected = [t[0] for t in sorted_tables[:2]]
            except Exception as e:
                logger.error(f"Error calculating table connectivity for {db_name}: {e}")
                # Fallback: maybe return first table alphabetically? Or empty.
                if all_tables:
                    detected = all_tables[:1]  # Simple fallback

        return detected

    def _get_domain_tables(self, domain: str, all_tables: List[str]) -> List[str]:
        """
        Get relevant tables for a domain.

        Args:
            domain (str): Domain name.
            all_tables (List[str]): All available tables.

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
            "social_media": ["users", "posts", "comments", "likes", "followers"],
        }

        relevant_tables = domain_table_mapping.get(domain, [])
        if relevant_tables:
            # Filter available tables by domain relevance
            return [t for t in relevant_tables if t in all_tables]

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
        if hasattr(schema_info, "relationships") and schema_info.relationships:
            for relationship in schema_info.relationships:
                # Assuming relationship is a tuple/list like (src_tbl, src_col, dst_tbl, dst_col)
                if len(relationship) >= 3:
                    src_table, _, dst_table = (
                        relationship[0],
                        relationship[1],
                        relationship[2],
                    )
                    if src_table in connectivity:
                        connectivity[src_table] += 1
                    if dst_table in connectivity:
                        connectivity[dst_table] += 1
                else:
                    logger.warning(f"Unexpected relationship format: {relationship}")

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
            query_text (str): Natural language query.
            db_name (str): Database name.
            adaptive_context (Dict[str, Any]): Adaptive context components.
            ambiguities (List[Any]): List of ambiguities.
            domain (Optional[str]): Optional domain for domain-specific handling.

        Returns:
            Prompt string for the LLM.
        """
        # Build the prompt
        prompt_parts = [
            "**Role**: World-class SQL Engineer + Database Architect",
            f"**Task**: Solve the following problem for the '{domain if domain else db_name}' "
            f"domain using the '{db_name}' database schema.",
            f"\n**Question**:\n{query_text}",
            f"\n**Schema Context**:\n{adaptive_context.get('schema_summary', 'Not available.')}",
        ]

        # Add resolved ambiguities if any
        resolved_ambiguities = [
            a
            for a in ambiguities
            if hasattr(a, "resolved_option") and a.resolved_option is not None
        ]
        if resolved_ambiguities:
            ambiguity_info = "\n**Resolved Ambiguities**:"
            for a in resolved_ambiguities:
                # Ensure description and resolved_option are strings
                desc = str(getattr(a, "description", "N/A"))
                res_opt = str(getattr(a, "resolved_option", "N/A"))
                ambiguity_info += f"\n- {desc}: {res_opt}"
            prompt_parts.append(ambiguity_info)

        # Add other context parts safely
        context_keys = [
            ("type_constraints", "Type Constraints"),
            ("common_mistakes", "Common Mistakes"),
            ("examples", "Examples"),
            ("query_patterns", "Query Patterns"),
            ("reasoning_chain", "Reasoning Chain"),
        ]
        for key, title in context_keys:
            if adaptive_context.get(key):
                prompt_parts.append(f"\n**{title}**:\n{adaptive_context[key]}")

        # Add domain-specific patterns if available
        domain_patterns = adaptive_context.get("domain_patterns")
        if domain_patterns and isinstance(domain_patterns, list):
            domain_info = "\n**Domain-Specific Patterns**:"
            for pattern in domain_patterns:
                if isinstance(pattern, dict):
                    desc = pattern.get("description", "N/A")
                    sql = pattern.get("template_sql", "N/A")
                    domain_info += f"\n- {desc}: {sql}"
            prompt_parts.append(domain_info)

        # Add response format instruction
        prompt_parts.append(
            "\n**Response Format**: Provide only the SQL query, enclosed in ```sql ... ```."
        )

        # Combine all parts
        prompt = "\n".join(prompt_parts)

        return prompt

    def _generate_sql_candidates(
        self, prompt: str, query_text: str, num_candidates: int = 3
    ) -> List[str]:
        """
        Generate SQL candidates from the prompt using the LLM.

        Args:
            prompt (str): Prompt for the LLM.
            query_text (str): Original natural language query (used for fallback).
            num_candidates (int): Number of candidates to generate.

        Returns:
            List of SQL candidate strings.

        Args:
            prompt (str): Prompt for the LLM.
            num_candidates (int): Number of candidates to generate.

        Returns:
            List of SQL candidate strings.
        """
        # Use Hugging Face LLM to generate SQL
        try:
            # Extract question and schema context more robustly
            question_match = re.search(
                r"\*\*Question\*\*:\s*(.*?)\s*\n\*\*", prompt, re.DOTALL
            )
            question = (
                question_match.group(1).strip() if question_match else query_text
            )  # Fallback

            schema_context_match = re.search(
                r"\*\*Schema Context\*\*:\s*(.*?)\s*\n\*\*", prompt, re.DOTALL
            )
            schema_context = (
                schema_context_match.group(1).strip() if schema_context_match else None
            )

            # Generate SQL using the LLM (assuming generate_sql_from_llm exists)
            # Use self.model_name configured during initialization
            model_configs = {
                "model_name": self.model_name,
                "temperature": 0.3 if num_candidates == 1 else 0.7,
                "max_tokens": self.max_tokens,  # Use configured max_tokens
                "num_candidates": num_candidates,
                "clean_output": True,  # Assumes this extracts SQL from ```sql ... ```
            }

            logger.info(
                f"Generating SQL with {self.model_name} for question: {question[:50]}..."
            )

            candidates = generate_sql_from_llm(
                question=question, schema_context=schema_context, **model_configs
            )

            # Handle list or string return type
            if isinstance(candidates, str):
                candidates = [candidates]

            # Ensure we have the requested number of candidates, padding if necessary
            if len(candidates) < num_candidates:
                padding_needed = num_candidates - len(candidates)
                logger.warning(
                    f"Generated {len(candidates)} candidates, requested {num_candidates}. "
                    f"Padding with {'last candidate' if candidates else 'default'}."
                )
                padding_sql = candidates[-1] if candidates else "SELECT 1;"
                candidates.extend([padding_sql] * padding_needed)

            return candidates[:num_candidates]  # Return exactly the number requested

        except Exception as e:
            logger.error(f"Error generating SQL with {self.model_name}: {e}")
            # Fallback: return list of empty strings or simple SELECT
            return ["SELECT 'generation_error';"] * num_candidates

    def _validate_candidates(
        self, sql_candidates: List[str], db_name: str
    ) -> Tuple[str, Dict[str, Any]]:  # noqa: C901 (Function too complex)
        """
        Validate SQL candidates and pick the best one.

        Args:
            sql_candidates (List[str]): List of SQL candidate strings.
            db_name (str): Database name.

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
                try:
                    # Run evaluation with the execution trainer
                    # Using first candidate as reference might not be ideal, but simple
                    reference_sql = sql_candidates[0]
                    eval_metrics = self.execution_trainer.evaluate_query(
                        sql, reference_sql, db_name
                    )

                    # Store metrics safely using getattr
                    metrics.append(
                        {
                            "candidate_id": i,
                            "sql": sql,
                            "success": getattr(
                                eval_metrics, "execution_success", False
                            ),
                            "syntax_correctness": getattr(
                                eval_metrics, "syntax_correctness", 0.0
                            ),
                            "semantic_correctness": getattr(
                                eval_metrics, "semantic_correctness", 0.0
                            ),
                            "result_match": getattr(
                                eval_metrics, "result_match_score", 0.0
                            ),
                            "efficiency": getattr(
                                eval_metrics, "execution_efficiency", None
                            ),
                            "overall_score": getattr(
                                eval_metrics, "overall_score", 0.0
                            ),
                        }
                    )
                except Exception as eval_e:
                    logger.error(
                        f"Error evaluating candidate {i} for {db_name}: {eval_e}"
                    )
                    # Add placeholder metric indicating failure
                    metrics.append(
                        {
                            "candidate_id": i,
                            "sql": sql,
                            "overall_score": -1.0,
                            "error": str(eval_e),
                        }
                    )

            # Sort by overall score (highest first)
            metrics.sort(key=lambda x: x["overall_score"], reverse=True)

            # Select the best candidate
            best_candidate = metrics[0]["sql"] if metrics else sql_candidates[0]

            return best_candidate, {"candidate_metrics": metrics}

        except Exception as e:
            logger.error(f"Error during SQL validation loop: {e}")
            # Fallback: return first candidate if metrics failed, include error info
            return sql_candidates[0], {
                "error": f"Validation loop failed: {e}",
                "candidate_metrics": metrics,
            }

    def _load_query_patterns(self) -> Dict[str, Any]:
        """
        Load query patterns from JSON file.

        Returns:
            Dictionary containing query patterns.
        """
        patterns_path = os.path.join(self.cache_dir, "query_patterns.json")

        try:
            if not os.path.exists(patterns_path):
                logger.warning(f"Query patterns file not found: {patterns_path}")
                return {}
            with open(patterns_path, "r", encoding="utf-8") as f:
                patterns = json.load(f)
            logger.info(f"Loaded query patterns from {patterns_path}")
            return patterns
        except json.JSONDecodeError as e:
            logger.error(
                f"Error decoding query patterns JSON from {patterns_path}: {e}"
            )
            return {}
        except IOError as e:
            logger.error(f"Error reading query patterns file {patterns_path}: {e}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error loading query patterns: {e}")
            return {}

    def _load_schema_embeddings(self) -> None:
        """
        Load schema embeddings from cache.
        """
        embeddings_path = os.path.join(self.cache_dir, "schema_embeddings.pkl")

        try:
            if not os.path.exists(embeddings_path):
                logger.info(f"Schema embeddings file not found: {embeddings_path}")
                return
            with open(embeddings_path, "rb") as f:
                self.schema_embeddings_cache = pickle.load(f)
            # Pass embeddings to the schema analogizer if it exists
            if hasattr(self, "schema_analogizer") and self.schema_analogizer:
                self.schema_analogizer.schema_embeddings = self.schema_embeddings_cache
            logger.info(f"Loaded schema embeddings from {embeddings_path}")
        except pickle.UnpicklingError as e:
            logger.error(
                f"Error unpickling schema embeddings from {embeddings_path}: {e}"
            )
        except IOError as e:
            logger.error(f"Error reading schema embeddings file {embeddings_path}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error loading schema embeddings: {e}")


# --- BIRD Benchmark Pipeline Functions ---


def load_bird_data(
    data_dir: str,
    tables_file: str = "tables.json",
    data_file: str = "data.json",
    gold_sql_file: Optional[str] = None,
) -> Dict[str, Any]:
    """Loads BIRD dataset components (dev or train)."""
    logger.info(f"Attempting to load BIRD data from: {data_dir}")
    data = {"items": [], "tables": {}, "gold_sql": [], "db_base_path": ""}
    try:
        # Load main data file (e.g., dev.json, train.json)
        data_path = os.path.join(data_dir, data_file)
        if not os.path.exists(data_path):
            logger.error(f"Data file not found: {data_path}")
        else:
            with open(data_path, "r", encoding="utf-8") as f:
                data["items"] = json.load(f)
            logger.info(f"Loaded {len(data['items'])} items from {data_path}")

        # Load tables file (e.g., dev_tables.json, train_tables.json)
        tables_path = os.path.join(data_dir, tables_file)
        if not os.path.exists(tables_path):
            logger.warning(f"Tables file not found: {tables_path}")
        else:
            with open(tables_path, "r", encoding="utf-8") as f:
                data["tables"] = json.load(f)
            logger.info(f"Loaded table definitions from {tables_path}")

        # Load gold SQL file if provided (e.g., dev.sql, train_gold.sql)
        if gold_sql_file:
            gold_sql_path = os.path.join(data_dir, gold_sql_file)
            if not os.path.exists(gold_sql_path):
                logger.warning(f"Gold SQL file not found: {gold_sql_path}")
            else:
                with open(gold_sql_path, "r", encoding="utf-8") as f:
                    # One SQL query per line, matching the order in data['items']
                    data["gold_sql"] = [line.strip() for line in f if line.strip()]
                logger.info(
                    f"Loaded {len(data['gold_sql'])} gold SQL queries from {gold_sql_path}"
                )
                # Validation: Check if counts match (only if items were loaded)
                if data["items"] and len(data["items"]) != len(data["gold_sql"]):
                    logger.warning(
                        f"Item count ({len(data['items'])}) mismatch with gold SQL "
                        f"count ({len(data['gold_sql'])}) in {data_dir}"
                    )

        # Define expected database path relative to data_dir
        data["db_base_path"] = os.path.join(data_dir, "dev_databases")
        logger.info(f"Expecting databases in: {data['db_base_path']}")

    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON file in {data_dir}: {e}")
    except IOError as e:
        logger.error(f"Error reading file in {data_dir}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error loading BIRD data from {data_dir}: {e}")

    return data


def evaluate_bird(
    alb_sql_instance: AlbSQL, dev_data: Dict[str, Any], limit: Optional[int] = None
):  # noqa: C901 (Function too complex)
    """Evaluates the AlbSQL pipeline on the BIRD dev set."""
    logger.info("Starting BIRD evaluation...")

    items_to_evaluate = dev_data.get("items", [])
    gold_sql_list = dev_data.get("gold_sql", [])
    db_base_path = dev_data.get("db_base_path")

    if not items_to_evaluate or not gold_sql_list or not db_base_path:
        logger.error(
            "Dev data is incomplete (items, gold_sql, or db_base_path missing). Cannot run evaluation."
        )
        return

    if len(items_to_evaluate) != len(gold_sql_list):
        logger.error(
            f"Dev items count ({len(items_to_evaluate)}) differs from gold SQL count "
            f"({len(gold_sql_list)}). Aborting evaluation."
        )
        return

    if limit is not None and limit > 0:
        logger.warning(f"Limiting evaluation to the first {limit} examples.")
        items_to_evaluate = items_to_evaluate[:limit]
        gold_sql_list = gold_sql_list[:limit]
    elif limit is not None and limit <= 0:
        logger.warning("Limit is non-positive, evaluating all examples.")

    exact_match_count = 0
    execution_match_count = 0  # Renamed for clarity
    total_count = len(items_to_evaluate)
    evaluation_results = []  # Renamed for clarity
    start_time = time.time()

    for i, item in enumerate(items_to_evaluate):
        query_text = item.get("question", "Missing question")
        db_id = item.get("db_id", "Missing db_id")
        gold_sql = gold_sql_list[i]  # Assumes list index matches item index
        db_path = os.path.join(db_base_path, db_id, f"{db_id}.sqlite")

        log_prefix = f"Item {i+1}/{total_count} (DB: {db_id})"
        logger.info(f"{log_prefix}: Evaluating question: {query_text[:80]}...")

        result_entry = {
            "index": i,
            "db_id": db_id,
            "question": query_text,
            "generated_sql": "N/A",
            "gold_sql": gold_sql,
            "exact_match": False,
            "execution_match": "skipped",
            "error": None,
        }

        if not os.path.exists(db_path):
            logger.warning(
                f"{log_prefix}: Database file not found at {db_path}. Skipping execution."
            )
            result_entry["error"] = "DB not found"
            evaluation_results.append(result_entry)
            continue

        # Generate SQL using AlbSQL instance
        try:
            # WARNING: Assumes generate_sql can function adequately without a live connector
            # or that schema information is loaded/cached appropriately beforehand.
            # Consider enhancing AlbSQL to accept db_path or schema details directly.
            generated_result = alb_sql_instance.generate_sql(
                query_text=query_text,
                db_name=db_id,  # Use db_id as the identifier
                clarify_ambiguities=False,  # Typically off for benchmarks
                execution_aware=False,  # Disable internal execution validation
            )
            generated_sql = generated_result.get("sql", "")
            result_entry["generated_sql"] = generated_sql
        except Exception as gen_e:
            logger.error(f"{log_prefix}: Error during SQL generation: {gen_e}")
            result_entry["error"] = f"SQL Generation Error: {gen_e}"
            generated_sql = ""  # Ensure generated_sql is defined for later steps

        # --- Evaluation Metrics ---

        # 1. Exact Match (EM) Score
        normalized_generated = normalize_sql(generated_sql)
        normalized_gold = normalize_sql(gold_sql)
        is_exact_match = normalized_generated == normalized_gold
        if is_exact_match:
            exact_match_count += 1
        result_entry["exact_match"] = is_exact_match
        logger.debug(f"{log_prefix}: Exact Match: {is_exact_match}")

        # 2. Execution Accuracy (EX) Score
        is_execution_match = False  # Default to False
        result_entry["execution_match"] = False  # Default

        gold_results, gold_error = execute_sql(db_path, gold_sql)
        if gold_error:
            logger.warning(
                f"{log_prefix}: Error executing GOLD SQL: {gold_error}. Cannot determine execution match."
            )
            result_entry["error"] = f"Gold SQL Error: {gold_error}"
            result_entry["execution_match"] = "error_gold"
        else:
            # Only execute generated SQL if gold SQL succeeded
            generated_results, generated_error = execute_sql(db_path, generated_sql)
            if generated_error:
                logger.debug(
                    f"{log_prefix}: Error executing GENERATED SQL: {generated_error}"
                )
                result_entry["error"] = f"Generated SQL Error: {generated_error}"
                result_entry["execution_match"] = "error_gen"
            elif gold_results is not None and generated_results is not None:
                # Compare results (execute_sql sorts them)
                is_execution_match = gold_results == generated_results
                if is_execution_match:
                    execution_match_count += 1
                result_entry["execution_match"] = is_execution_match
                logger.debug(f"{log_prefix}: Execution Match: {is_execution_match}")
            else:
                # This case (no error but None results) should be rare
                logger.warning(
                    f"{log_prefix}: Could not compare results (Gold: {gold_results is not None}, Gen: {generated_results is not None})"
                )
                result_entry["error"] = "Result comparison failed unexpectedly"
                result_entry["execution_match"] = "error_comparison"

        evaluation_results.append(result_entry)

        # Optional: Log progress periodically
        # Log progress periodically
        if (i + 1) % 50 == 0:
            logger.info(f"  Processed {i+1}/{total_count} examples...")

    end_time = time.time()
    duration = end_time - start_time

    # --- Final Metrics Calculation ---
    em_accuracy = (exact_match_count / total_count) * 100 if total_count > 0 else 0
    ex_accuracy = (execution_match_count / total_count) * 100 if total_count > 0 else 0

    logger.info("-" * 30)
    logger.info("BIRD Evaluation Summary:")
    logger.info(f"  Total Examples Evaluated: {total_count}")
    logger.info(
        f"  Exact Match (EM) Accuracy: {em_accuracy:.2f}% "
        f"({exact_match_count}/{total_count})"
    )
    logger.info(
        f"  Execution (EX) Accuracy: {ex_accuracy:.2f}% "
        f"({execution_match_count}/{total_count})"
    )
    logger.info(f"  Total Evaluation Duration: {duration:.2f} seconds")
    logger.info("-" * 30)

    # --- Save Detailed Results ---
    results_file = "bird_evaluation_results.json"
    try:
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
        logger.info(f"Detailed evaluation results saved to {results_file}")
    except IOError as e:
        logger.error(f"Failed to save evaluation results to {results_file}: {e}")
    except TypeError as e:
        logger.error(f"Failed to serialize evaluation results to JSON: {e}")


# --- Main Execution Logic ---


def run_bird_evaluation():
    """Sets up and runs the BIRD evaluation pipeline."""
    logger.info("Setting up BIRD benchmark pipeline...")

    # Define data paths
    # train_data_dir = os.path.join("data", "train", "train") # Commented out - unused
    dev_data_dir = os.path.join("data", "dev", "dev_20240627")

    # Load Development Data
    dev_data = load_bird_data(
        dev_data_dir,
        tables_file="dev_tables.json",
        data_file="dev.json",
        gold_sql_file="dev.sql",
    )
    if not dev_data.get("items"):
        logger.error("Dev data loading failed or is empty. Cannot proceed.")
        return

    # Check for database directory
    db_dir = dev_data.get("db_base_path", "")
    if not os.path.isdir(db_dir):
        logger.error(
            f"Database directory not found: {db_dir}. Please extract databases."
        )
        logger.error(
            f"Expected location: '{os.path.join(dev_data_dir, 'databases')}' "
            f"containing subdirectories for each db_id."
        )
        logger.error(
            f"You might need to unzip '{os.path.join(dev_data_dir, 'dev_databases.zip')}' "
            f"into '{db_dir}'"
        )
        return

    # Initialize AlbSQL for evaluation
    logger.info("Initializing AlbSQL instance for BIRD evaluation...")
    # TODO: Confirm/update model_name with the actual BIRD-optimized model identifier
    alb_sql_eval_instance = AlbSQL(
        model_name="tscholak/1rpp-sql-base",
        db_connector=None,  # Evaluation function handles DB connections
    )

    # Run Evaluation (limit=None evaluates all)
    evaluate_bird(alb_sql_eval_instance, dev_data, limit=None)

    logger.info("BIRD benchmark evaluation complete.")


def run_original_example():
    """Runs the original example usage of AlbSQL."""
    logger.info("Running original example usage...")
    # Initialize with default settings (might need a connector depending on setup)
    # WARNING: Ensure a suitable db_connector or schema setup exists for 'e_commerce'
    try:
        alb_sql_original = AlbSQL(model_name="gpt-4")  # Example uses gpt-4
        query = "Find users who placed orders in the last month"
        db_name = "e_commerce"  # Assumes this DB is configured/accessible
        result = alb_sql_original.generate_sql(query_text=query, db_name=db_name)

        print("\n--- Original Example Result ---")
        print(f"Query: {query}")
        print(f"DB Name: {db_name}")
        print(f"Generated SQL:\n{result.get('sql', 'N/A')}")
        print(f"Detected tables: {result.get('tables', [])}")
        print(f"Complexity: {result.get('complexity', 'unknown')}")
        print("--- End Original Example ---")

    except Exception as e:
        logger.error(f"Error running original example: {e}")
        print(f"\nError running original example for db '{db_name}': {e}")


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(description="Alb-SQL: Run evaluation or example.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["eval", "example"],
        default="eval",
        help="Mode to run: 'eval' for BIRD evaluation, 'example' for original example.",
    )
    args = parser.parse_args()

    if args.mode == "eval":
        run_bird_evaluation()
    elif args.mode == "example":
        run_original_example()
    else:
        # Should not happen with choices defined
        logger.error(f"Invalid mode specified: {args.mode}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",  # Slightly adjusted format
    )
    main()
