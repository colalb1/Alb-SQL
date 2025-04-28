"""
Evaluation metrics for SQL generation models.
"""

import re
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score

from ..utils.database import compare_query_results


class EvaluationMetrics:
    """Metrics for evaluating SQL generation models."""

    @staticmethod
    def exact_match(
        predictions: List[str], references: List[str], normalize: bool = True
    ) -> float:
        """
        Calculate exact match accuracy between predicted and reference SQL queries.

        Args:
            predictions: List of predicted SQL queries
            references: List of reference SQL queries
            normalize: Whether to normalize queries before comparison

        Returns:
            Exact match accuracy
        """
        if len(predictions) != len(references):
            raise ValueError("Number of predictions and references must match")

        matches = 0

        for pred, ref in zip(predictions, references):
            if normalize:
                pred = EvaluationMetrics.normalize_query(pred)
                ref = EvaluationMetrics.normalize_query(ref)

            if pred == ref:
                matches += 1

        return matches / len(predictions) if len(predictions) > 0 else 0.0

    @staticmethod
    def execution_match(
        predictions: List[str],
        references: List[str],
        db_paths: List[str],
        ignore_order: bool = True,
        timeout: float = 30.0,
    ) -> Tuple[float, List[bool]]:
        """
        Calculate execution match accuracy between predicted and reference SQL queries.

        Args:
            predictions: List of predicted SQL queries
            references: List of reference SQL queries
            db_paths: List of paths to database files
            ignore_order: Whether to ignore row order when comparing results
            timeout: Timeout in seconds for query execution

        Returns:
            Tuple of (execution match accuracy, list of match results)
        """
        from ..utils.database import execute_query_with_timeout

        if not (len(predictions) == len(references) == len(db_paths)):
            raise ValueError(
                "Number of predictions, references, and database paths must match"
            )

        matches = []

        for pred, ref, db_path in zip(predictions, references, db_paths):
            # Execute reference query
            ref_success, ref_result = execute_query_with_timeout(db_path, ref, timeout)

            if not ref_success:
                # If reference query fails, count as no match
                matches.append(False)
                continue

            # Execute predicted query
            pred_success, pred_result = execute_query_with_timeout(
                db_path, pred, timeout
            )

            if not pred_success:
                # If predicted query fails, count as no match
                matches.append(False)
                continue

            # Compare results
            results_match, _ = compare_query_results(
                pred_result, ref_result, ignore_order=ignore_order
            )

            matches.append(results_match)

        return sum(matches) / len(matches) if matches else 0.0, matches

    @staticmethod
    def token_level_metrics(
        predictions: List[str], references: List[str], normalize: bool = True
    ) -> Dict[str, float]:
        """
        Calculate token-level metrics (precision, recall, F1) between predicted and reference SQL queries.

        Args:
            predictions: List of predicted SQL queries
            references: List of reference SQL queries
            normalize: Whether to normalize queries before comparison

        Returns:
            Dictionary of token-level metrics
        """
        if len(predictions) != len(references):
            raise ValueError("Number of predictions and references must match")

        # Tokenize queries
        pred_tokens_list = []
        ref_tokens_list = []

        for pred, ref in zip(predictions, references):
            if normalize:
                pred = EvaluationMetrics.normalize_query(pred)
                ref = EvaluationMetrics.normalize_query(ref)

            pred_tokens = EvaluationMetrics.tokenize_query(pred)
            ref_tokens = EvaluationMetrics.tokenize_query(ref)

            pred_tokens_list.append(pred_tokens)
            ref_tokens_list.append(ref_tokens)

        # Calculate metrics
        precision = EvaluationMetrics._token_precision(
            pred_tokens_list, ref_tokens_list
        )
        recall = EvaluationMetrics._token_recall(pred_tokens_list, ref_tokens_list)
        f1 = EvaluationMetrics._token_f1(pred_tokens_list, ref_tokens_list)

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    @staticmethod
    def _token_precision(
        pred_tokens_list: List[List[str]], ref_tokens_list: List[List[str]]
    ) -> float:
        """Calculate token-level precision."""
        precisions = []

        for pred_tokens, ref_tokens in zip(pred_tokens_list, ref_tokens_list):
            # Convert to sets for intersection
            pred_set = set(pred_tokens)
            ref_set = set(ref_tokens)

            # Calculate precision
            if pred_set:
                precision = len(pred_set.intersection(ref_set)) / len(pred_set)
                precisions.append(precision)

        return np.mean(precisions) if precisions else 0.0

    @staticmethod
    def _token_recall(
        pred_tokens_list: List[List[str]], ref_tokens_list: List[List[str]]
    ) -> float:
        """Calculate token-level recall."""
        recalls = []

        for pred_tokens, ref_tokens in zip(pred_tokens_list, ref_tokens_list):
            # Convert to sets for intersection
            pred_set = set(pred_tokens)
            ref_set = set(ref_tokens)

            # Calculate recall
            if ref_set:
                recall = len(pred_set.intersection(ref_set)) / len(ref_set)
                recalls.append(recall)

        return np.mean(recalls) if recalls else 0.0

    @staticmethod
    def _token_f1(
        pred_tokens_list: List[List[str]], ref_tokens_list: List[List[str]]
    ) -> float:
        """Calculate token-level F1 score."""
        f1_scores = []

        for pred_tokens, ref_tokens in zip(pred_tokens_list, ref_tokens_list):
            # Convert to sets for intersection
            pred_set = set(pred_tokens)
            ref_set = set(ref_tokens)

            # Calculate precision and recall
            if pred_set and ref_set:
                precision = len(pred_set.intersection(ref_set)) / len(pred_set)
                recall = len(pred_set.intersection(ref_set)) / len(ref_set)

                # Calculate F1
                if precision + recall > 0:
                    f1 = 2 * precision * recall / (precision + recall)
                    f1_scores.append(f1)

        return np.mean(f1_scores) if f1_scores else 0.0

    @staticmethod
    def normalize_query(query: str) -> str:
        """
        Normalize a SQL query for comparison.

        Args:
            query: SQL query to normalize

        Returns:
            Normalized SQL query
        """
        # Convert to lowercase
        query = query.lower()

        # Remove extra whitespace
        query = re.sub(r"\s+", " ", query).strip()

        # Remove trailing semicolon
        query = query.rstrip(";")

        # Normalize quotes
        query = query.replace('"', "'")

        # Normalize aliases (e.g., "AS t" -> "t")
        query = re.sub(r"\s+as\s+([a-zA-Z0-9_]+)", r" \1", query)

        return query

    @staticmethod
    def tokenize_query(query: str) -> List[str]:
        """
        Tokenize a SQL query.

        Args:
            query: SQL query to tokenize

        Returns:
            List of tokens
        """
        # Split on whitespace and punctuation
        tokens = re.findall(r"[a-zA-Z0-9_]+|[^\s\w]", query)

        # Filter out empty tokens
        tokens = [token for token in tokens if token.strip()]

        return tokens

    @staticmethod
    def component_match(
        predictions: List[str], references: List[str], normalize: bool = True
    ) -> Dict[str, float]:
        """
        Calculate component-level match metrics for SQL queries.

        Args:
            predictions: List of predicted SQL queries
            references: List of reference SQL queries
            normalize: Whether to normalize queries before comparison

        Returns:
            Dictionary of component-level metrics
        """
        if len(predictions) != len(references):
            raise ValueError("Number of predictions and references must match")

        # Extract components
        pred_components = [
            EvaluationMetrics.extract_components(pred, normalize)
            for pred in predictions
        ]
        ref_components = [
            EvaluationMetrics.extract_components(ref, normalize) for ref in references
        ]

        # Calculate metrics for each component
        metrics = {}

        # Components to evaluate
        components = ["select", "from", "where", "group_by", "order_by", "limit"]

        for component in components:
            # Extract component values
            pred_values = [comp.get(component, "") for comp in pred_components]
            ref_values = [comp.get(component, "") for comp in ref_components]

            # Calculate accuracy
            accuracy = accuracy_score(
                [bool(val) for val in ref_values], [bool(val) for val in pred_values]
            )

            # Calculate exact match for non-empty components
            exact_matches = []
            for pred_val, ref_val in zip(pred_values, ref_values):
                if ref_val:  # Only consider cases where reference has the component
                    exact_matches.append(pred_val == ref_val)

            exact_match = np.mean(exact_matches) if exact_matches else 0.0

            metrics[f"{component}_accuracy"] = accuracy
            metrics[f"{component}_exact_match"] = exact_match

        return metrics

    @staticmethod
    def extract_components(query: str, normalize: bool = True) -> Dict[str, str]:
        """
        Extract components from a SQL query.

        Args:
            query: SQL query to extract components from
            normalize: Whether to normalize components

        Returns:
            Dictionary of query components
        """
        if normalize:
            query = EvaluationMetrics.normalize_query(query)

        components = {}

        # Extract SELECT
        select_match = re.search(
            r"select\s+(.*?)(?:\s+from\s+|$)", query, re.IGNORECASE
        )
        if select_match:
            components["select"] = select_match.group(1).strip()

        # Extract FROM
        from_match = re.search(
            r"from\s+(.*?)(?:\s+where\s+|\s+group\s+by\s+|\s+order\s+by\s+|\s+limit\s+|$)",
            query,
            re.IGNORECASE,
        )
        if from_match:
            components["from"] = from_match.group(1).strip()

        # Extract WHERE
        where_match = re.search(
            r"where\s+(.*?)(?:\s+group\s+by\s+|\s+order\s+by\s+|\s+limit\s+|$)",
            query,
            re.IGNORECASE,
        )
        if where_match:
            components["where"] = where_match.group(1).strip()

        # Extract GROUP BY
        group_by_match = re.search(
            r"group\s+by\s+(.*?)(?:\s+having\s+|\s+order\s+by\s+|\s+limit\s+|$)",
            query,
            re.IGNORECASE,
        )
        if group_by_match:
            components["group_by"] = group_by_match.group(1).strip()

        # Extract HAVING
        having_match = re.search(
            r"having\s+(.*?)(?:\s+order\s+by\s+|\s+limit\s+|$)", query, re.IGNORECASE
        )
        if having_match:
            components["having"] = having_match.group(1).strip()

        # Extract ORDER BY
        order_by_match = re.search(
            r"order\s+by\s+(.*?)(?:\s+limit\s+|$)", query, re.IGNORECASE
        )
        if order_by_match:
            components["order_by"] = order_by_match.group(1).strip()

        # Extract LIMIT
        limit_match = re.search(r"limit\s+(.*?)$", query, re.IGNORECASE)
        if limit_match:
            components["limit"] = limit_match.group(1).strip()

        return components
