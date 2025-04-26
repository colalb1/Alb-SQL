"""
SQL Comparison Utilities

This module provides utilities for comparing SQL queries with flexibility
for different formatting, aliasing, and other non-semantic differences.
"""

import re
from typing import Dict, Set, Tuple

import sqlparse


def normalize_sql(sql: str) -> str:
    """
    Normalize SQL query for comparison by removing whitespace, standardizing
    case, and other formatting adjustments.

    Args:
        sql (str): SQL query to normalize.

    Returns:
        Normalized SQL query.
    """
    # Parse and format SQL with sqlparse
    sql = sqlparse.format(
        sql,
        keyword_case="upper",
        identifier_case="lower",
        strip_comments=True,
        reindent=True,
        indent_width=2,
    )

    # Remove extra whitespace
    sql = " ".join(sql.split())

    # Replace multiple spaces with a single space
    sql = re.sub(r"\s+", " ", sql)

    # Standardize single and double quotes
    sql = sql.replace("'", '"')

    return sql


def extract_tables(sql: str) -> Set[str]:
    """
    Extract table names from a SQL query.

    Args:
        sql (str): SQL query.

    Returns:
        Set of table names.
    """
    # Parse SQL
    parsed = sqlparse.parse(sql)[0]
    tables = set()

    # Find table references - this is a simplified approach
    # For a robust solution, consider using a full SQL parser
    from_seen = False
    join_seen = False

    for token in parsed.flatten():
        token_value = token.value.lower()

        if token_value == "from":
            from_seen = True
            continue

        if token_value in (
            "join",
            "inner join",
            "left join",
            "right join",
            "full join",
        ):
            join_seen = True
            continue

        if (from_seen or join_seen) and token.ttype == sqlparse.tokens.Name:
            tables.add(token_value)
            from_seen = False
            join_seen = False

    return tables


def extract_columns(sql: str) -> Set[str]:
    """
    Extract column names from a SQL query using sqlparse structure.

    Args:
        sql (str): SQL query.

    Returns:
        Set of column names (lowercase).
    """
    parsed = sqlparse.parse(sql)[0]
    columns = set()

    # Iterate through top-level tokens. We are looking for Statement objects.
    for stmt_candidate in parsed.tokens:
        # Check if it's a Statement object and specifically a SELECT statement
        if (
            isinstance(stmt_candidate, sqlparse.sql.Statement)
            and stmt_candidate.get_type() == "SELECT"
        ):
            identifier_list_found = False
            # Iterate through tokens *within* the identified SELECT statement
            for token in stmt_candidate.tokens:
                # Found the list of columns/expressions
                if isinstance(token, sqlparse.sql.IdentifierList):
                    identifier_list_found = True
                    for identifier in token.get_identifiers():
                        # Use get_real_name() to handle aliases like 'col as alias' -> 'col'
                        # For functions like COUNT(*), get_real_name() might be None
                        real_name = identifier.get_real_name()

                        if real_name:
                            val = real_name
                            # Handle qualified names like 'table.col' -> 'col'
                            # Check if it's a simple identifier before splitting
                            if "." in val and isinstance(
                                identifier, sqlparse.sql.Identifier
                            ):
                                # Avoid splitting function calls like schema.func()
                                is_function = any(
                                    isinstance(t, sqlparse.sql.Function)
                                    for t in identifier.tokens
                                )
                                if not is_function:
                                    val = val.split(".")[-1]
                        else:
                            # Fallback for '*' or complex expressions/functions without aliases
                            val = identifier.value
                            # If it's a wildcard token specifically
                            if identifier.ttype == sqlparse.tokens.Wildcard:
                                val = "*"  # Ensure consistency

                        columns.add(val.lower())
                    break  # Processed the main identifier list

            # Handle simple 'SELECT *' case if IdentifierList wasn't found or didn't contain '*'
            if not identifier_list_found or "*" not in columns:
                is_select_star = False
                select_token_idx = -1
                # Find SELECT token index
                for idx, token in enumerate(stmt.tokens):
                    if token.value.upper() == "SELECT":
                        select_token_idx = idx
                        break
                # Check token immediately after SELECT (ignoring whitespace)
                if select_token_idx != -1:
                    next_significant_token = None
                    for next_idx in range(select_token_idx + 1, len(stmt.tokens)):
                        if not stmt.tokens[next_idx].is_whitespace:
                            next_significant_token = stmt.tokens[next_idx]
                            break
                    if (
                        next_significant_token
                        and next_significant_token.ttype == sqlparse.tokens.Wildcard
                    ):
                        columns.add("*")

    return columns


def compare_sql_structure(sql1: str, sql2: str) -> Tuple[float, Dict]:
    """
    Compare SQL queries for structural similarity, considering:
    - Tables used
    - Columns selected
    - Presence of key operations (JOIN, GROUP BY, etc.)

    Args:
        sql1 (str): First SQL query.
        sql2 (str): Second SQL query.

    Returns:
        Tuple of (similarity_score, details) where similarity_score is a float
        between 0 and 1, and details is a dictionary with comparison information.
    """
    # Normalize SQL queries
    norm_sql1 = normalize_sql(sql1)
    norm_sql2 = normalize_sql(sql2)

    # Extract components
    tables1 = extract_tables(norm_sql1)
    tables2 = extract_tables(norm_sql2)

    columns1 = extract_columns(norm_sql1)
    columns2 = extract_columns(norm_sql2)

    # Check for key operations
    operations = {
        "join": (
            bool(re.search(r"\bjoin\b", norm_sql1, re.IGNORECASE)),
            bool(re.search(r"\bjoin\b", norm_sql2, re.IGNORECASE)),
        ),
        "where": (
            bool(re.search(r"\bwhere\b", norm_sql1, re.IGNORECASE)),
            bool(re.search(r"\bwhere\b", norm_sql2, re.IGNORECASE)),
        ),
        "group_by": (
            bool(re.search(r"\bgroup by\b", norm_sql1, re.IGNORECASE)),
            bool(re.search(r"\bgroup by\b", norm_sql2, re.IGNORECASE)),
        ),
        "having": (
            bool(re.search(r"\bhaving\b", norm_sql1, re.IGNORECASE)),
            bool(re.search(r"\bhaving\b", norm_sql2, re.IGNORECASE)),
        ),
        "order_by": (
            bool(re.search(r"\border by\b", norm_sql1, re.IGNORECASE)),
            bool(re.search(r"\border by\b", norm_sql2, re.IGNORECASE)),
        ),
        "limit": (
            bool(re.search(r"\blimit\b", norm_sql1, re.IGNORECASE)),
            bool(re.search(r"\blimit\b", norm_sql2, re.IGNORECASE)),
        ),
        "distinct": (
            bool(re.search(r"\bdistinct\b", norm_sql1, re.IGNORECASE)),
            bool(re.search(r"\bdistinct\b", norm_sql2, re.IGNORECASE)),
        ),
        "union": (
            bool(re.search(r"\bunion\b", norm_sql1, re.IGNORECASE)),
            bool(re.search(r"\bunion\b", norm_sql2, re.IGNORECASE)),
        ),
        "subquery": (
            bool(re.search(r"\(\s*select", norm_sql1, re.IGNORECASE)),
            bool(re.search(r"\(\s*select", norm_sql2, re.IGNORECASE)),
        ),
    }

    # Calculate similarities
    table_similarity = len(tables1.intersection(tables2)) / max(
        len(tables1.union(tables2)), 1
    )
    column_similarity = len(columns1.intersection(columns2)) / max(
        len(columns1.union(columns2)), 1
    )

    operation_matches = sum(1 for op, (op1, op2) in operations.items() if op1 == op2)
    operation_similarity = operation_matches / len(operations)

    # Overall similarity (weighted)
    similarity = (
        0.4 * table_similarity + 0.3 * column_similarity + 0.3 * operation_similarity
    )

    details = {
        "table_similarity": table_similarity,
        "tables1": list(tables1),
        "tables2": list(tables2),
        "column_similarity": column_similarity,
        "columns1": list(columns1),
        "columns2": list(columns2),
        "operation_similarity": operation_similarity,
        "operations": {
            op: {"sql1": op1, "sql2": op2} for op, (op1, op2) in operations.items()
        },
    }

    return similarity, details


def are_sqls_equivalent(sql1: str, sql2: str, threshold: float = 0.9) -> bool:
    """
    Determine if two SQL queries are semantically equivalent.

    Args:
        sql1 (str): First SQL query.
        sql2 (str): Second SQL query.
        threshold (float): Similarity threshold to consider equivalent (0-1).

    Returns:
        True if the queries are considered equivalent, False otherwise.
    """
    similarity, _ = compare_sql_structure(sql1, sql2)
    return similarity >= threshold


def analyze_sql_differences(sql1: str, sql2: str) -> Dict:
    """
    Provide a detailed analysis of differences between two SQL queries.

    Args:
        sql1 (str): First SQL query.
        sql2 (str): Second SQL query.

    Returns:
        Dictionary with detailed difference analysis.
    """
    similarity, details = compare_sql_structure(sql1, sql2)

    # Extended analysis
    norm_sql1 = normalize_sql(sql1)
    norm_sql2 = normalize_sql(sql2)

    # Check specific differences
    analysis = {
        "similarity_score": similarity,
        "structure_details": details,
        "different_tables": list(
            set(details["tables1"]).symmetric_difference(set(details["tables2"]))
        ),
        "different_columns": list(
            set(details["columns1"]).symmetric_difference(set(details["columns2"]))
        ),
        "different_operations": [
            op
            for op, vals in details["operations"].items()
            if vals["sql1"] != vals["sql2"]
        ],
    }

    # Check for aliasing differences
    alias_pattern = r"(\w+)\s+as\s+(\w+)"
    aliases1 = dict(re.findall(alias_pattern, norm_sql1, re.IGNORECASE))
    aliases2 = dict(re.findall(alias_pattern, norm_sql2, re.IGNORECASE))

    analysis["aliasing_differences"] = {
        "aliases_only_in_sql1": {
            k: v for k, v in aliases1.items() if k not in aliases2
        },
        "aliases_only_in_sql2": {
            k: v for k, v in aliases2.items() if k not in aliases1
        },
        "different_aliases": {
            k: (aliases1[k], aliases2[k])
            for k in aliases1
            if k in aliases2 and aliases1[k] != aliases2[k]
        },
    }

    return analysis


if __name__ == "__main__":
    # Example usage
    sql1 = """
    SELECT u.name, COUNT(o.id) as order_count
    FROM users u 
    JOIN orders o ON u.id = o.user_id
    WHERE o.status = 'completed'
    GROUP BY u.name
    HAVING COUNT(o.id) > 5
    ORDER BY order_count DESC
    LIMIT 10
    """

    sql2 = """
    SELECT users.name, COUNT(orders.id) AS total_orders
    FROM users
    INNER JOIN orders ON users.id = orders.user_id
    WHERE orders.status = 'completed'
    GROUP BY users.name
    HAVING COUNT(orders.id) > 5
    ORDER BY total_orders DESC
    LIMIT 10
    """

    similarity, details = compare_sql_structure(sql1, sql2)
    print(f"Similarity: {similarity:.2f}")
    print(f"Tables 1: {details['tables1']}")
    print(f"Tables 2: {details['tables2']}")
    print(f"Equivalent: {are_sqls_equivalent(sql1, sql2)}")

    differences = analyze_sql_differences(sql1, sql2)
    print(
        f"Different aliases: {differences['aliasing_differences']['different_aliases']}"
    )
