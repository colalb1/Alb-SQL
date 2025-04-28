"""
SQL validation utilities.
"""

import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import sqlparse


class SQLValidator:
    """Validates SQL queries against database schemas."""

    def __init__(self, db_path: Union[str, Path]):
        """
        Initialize the SQL validator.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = Path(db_path) if isinstance(db_path, str) else db_path
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database file not found: {self.db_path}")

        self.conn = None
        self._connect()
        self.schema = self._extract_schema()

    def _connect(self) -> None:
        """Connect to the SQLite database."""
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row

    def _extract_schema(self) -> Dict[str, List[str]]:
        """Extract the database schema."""
        if not self.conn:
            self._connect()

        cursor = self.conn.cursor()

        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]

        schema = {}
        for table in tables:
            # Get columns for each table
            cursor.execute(f"PRAGMA table_info({table});")
            columns = [row[1] for row in cursor.fetchall()]
            schema[table] = columns

        return schema

    def validate_syntax(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        Validate SQL syntax.

        Args:
            query: SQL query to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Parse the SQL query
            parsed = sqlparse.parse(query)
            if not parsed:
                return False, "Empty query"

            # Basic syntax validation
            formatted_query = sqlparse.format(
                query, reindent=True, keyword_case="upper"
            )

            # Check for basic SQL syntax errors
            if not self._check_basic_syntax(formatted_query):
                return False, "Basic syntax error"

            return True, None
        except Exception as e:
            return False, str(e)

    def _check_basic_syntax(self, query: str) -> bool:
        """Check for basic SQL syntax errors."""
        # Check for balanced parentheses
        if query.count("(") != query.count(")"):
            return False

        # Check for unclosed quotes
        if query.count("'") % 2 != 0 or query.count('"') % 2 != 0:
            return False

        # Check for basic SQL keywords
        has_select = bool(re.search(r"\bSELECT\b", query, re.IGNORECASE))
        has_from = bool(re.search(r"\bFROM\b", query, re.IGNORECASE))

        if not has_select or not has_from:
            return False

        return True

    def validate_against_schema(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        Validate SQL query against the database schema.

        Args:
            query: SQL query to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        # First check syntax
        syntax_valid, syntax_error = self.validate_syntax(query)
        if not syntax_valid:
            return False, f"Syntax error: {syntax_error}"

        # Extract table and column references
        tables, columns = self._extract_references(query)

        # Validate table references
        for table in tables:
            if table not in self.schema:
                return False, f"Unknown table: {table}"

        # Validate column references
        for col_ref in columns:
            if "." in col_ref:
                # Qualified column reference (table.column)
                table, column = col_ref.split(".")
                if table not in self.schema:
                    return False, f"Unknown table in column reference: {table}"
                if column != "*" and column not in self.schema[table]:
                    return False, f"Unknown column {column} in table {table}"
            else:
                # Unqualified column reference
                if col_ref != "*":
                    # Check if column exists in any table
                    found = False
                    for table_cols in self.schema.values():
                        if col_ref in table_cols:
                            found = True
                            break

                    if not found:
                        return False, f"Unknown column: {col_ref}"

        return True, None

    def _extract_references(self, query: str) -> Tuple[List[str], List[str]]:
        """
        Extract table and column references from a SQL query.

        Returns:
            Tuple of (tables, columns)
        """
        # Simple regex-based extraction (not comprehensive)
        tables = []
        columns = []

        # Extract table references from FROM and JOIN clauses
        from_pattern = r"\bFROM\s+([a-zA-Z0-9_]+)"
        join_pattern = r"\bJOIN\s+([a-zA-Z0-9_]+)"

        tables.extend(re.findall(from_pattern, query, re.IGNORECASE))
        tables.extend(re.findall(join_pattern, query, re.IGNORECASE))

        # Extract column references
        # This is a simplified approach and won't catch all cases
        select_pattern = r"\bSELECT\s+(.*?)\s+FROM"
        select_matches = re.search(select_pattern, query, re.IGNORECASE | re.DOTALL)

        if select_matches:
            select_clause = select_matches.group(1)
            # Handle * case
            if "*" in select_clause:
                columns.append("*")
            else:
                # Split by commas and clean up
                cols = [c.strip() for c in select_clause.split(",")]
                for col in cols:
                    # Handle aliasing and functions
                    if " AS " in col.upper():
                        col = col.split(" AS ")[0].strip()

                    # Handle function calls
                    if "(" in col and ")" in col:
                        # Extract column references from function arguments
                        arg_match = re.search(r"\((.*?)\)", col)
                        if arg_match:
                            arg = arg_match.group(1).strip()
                            if arg != "*" and not arg.isdigit():
                                columns.append(arg)
                    else:
                        columns.append(col)

        # Extract column references from WHERE clause
        where_pattern = r"\bWHERE\s+(.*?)(?:\bGROUP BY\b|\bORDER BY\b|\bLIMIT\b|$)"
        where_matches = re.search(where_pattern, query, re.IGNORECASE | re.DOTALL)

        if where_matches:
            where_clause = where_matches.group(1)
            # Extract column references (simplified)
            col_pattern = r"([a-zA-Z0-9_]+\.[a-zA-Z0-9_]+|[a-zA-Z0-9_]+)\s*[=<>]"
            where_cols = re.findall(col_pattern, where_clause)
            columns.extend(where_cols)

        return tables, columns

    def execute_query(self, query: str) -> Tuple[bool, Union[List[Dict], str]]:
        """
        Execute a SQL query against the database.

        Args:
            query: SQL query to execute

        Returns:
            Tuple of (success, result)
                - success: Boolean indicating if query executed successfully
                - result: List of row dictionaries if successful, error message if not
        """
        try:
            if not self.conn:
                self._connect()

            cursor = self.conn.cursor()
            cursor.execute(query)

            # Convert results to list of dictionaries
            columns = [col[0] for col in cursor.description]
            results = []

            for row in cursor.fetchall():
                results.append({columns[i]: row[i] for i in range(len(columns))})

            return True, results
        except Exception as e:
            return False, str(e)

    def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def __del__(self) -> None:
        """Destructor to ensure connection is closed."""
        self.close()
