"""
Database utilities for SQL execution.
"""

import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd


class DatabaseManager:
    """Manages database connections and executions."""

    def __init__(self, db_dir: Union[str, Path]):
        """
        Initialize the database manager.

        Args:
            db_dir: Directory containing database files
        """
        self.db_dir = Path(db_dir) if isinstance(db_dir, str) else db_dir
        if not self.db_dir.exists():
            raise FileNotFoundError(f"Database directory not found: {self.db_dir}")

        self.connections: Dict[str, sqlite3.Connection] = {}

    def get_connection(self, db_id: str) -> sqlite3.Connection:
        """
        Get a connection to a specific database.

        Args:
            db_id: Database ID

        Returns:
            SQLite connection
        """
        if db_id in self.connections:
            return self.connections[db_id]

        # Find the database file
        db_path = self._find_db_path(db_id)

        # Create a new connection
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        self.connections[db_id] = conn

        return conn

    def _find_db_path(self, db_id: str) -> Path:
        """
        Find the path to a database file.

        Args:
            db_id: Database ID

        Returns:
            Path to the database file
        """
        # Check for dev database structure
        dev_path = self.db_dir / "dev_databases" / db_id / f"{db_id}.sqlite"
        if dev_path.exists():
            return dev_path

        # Check for train database structure
        train_path = self.db_dir / "train_databases" / db_id / f"{db_id}.sqlite"
        if train_path.exists():
            return train_path

        # Direct path
        direct_path = self.db_dir / f"{db_id}.sqlite"
        if direct_path.exists():
            return direct_path

        raise FileNotFoundError(f"Database file not found for ID: {db_id}")

    def execute_query(
        self, db_id: str, query: str
    ) -> Tuple[bool, Union[pd.DataFrame, str]]:
        """
        Execute a SQL query against a database.

        Args:
            db_id: Database ID
            query: SQL query to execute

        Returns:
            Tuple of (success, result)
                - success: Boolean indicating if query executed successfully
                - result: DataFrame with results if successful, error message if not
        """
        try:
            conn = self.get_connection(db_id)

            # Execute query and fetch results
            result = pd.read_sql_query(query, conn)

            return True, result
        except Exception as e:
            return False, str(e)

    def get_table_info(self, db_id: str, table_name: str) -> Optional[pd.DataFrame]:
        """
        Get information about a table's structure.

        Args:
            db_id: Database ID
            table_name: Name of the table

        Returns:
            DataFrame with table structure information or None if table doesn't exist
        """
        try:
            conn = self.get_connection(db_id)

            # Get table info
            query = f"PRAGMA table_info({table_name});"
            result = pd.read_sql_query(query, conn)

            if result.empty:
                return None

            return result
        except Exception:
            return None

    def get_tables(self, db_id: str) -> List[str]:
        """
        Get a list of all tables in a database.

        Args:
            db_id: Database ID

        Returns:
            List of table names
        """
        try:
            conn = self.get_connection(db_id)

            # Get all tables
            query = "SELECT name FROM sqlite_master WHERE type='table';"
            result = pd.read_sql_query(query, conn)

            return result["name"].tolist()
        except Exception:
            return []

    def get_foreign_keys(self, db_id: str, table_name: str) -> List[Dict]:
        """
        Get foreign key information for a table.

        Args:
            db_id: Database ID
            table_name: Name of the table

        Returns:
            List of foreign key information dictionaries
        """
        try:
            conn = self.get_connection(db_id)

            # Get foreign keys
            query = f"PRAGMA foreign_key_list({table_name});"
            result = pd.read_sql_query(query, conn)

            if result.empty:
                return []

            # Convert to list of dictionaries
            return result.to_dict("records")
        except Exception:
            return []

    def get_sample_data(
        self, db_id: str, table_name: str, limit: int = 5
    ) -> Optional[pd.DataFrame]:
        """
        Get sample data from a table.

        Args:
            db_id: Database ID
            table_name: Name of the table
            limit: Maximum number of rows to return

        Returns:
            DataFrame with sample data or None if table doesn't exist
        """
        try:
            conn = self.get_connection(db_id)

            # Get sample data
            query = f"SELECT * FROM {table_name} LIMIT {limit};"
            result = pd.read_sql_query(query, conn)

            return result
        except Exception:
            return None

    def close_all(self) -> None:
        """Close all database connections."""
        for conn in self.connections.values():
            conn.close()

        self.connections.clear()

    def __del__(self) -> None:
        """Destructor to ensure all connections are closed."""
        self.close_all()


def execute_query_with_timeout(
    db_path: Union[str, Path], query: str, timeout: float = 30.0
) -> Tuple[bool, Union[pd.DataFrame, str]]:
    """
    Execute a SQL query with a timeout.

    Args:
        db_path: Path to the database file
        query: SQL query to execute
        timeout: Timeout in seconds

    Returns:
        Tuple of (success, result)
            - success: Boolean indicating if query executed successfully
            - result: DataFrame with results if successful, error message if not
    """
    try:
        # Connect to the database with timeout
        conn = sqlite3.connect(str(db_path), timeout=timeout)

        # Execute query and fetch results
        result = pd.read_sql_query(query, conn)
        conn.close()

        return True, result
    except Exception as e:
        return False, str(e)


def compare_query_results(
    result1: pd.DataFrame, result2: pd.DataFrame, ignore_order: bool = True
) -> Tuple[bool, Optional[str]]:
    """
    Compare the results of two SQL queries.

    Args:
        result1: First query result
        result2: Second query result
        ignore_order: Whether to ignore row order when comparing

    Returns:
        Tuple of (are_equal, difference_description)
    """
    # Check if columns match
    if set(result1.columns) != set(result2.columns):
        missing_in_1 = set(result2.columns) - set(result1.columns)
        missing_in_2 = set(result1.columns) - set(result2.columns)
        return (
            False,
            f"Column mismatch: Missing in first: {missing_in_1}, Missing in second: {missing_in_2}",
        )

    # Sort results if ignoring order
    if ignore_order:
        result1 = result1.sort_values(by=list(result1.columns)).reset_index(drop=True)
        result2 = result2.sort_values(by=list(result2.columns)).reset_index(drop=True)

    # Check if row counts match
    if len(result1) != len(result2):
        return False, f"Row count mismatch: {len(result1)} vs {len(result2)}"

    # Check if all values match
    if not result1.equals(result2):
        # Find the first difference
        for i in range(len(result1)):
            for col in result1.columns:
                if result1.iloc[i][col] != result2.iloc[i][col]:
                    return (
                        False,
                        f"Value mismatch at row {i}, column '{col}': {result1.iloc[i][col]} vs {result2.iloc[i][col]}",
                    )

    return True, None
