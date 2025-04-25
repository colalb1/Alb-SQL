"""
Unit tests for SQL comparison utilities.

This module tests the SQL comparison utility functions that are used to compare
SQL queries with flexibility for different formatting and aliasing.
"""

import pytest

from tests.utils.sql_comparison import (
    analyze_sql_differences,
    are_sqls_equivalent,
    compare_sql_structure,
    extract_columns,
    extract_tables,
    normalize_sql,
)


class TestSqlComparison:
    """Tests for the SQL comparison utilities."""

    def test_normalize_sql(self):
        """Test SQL normalization."""
        # Test with different formatting
        sql1 = "SELECT id, name FROM users WHERE age > 18"
        sql2 = """
        SELECT
            id,
            name
        FROM
            users
        WHERE
            age > 18
        """

        norm1 = normalize_sql(sql1)
        norm2 = normalize_sql(sql2)

        assert norm1 == norm2

        # Test with different case
        sql3 = "select ID, NAME from USERS where AGE > 18"
        norm3 = normalize_sql(sql3)

        assert (
            norm1.upper() == norm3.upper()
        )  # Case might be different based on normalization

        # Test with single vs double quotes
        sql4 = "SELECT id, name FROM users WHERE name = 'John'"
        sql5 = 'SELECT id, name FROM users WHERE name = "John"'

        norm4 = normalize_sql(sql4)
        norm5 = normalize_sql(sql5)

        assert "John" in norm4
        assert norm4 == norm5

    def test_extract_tables(self):
        """Test extracting table names from SQL queries."""
        # Simple query
        sql1 = "SELECT * FROM users"
        tables1 = extract_tables(sql1)
        assert "users" in tables1
        assert len(tables1) == 1

        # Query with multiple tables
        sql2 = "SELECT u.name, o.total FROM users u JOIN orders o ON u.id = o.user_id"
        tables2 = extract_tables(sql2)
        assert "users" in tables2
        assert "orders" in tables2
        assert len(tables2) == 2

        # Query with aliases
        sql3 = "SELECT u.name FROM users AS u"
        tables3 = extract_tables(sql3)
        assert "users" in tables3

    def test_extract_columns(self):
        """Test extracting column names from SQL queries."""
        # Simple query
        sql1 = "SELECT id, name, email FROM users"
        columns1 = extract_columns(sql1)
        assert "id" in columns1
        assert "name" in columns1
        assert "email" in columns1
        assert len(columns1) == 3

        # Query with table aliases
        sql2 = "SELECT u.id, u.name FROM users u"
        columns2 = extract_columns(sql2)
        assert "id" in columns2
        assert "name" in columns2
        assert len(columns2) == 2

        # Query with column aliases
        sql3 = "SELECT id, name AS full_name FROM users"
        columns3 = extract_columns(sql3)
        assert "id" in columns3
        assert (
            "name" in columns3 or "full_name" in columns3
        )  # Depending on implementation

    def test_compare_sql_structure(self):
        """Test comparing SQL query structures."""
        # Similar queries with formatting differences
        sql1 = "SELECT id, name FROM users WHERE age > 18 ORDER BY name"
        sql2 = """
        SELECT
            id,
            name
        FROM
            users
        WHERE
            age > 18
        ORDER BY
            name
        """

        similarity, details = compare_sql_structure(sql1, sql2)
        assert similarity > 0.9  # Should be very similar

        # Different queries
        sql3 = "SELECT id, name FROM users"
        sql4 = "SELECT id, name, email FROM users WHERE age > 18"

        similarity, details = compare_sql_structure(sql3, sql4)
        assert similarity < 0.9  # Should be less similar

        # Check details structure
        assert "table_similarity" in details
        assert "column_similarity" in details
        assert "operation_similarity" in details
        assert "tables1" in details
        assert "tables2" in details
        assert "columns1" in details
        assert "columns2" in details
        assert "operations" in details

    def test_are_sqls_equivalent(self):
        """Test determining if SQL queries are equivalent."""
        # Equivalent queries with different formatting
        sql1 = "SELECT id, name FROM users WHERE age > 18 ORDER BY name"
        sql2 = """
        SELECT
            id,
            name
        FROM
            users
        WHERE
            age > 18
        ORDER BY
            name
        """

        assert are_sqls_equivalent(sql1, sql2)

        # Different queries
        sql3 = "SELECT id, name FROM users"
        sql4 = "SELECT id, name, email FROM users WHERE age > 18"

        assert not are_sqls_equivalent(sql3, sql4)

        # Test with different threshold
        sql5 = "SELECT id, name FROM users ORDER BY name"
        sql6 = "SELECT id, name FROM users ORDER BY id"

        # These may be equivalent with a low threshold but not with a high threshold
        assert are_sqls_equivalent(sql5, sql6, threshold=0.7)
        assert not are_sqls_equivalent(sql5, sql6, threshold=0.99)

    def test_analyze_sql_differences(self):
        """Test analyzing differences between SQL queries."""
        # Queries with different columns
        sql1 = "SELECT id, name FROM users"
        sql2 = "SELECT id, name, email FROM users"

        differences = analyze_sql_differences(sql1, sql2)
        assert "email" in differences["different_columns"]

        # Queries with different operations
        sql3 = "SELECT id, name FROM users"
        sql4 = "SELECT id, name FROM users WHERE age > 18"

        differences = analyze_sql_differences(sql3, sql4)
        assert "where" in differences["different_operations"]

        # Queries with different aliases
        sql5 = "SELECT id, name AS full_name FROM users"
        sql6 = "SELECT id, name AS username FROM users"

        differences = analyze_sql_differences(sql5, sql6)
        assert len(differences["aliasing_differences"]["different_aliases"]) > 0

    def test_complex_queries(self):
        """Test comparing complex SQL queries."""
        # Complex query with JOIN, GROUP BY, HAVING, ORDER BY
        sql1 = """
        SELECT
            u.name,
            COUNT(o.id) AS order_count
        FROM
            users u
        JOIN
            orders o ON u.id = o.user_id
        WHERE
            o.status = 'completed'
        GROUP BY
            u.name
        HAVING
            COUNT(o.id) > 5
        ORDER BY
            order_count DESC
        LIMIT 10
        """

        # Same query with different aliases and formatting
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
        assert similarity > 0.8  # Should be quite similar

        assert are_sqls_equivalent(sql1, sql2, threshold=0.8)

        differences = analyze_sql_differences(sql1, sql2)
        assert "different_aliases" in differences["aliasing_differences"]


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
