"""
Pytest configuration and shared fixtures.

This file contains fixtures that can be used across all test files.
"""

import json
import os
import sys
from unittest.mock import MagicMock

import pytest

# Add project root to sys.path to allow importing main
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from main import AlbSQL


@pytest.fixture(scope="session")  # Added scope
def sample_schema_info():
    """
    Fixture providing a sample schema info for an e-commerce database.
    """
    from agents.schema_analyzer_agent import ColumnInfo, SchemaInfo, TableInfo

    # Create a SchemaInfo object for e-commerce DB
    schema_info = SchemaInfo(name="e_commerce")

    # Users table
    users_table = TableInfo(
        name="users",
        description="Table storing user information",
        sample_row_count=1000,
    )
    users_table.columns["id"] = ColumnInfo(
        name="id",
        data_type="INTEGER",
        nullable=False,
        is_primary_key=True,
        sample_values=["1", "2", "3"],
    )
    users_table.columns["name"] = ColumnInfo(
        name="name",
        data_type="VARCHAR(100)",
        nullable=False,
        sample_values=["John Smith", "Jane Doe"],
    )
    users_table.columns["email"] = ColumnInfo(
        name="email",
        data_type="VARCHAR(255)",
        nullable=False,
        sample_values=["john@example.com", "jane@example.com"],
    )
    users_table.primary_keys = ["id"]
    schema_info.tables["users"] = users_table

    # Products table
    products_table = TableInfo(
        name="products",
        description="Table storing product information",
        sample_row_count=500,
    )
    products_table.columns["id"] = ColumnInfo(
        name="id",
        data_type="INTEGER",
        nullable=False,
        is_primary_key=True,
        sample_values=["1", "2", "3"],
    )
    products_table.columns["name"] = ColumnInfo(
        name="name",
        data_type="VARCHAR(100)",
        nullable=False,
        sample_values=["Laptop", "Smartphone"],
    )
    products_table.columns["price"] = ColumnInfo(
        name="price",
        data_type="DECIMAL(10, 2)",
        nullable=False,
        sample_values=["999.99", "599.99"],
    )
    products_table.columns["category_id"] = ColumnInfo(
        name="category_id",
        data_type="INTEGER",
        nullable=True,
        is_foreign_key=True,
        referenced_table="categories",
        referenced_column="id",
        sample_values=["1", "2"],
    )
    products_table.primary_keys = ["id"]
    products_table.foreign_keys["category_id"] = ("categories", "id")
    schema_info.tables["products"] = products_table

    # Add relationship
    schema_info.relationships.append(("products", "category_id", "categories", "id"))

    return schema_info


@pytest.fixture(scope="session")  # Added scope
def mock_db_connector():
    """
    Fixture providing a mock database connector.
    """
    connector = MagicMock()

    # Mock the execute_query method
    def execute_query(db_name, query):
        # Return success for any query
        return {
            "success": True,
            "rows": [{"id": 1, "name": "Test"}],
            "row_count": 1,
            "execution_time": 0.1,
        }

    connector.execute_query.side_effect = execute_query

    # Mock the get_schema method
    def get_schema(db_name):
        # Return a simple schema structure
        return {
            "tables": {
                "users": {
                    "columns": {
                        "id": {"type": "INTEGER", "primary_key": True},
                        "name": {"type": "VARCHAR(100)"},
                        "email": {"type": "VARCHAR(255)"},
                    }
                },
                "products": {
                    "columns": {
                        "id": {"type": "INTEGER", "primary_key": True},
                        "name": {"type": "VARCHAR(100)"},
                        "price": {"type": "DECIMAL(10, 2)"},
                    }
                },
            }
        }

    connector.get_schema.side_effect = get_schema

    return connector


@pytest.fixture(scope="session")  # Added scope
def alb_sql_instance(mock_db_connector):
    """
    Fixture providing an instance of the AlbSQL class with a mock connector.
    """
    # Note: This will still attempt to load the configured LLM model
    # which might be slow or fail if the model isn't available/configured correctly.
    # The mock_db_connector handles database interactions during tests.
    try:
        instance = AlbSQL(db_connector=mock_db_connector)
        # Prevent actual model loading/generation during fixture setup if possible,
        # depending on how tests use it. For now, just instantiate.
        return instance
    except Exception as e:
        pytest.fail(f"Failed to initialize AlbSQL instance in fixture: {e}")


@pytest.fixture(scope="session")  # Added scope
def sample_bird_dataset():
    """
    Fixture providing a sample of the BIRD dataset format.
    """
    # Create a small sample dataset in BIRD format
    dataset = [
        {
            "question": "What are the names of users who placed orders in the last month?",
            "db_id": "e_commerce",
            "sql": "SELECT u.name FROM users u JOIN orders o ON u.id = o.user_id WHERE o.order_date >= CURRENT_DATE - INTERVAL '1 month'",
            "tables": ["users", "orders"],
            "schema": {
                "users": ["id", "name", "email"],
                "orders": ["id", "user_id", "order_date", "total_amount"],
            },
        },
        {
            "question": "What is the total amount of orders for each product category?",
            "db_id": "e_commerce",
            "sql": "SELECT c.name, SUM(o.total_amount) FROM categories c JOIN products p ON c.id = p.category_id JOIN order_items oi ON p.id = oi.product_id JOIN orders o ON oi.order_id = o.id GROUP BY c.name",
            "tables": ["categories", "products", "order_items", "orders"],
            "schema": {
                "categories": ["id", "name"],
                "products": ["id", "name", "price", "category_id"],
                "order_items": ["id", "order_id", "product_id", "quantity", "price"],
                "orders": ["id", "user_id", "order_date", "total_amount"],
            },
        },
    ]

    return dataset


@pytest.fixture(scope="session")  # Added scope
def sample_ambiguities():
    """
    Fixture providing sample ambiguities.
    """
    from agents.ambiguity_resolver import Ambiguity, AmbiguityType

    # Create some sample ambiguities
    ambiguities = [
        Ambiguity(
            type=AmbiguityType.COLUMN_REFERENCE,
            description="The column 'name' exists in multiple tables",
            context="In query: 'Find names of all products'",
            options=["products.name", "categories.name"],
            confidence=0.5,
            impact=0.8,
        ),
        Ambiguity(
            type=AmbiguityType.TEMPORAL,
            description="Ambiguous time reference detected",
            context="Matched pattern: 'recent' in query: 'Show me recent orders'",
            options=["Last 7 days", "Last 30 days", "Last 90 days"],
            confidence=0.3,
            impact=0.8,
        ),
    ]

    return ambiguities


@pytest.fixture(scope="session")  # Added scope
def sample_embeddings():
    """
    Fixture providing sample schema embeddings.
    """
    import numpy as np

    # Create sample embeddings
    embeddings = {
        "e_commerce": {
            "table:users": np.random.randn(768),
            "table:products": np.random.randn(768),
            "column:users.id": np.random.randn(768),
            "column:users.name": np.random.randn(768),
            "column:products.id": np.random.randn(768),
            "column:products.name": np.random.randn(768),
        }
    }

    # Normalize embeddings
    for db in embeddings:
        for key in embeddings[db]:
            embeddings[db][key] = embeddings[db][key] / np.linalg.norm(
                embeddings[db][key]
            )

    return embeddings


@pytest.fixture
def bird_dev_sample(tmpdir):
    """
    Fixture providing a sample from BIRD dev dataset.
    Returns the path to the dev.json file.
    """
    # Create a minimal BIRD dev dataset
    dev_data = [
        {
            "question": "How many users have placed at least one order?",
            "db_id": "e_commerce",
            "sql": "SELECT COUNT(DISTINCT user_id) FROM orders",
            "tables": ["orders"],
            "schema": {"orders": ["id", "user_id", "order_date", "total_amount"]},
        },
        {
            "question": "What is the average price of products in each category?",
            "db_id": "e_commerce",
            "sql": "SELECT c.name, AVG(p.price) FROM categories c JOIN products p ON c.id = p.category_id GROUP BY c.name",
            "tables": ["categories", "products"],
            "schema": {
                "categories": ["id", "name"],
                "products": ["id", "name", "price", "category_id"],
            },
        },
    ]

    # Save to a temporary file
    dev_path = tmpdir.join("dev.json")
    with open(dev_path, "w") as f:
        json.dump(dev_data, f)

    return str(dev_path)


@pytest.fixture
def bird_train_sample(tmpdir):
    """
    Fixture providing a sample from BIRD train dataset.
    Returns the path to the train.json file.
    """
    # Create a minimal BIRD train dataset
    train_data = [
        {
            "question": "List all user names and emails",
            "db_id": "e_commerce",
            "sql": "SELECT name, email FROM users",
            "tables": ["users"],
            "schema": {"users": ["id", "name", "email"]},
        },
        {
            "question": "What are the top 5 most expensive products?",
            "db_id": "e_commerce",
            "sql": "SELECT name, price FROM products ORDER BY price DESC LIMIT 5",
            "tables": ["products"],
            "schema": {"products": ["id", "name", "price", "category_id"]},
        },
    ]

    # Save to a temporary file
    train_path = tmpdir.join("train.json")
    with open(train_path, "w") as f:
        json.dump(train_data, f)

    return str(train_path)


@pytest.fixture
def bird_tables_sample(tmpdir):
    """
    Fixture providing a sample tables.json in BIRD format.
    Returns the path to the tables.json file.
    """
    # Create a minimal tables.json
    tables_data = {
        "e_commerce": {
            "table_names": ["users", "products", "categories", "orders", "order_items"],
            "column_names": [
                ["users", "id"],
                ["users", "name"],
                ["users", "email"],
                ["products", "id"],
                ["products", "name"],
                ["products", "price"],
                ["products", "category_id"],
                ["categories", "id"],
                ["categories", "name"],
                ["orders", "id"],
                ["orders", "user_id"],
                ["orders", "order_date"],
                ["orders", "total_amount"],
                ["order_items", "id"],
                ["order_items", "order_id"],
                ["order_items", "product_id"],
                ["order_items", "quantity"],
                ["order_items", "price"],
            ],
            "column_types": [
                "number",
                "text",
                "text",
                "number",
                "text",
                "number",
                "number",
                "number",
                "text",
                "number",
                "number",
                "date",
                "number",
                "number",
                "number",
                "number",
                "number",
                "number",
            ],
            "primary_keys": [
                ["users", "id"],
                ["products", "id"],
                ["categories", "id"],
                ["orders", "id"],
                ["order_items", "id"],
            ],
            "foreign_keys": [
                [["products", "category_id"], ["categories", "id"]],
                [["orders", "user_id"], ["users", "id"]],
                [["order_items", "order_id"], ["orders", "id"]],
                [["order_items", "product_id"], ["products", "id"]],
            ],
        }
    }

    # Save to a temporary file
    tables_path = tmpdir.join("tables.json")
    with open(tables_path, "w") as f:
        json.dump(tables_data, f)

    return str(tables_path)


@pytest.fixture
def baseline_results(tmpdir):
    """
    Fixture providing a sample baseline results file.
    Returns the path to the baseline.json file.
    """
    # Create a minimal baseline results file
    baseline_data = {
        "timestamp": "2025-01-01T00:00:00",
        "model": "gpt-4",
        "metrics": {
            "execution_success_rate": 0.92,
            "syntax_correctness": 0.95,
            "semantic_correctness": 0.89,
            "result_match_rate": 0.87,
            "execution_efficiency": 0.83,
            "overall_score": 0.90,
        },
        "sample_results": {
            "dev_1": {
                "question": "How many users have placed at least one order?",
                "predicted_sql": "SELECT COUNT(DISTINCT user_id) FROM orders",
                "gold_sql": "SELECT COUNT(DISTINCT user_id) FROM orders",
                "execution_success": True,
                "result_match": True,
                "execution_time": 0.05,
            },
            "dev_2": {
                "question": "What is the average price of products in each category?",
                "predicted_sql": "SELECT c.name, AVG(p.price) FROM categories c JOIN products p ON c.id = p.category_id GROUP BY c.name",
                "gold_sql": "SELECT c.name, AVG(p.price) FROM categories c JOIN products p ON c.id = p.category_id GROUP BY c.name",
                "execution_success": True,
                "result_match": True,
                "execution_time": 0.08,
            },
        },
    }

    # Save to a temporary file
    baseline_path = tmpdir.join("baseline.json")
    with open(baseline_path, "w") as f:
        json.dump(baseline_data, f)

    return str(baseline_path)
