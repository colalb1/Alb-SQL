"""
Unit tests for the SchemaAnalogizer class.

This module tests the SchemaAnalogizer class, which finds analogies between schema
elements across different databases.
"""

import os
import tempfile

import numpy as np

from core.schema_analogizer import SchemaAnalogizer


class TestSchemaAnalogizer:
    """Tests for the SchemaAnalogizer class."""

    def test_initialization(self):
        """Test that the SchemaAnalogizer initializes with correct default values."""
        analogizer = SchemaAnalogizer()

        assert analogizer.embedding_dim == 768
        assert analogizer.similarity_threshold == 0.85
        assert analogizer.schema_embeddings == {}
        assert analogizer.schema_relationships == {}
        assert analogizer.analogy_cache == {}

    def test_initialization_with_custom_values(self):
        """Test initialization with custom values."""
        analogizer = SchemaAnalogizer(embedding_dim=512, similarity_threshold=0.75)

        assert analogizer.embedding_dim == 512
        assert analogizer.similarity_threshold == 0.75

    def test_compute_embedding(self):
        """Test computing embeddings for schema elements."""
        analogizer = SchemaAnalogizer(embedding_dim=10)

        # Compute embedding
        embedding = analogizer.compute_embedding(
            db_name="test_db",
            element_name="users",
            element_type="table",
            description="Table storing user information",
        )

        # Check the embedding
        assert embedding.shape == (10,)
        assert np.isclose(np.linalg.norm(embedding), 1.0)  # Should be normalized

        # Check that the embedding was stored
        assert "test_db" in analogizer.schema_embeddings
        assert "table:users" in analogizer.schema_embeddings["test_db"]
        assert np.array_equal(
            analogizer.schema_embeddings["test_db"]["table:users"], embedding
        )

    def test_save_and_load_embeddings(self):
        """Test saving and loading embeddings to/from a file."""
        analogizer = SchemaAnalogizer(embedding_dim=10)

        # Compute some embeddings
        analogizer.compute_embedding(
            db_name="test_db",
            element_name="users",
            element_type="table",
            description="Table storing user information",
        )
        analogizer.compute_embedding(
            db_name="test_db",
            element_name="products",
            element_type="table",
            description="Table storing product information",
        )

        # Create a temporary file for testing
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            # Save embeddings
            analogizer.save_embeddings(temp_path)

            # Create a new analogizer and load the embeddings
            new_analogizer = SchemaAnalogizer(embedding_dim=10)
            new_analogizer.load_embeddings(temp_path)

            # Check that the embeddings were loaded correctly
            assert "test_db" in new_analogizer.schema_embeddings
            assert "table:users" in new_analogizer.schema_embeddings["test_db"]
            assert "table:products" in new_analogizer.schema_embeddings["test_db"]

            # Check that the actual embeddings match
            assert np.array_equal(
                analogizer.schema_embeddings["test_db"]["table:users"],
                new_analogizer.schema_embeddings["test_db"]["table:users"],
            )
            assert np.array_equal(
                analogizer.schema_embeddings["test_db"]["table:products"],
                new_analogizer.schema_embeddings["test_db"]["table:products"],
            )

        finally:
            # Clean up
            os.unlink(temp_path)

    def test_find_analogies(self):
        """Test finding analogies between schema elements."""
        analogizer = SchemaAnalogizer(embedding_dim=10, similarity_threshold=0.5)

        # Create controlled embeddings for testing
        # We'll make 'users' and 'customers' have similar embeddings
        users_embedding = np.ones(10) / np.sqrt(10)  # Normalized vector of ones
        customers_embedding = 0.9 * users_embedding + 0.1 * np.random.randn(10)
        customers_embedding = customers_embedding / np.linalg.norm(customers_embedding)

        # Add the embeddings directly to the schema_embeddings
        analogizer.schema_embeddings = {
            "db1": {
                "table:users": users_embedding,
                "table:products": np.random.randn(10),  # Random embedding
            },
            "db2": {
                "table:customers": customers_embedding,
                "table:items": np.random.randn(10),  # Random embedding
            },
        }

        # Normalize random embeddings
        for db in analogizer.schema_embeddings:
            for key in analogizer.schema_embeddings[db]:
                analogizer.schema_embeddings[db][key] = analogizer.schema_embeddings[
                    db
                ][key] / np.linalg.norm(analogizer.schema_embeddings[db][key])

        # Find analogies
        analogies = analogizer.find_analogies("db1", "table:users", "db2")

        # Check that 'customers' is found as an analogy for 'users'
        assert len(analogies) > 0
        assert analogies[0][0] == "table:customers"
        assert analogies[0][1] > 0.5  # Similarity should be above threshold

    def test_build_schema_relationship_graph(self):
        """Test building a schema relationship graph."""
        analogizer = SchemaAnalogizer()

        # Set up some schema embeddings
        analogizer.schema_embeddings = {
            "test_db": {
                "table:users": np.random.randn(10),
                "column:users.id": np.random.randn(10),
                "column:users.name": np.random.randn(10),
                "table:orders": np.random.randn(10),
                "column:orders.id": np.random.randn(10),
                "column:orders.user_id": np.random.randn(10),
            }
        }

        # Build relationship graph
        graph = analogizer.build_schema_relationship_graph("test_db")

        # Check relationships
        assert "column:users.id" in graph
        assert "column:users.name" in graph
        assert "table:users" in graph
        assert "column:orders.id" in graph
        assert "column:orders.user_id" in graph
        assert "table:orders" in graph

        # Check specific relationships
        for rel in graph["table:users"]:
            if rel[0] == "column:users.id":
                assert rel[1] == "has_column"
                assert rel[2] == 1.0

        for rel in graph["column:users.id"]:
            if rel[0] == "table:users":
                assert rel[1] == "belongs_to"
                assert rel[2] == 1.0

    def test_get_schema_analogy_examples(self):
        """Test getting schema analogy examples."""
        analogizer = SchemaAnalogizer(embedding_dim=10, similarity_threshold=0.5)

        # Create controlled embeddings for testing
        users_embedding = np.ones(10) / np.sqrt(10)
        customers_embedding = 0.9 * users_embedding + 0.1 * np.random.randn(10)
        customers_embedding = customers_embedding / np.linalg.norm(customers_embedding)

        # Add the embeddings
        analogizer.schema_embeddings = {
            "db1": {
                "table:users": users_embedding,
                "table:products": np.random.randn(10),
            },
            "db2": {
                "table:customers": customers_embedding,
                "table:items": np.random.randn(10),
            },
        }

        # Normalize random embeddings
        for db in analogizer.schema_embeddings:
            for key in analogizer.schema_embeddings[db]:
                analogizer.schema_embeddings[db][key] = analogizer.schema_embeddings[
                    db
                ][key] / np.linalg.norm(analogizer.schema_embeddings[db][key])

        # Get schema analogy examples
        examples = analogizer.get_schema_analogy_examples("db1")

        # Check examples
        assert len(examples) > 0
        assert "source_db" in examples[0]
        assert "source_element" in examples[0]
        assert "target_db" in examples[0]
        assert "target_element" in examples[0]
        assert "similarity" in examples[0]

    def test_find_analogy_chains(self):
        """Test finding chains of analogies between databases."""
        analogizer = SchemaAnalogizer(embedding_dim=10, similarity_threshold=0.5)

        # Create controlled embeddings for testing
        users_embedding = np.ones(10) / np.sqrt(10)
        customers_embedding = 0.9 * users_embedding + 0.1 * np.random.randn(10)
        customers_embedding = customers_embedding / np.linalg.norm(customers_embedding)

        # Add the embeddings directly to the schema_embeddings
        analogizer.schema_embeddings = {
            "db1": {
                "table:users": users_embedding,
                "table:products": np.random.randn(10),  # Random embedding
            },
            "db2": {
                "table:customers": customers_embedding,
                "table:items": np.random.randn(10),  # Random embedding
            },
        }

        # Normalize random embeddings
        for db in analogizer.schema_embeddings:
            for key in analogizer.schema_embeddings[db]:
                analogizer.schema_embeddings[db][key] = analogizer.schema_embeddings[
                    db
                ][key] / np.linalg.norm(analogizer.schema_embeddings[db][key])

        # Find analogy chains
        chains = analogizer.find_analogy_chains("db1", "table:users", "db2")

        # Check that chains were found
        assert len(chains) > 0

        # Check the structure of the first chain
        first_chain = chains[0]
        assert len(first_chain) >= 2  # Should have at least source and target
        assert first_chain[0][0] == "table:users"  # First element is source
        assert first_chain[0][1] == "db1"  # First DB is source
        assert first_chain[-1][1] == "db2"  # Last DB is target
