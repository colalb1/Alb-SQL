"""
Schema Analyzer Agent Module

This module implements an agent that analyzes database schemas to extract
useful information for SQL generation, such as relationships, constraints,
and semantic patterns.
"""

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class ColumnInfo:
    """Information about a database column."""

    name: str
    data_type: str
    nullable: bool = True
    is_primary_key: bool = False
    is_foreign_key: bool = False
    referenced_table: Optional[str] = None
    referenced_column: Optional[str] = None
    description: Optional[str] = None
    sample_values: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    stats: Dict[str, Union[float, int, str]] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)  # Added missing tags attribute


@dataclass
class TableInfo:
    """Information about a database table."""

    name: str
    columns: Dict[str, ColumnInfo] = field(default_factory=dict)
    primary_keys: List[str] = field(default_factory=list)
    foreign_keys: Dict[str, Tuple[str, str]] = field(default_factory=dict)
    description: Optional[str] = None
    sample_row_count: int = 0
    tags: List[str] = field(default_factory=list)


@dataclass
class SchemaInfo:
    """Information about a database schema."""

    name: str
    tables: Dict[str, TableInfo] = field(default_factory=dict)
    relationships: List[Tuple[str, str, str, str]] = field(default_factory=list)
    domain: Optional[str] = None
    description: Optional[str] = None


class SchemaAnalyzerAgent:
    """
    An agent that analyzes database schemas and extracts useful information
    for SQL generation.
    """

    def __init__(self, db_connector=None, cache_dir: Optional[str] = None):
        """
        Initialize the SchemaAnalyzerAgent.

        Args:
            db_connector: Connector to interact with databases.
            cache_dir (Optional[str]): Directory to cache schema information.
        """
        self.db_connector = db_connector
        self.cache_dir = cache_dir
        self.schema_cache: Dict[str, SchemaInfo] = {}
        # Patterns for common column types
        self.date_patterns = [
            r"date",
            r"time",
            r"year",
            r"month",
            r"day",
            r"created_at",
            r"updated_at",
        ]
        self.id_patterns = [r"_id$", r"^id_", r"^id$", r"uuid", r"guid", r"key"]
        self.name_patterns = [r"name", r"title", r"label", r"caption", r"description"]
        self.quantity_patterns = [
            r"count",
            r"amount",
            r"quantity",
            r"total",
            r"sum",
            r"avg",
        ]

    def analyze_schema(self, db_name: str, refresh: bool = False) -> SchemaInfo:
        """
        Analyze a database schema and return information about it.

        Args:
            db_name (str): Database name.
            refresh (bool): Whether to refresh cached schema information.

        Returns:
            SchemaInfo containing schema information.
        """
        # Check if schema is in cache
        if not refresh and db_name in self.schema_cache:
            logger.info(f"Using cached schema for {db_name}")
            return self.schema_cache[db_name]

        logger.info(f"Analyzing schema for {db_name}")

        # This would use the actual database connector
        # For now, we'll simulate schema extraction
        schema_info = self._extract_schema(db_name)

        # Analyze relationships
        self._analyze_relationships(schema_info)

        # Analyze column semantics
        self._analyze_column_semantics(schema_info)

        # Cache schema
        self.schema_cache[db_name] = schema_info

        return schema_info

    def _extract_schema(self, db_name: str) -> SchemaInfo:
        """
        Extract schema information from a database.

        Args:
            db_name (str): Database name.

        Returns:
            SchemaInfo containing schema information.
        """
        # This would use the actual database connector
        # For now, we'll create some placeholder schema information
        schema_info = SchemaInfo(name=db_name)

        # For demonstration, create a simple schema
        if db_name == "e_commerce":
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
                stats={"min": 1, "max": 1000, "avg": 500},
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
            users_table.columns["created_at"] = ColumnInfo(
                name="created_at",
                data_type="TIMESTAMP",
                nullable=False,
                sample_values=["2023-01-01 00:00:00", "2023-01-02 00:00:00"],
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

            # Orders table
            orders_table = TableInfo(
                name="orders",
                description="Table storing order information",
                sample_row_count=2000,
            )
            orders_table.columns["id"] = ColumnInfo(
                name="id",
                data_type="INTEGER",
                nullable=False,
                is_primary_key=True,
                sample_values=["1", "2", "3"],
            )
            orders_table.columns["user_id"] = ColumnInfo(
                name="user_id",
                data_type="INTEGER",
                nullable=False,
                is_foreign_key=True,
                referenced_table="users",
                referenced_column="id",
                sample_values=["1", "2"],
            )
            orders_table.columns["order_date"] = ColumnInfo(
                name="order_date",
                data_type="DATE",
                nullable=False,
                sample_values=["2023-01-01", "2023-01-02"],
            )
            orders_table.columns["total_amount"] = ColumnInfo(
                name="total_amount",
                data_type="DECIMAL(10, 2)",
                nullable=False,
                sample_values=["1299.99", "599.99"],
            )
            orders_table.primary_keys = ["id"]
            orders_table.foreign_keys["user_id"] = ("users", "id")
            schema_info.tables["orders"] = orders_table

            # Order Items table
            order_items_table = TableInfo(
                name="order_items",
                description="Table storing order item information",
                sample_row_count=5000,
            )
            order_items_table.columns["id"] = ColumnInfo(
                name="id",
                data_type="INTEGER",
                nullable=False,
                is_primary_key=True,
                sample_values=["1", "2", "3"],
            )
            order_items_table.columns["order_id"] = ColumnInfo(
                name="order_id",
                data_type="INTEGER",
                nullable=False,
                is_foreign_key=True,
                referenced_table="orders",
                referenced_column="id",
                sample_values=["1", "1", "2"],
            )
            order_items_table.columns["product_id"] = ColumnInfo(
                name="product_id",
                data_type="INTEGER",
                nullable=False,
                is_foreign_key=True,
                referenced_table="products",
                referenced_column="id",
                sample_values=["1", "2"],
            )
            order_items_table.columns["quantity"] = ColumnInfo(
                name="quantity",
                data_type="INTEGER",
                nullable=False,
                sample_values=["1", "2"],
            )
            order_items_table.columns["price"] = ColumnInfo(
                name="price",
                data_type="DECIMAL(10, 2)",
                nullable=False,
                sample_values=["999.99", "599.99"],
            )
            order_items_table.primary_keys = ["id"]
            order_items_table.foreign_keys["order_id"] = ("orders", "id")
            order_items_table.foreign_keys["product_id"] = ("products", "id")
            schema_info.tables["order_items"] = order_items_table

            # Categories table
            categories_table = TableInfo(
                name="categories",
                description="Table storing product categories",
                sample_row_count=10,
            )
            categories_table.columns["id"] = ColumnInfo(
                name="id",
                data_type="INTEGER",
                nullable=False,
                is_primary_key=True,
                sample_values=["1", "2", "3"],
            )
            categories_table.columns["name"] = ColumnInfo(
                name="name",
                data_type="VARCHAR(50)",
                nullable=False,
                sample_values=["Electronics", "Clothing"],
            )
            categories_table.primary_keys = ["id"]
            schema_info.tables["categories"] = categories_table

        elif db_name == "healthcare":
            # Add healthcare schema tables here...
            patients_table = TableInfo(
                name="patients",
                description="Table storing patient information",
                sample_row_count=5000,
            )
            patients_table.columns["id"] = ColumnInfo(
                name="id",
                data_type="INTEGER",
                nullable=False,
                is_primary_key=True,
                sample_values=["1", "2", "3"],
            )
            patients_table.columns["name"] = ColumnInfo(
                name="name",
                data_type="VARCHAR(100)",
                nullable=False,
                sample_values=["John Smith", "Jane Doe"],
            )
            patients_table.columns["date_of_birth"] = ColumnInfo(
                name="date_of_birth",
                data_type="DATE",
                nullable=False,
                sample_values=["1980-01-01", "1990-01-01"],
            )
            patients_table.columns["gender"] = ColumnInfo(
                name="gender",
                data_type="VARCHAR(10)",
                nullable=False,
                sample_values=["Male", "Female", "Other"],
            )
            patients_table.primary_keys = ["id"]
            schema_info.tables["patients"] = patients_table

            # More tables would be defined for a real system...
        else:
            logger.warning(f"No predefined schema for {db_name}, creating empty schema")

        return schema_info

    def _analyze_relationships(self, schema_info: SchemaInfo) -> None:
        """
        Analyze relationships between tables in a schema.

        Args:
            schema_info (SchemaInfo): SchemaInfo to analyze.
        """
        # Extract relationships from foreign keys
        relationships = []
        for table_name, table_info in schema_info.tables.items():
            for fk_column, (ref_table, ref_column) in table_info.foreign_keys.items():
                relationships.append((table_name, fk_column, ref_table, ref_column))

        # Store relationships
        schema_info.relationships = relationships

        logger.info(f"Found {len(relationships)} relationships in {schema_info.name}")

    def _analyze_column_semantics(self, schema_info: SchemaInfo) -> None:
        """
        Analyze column semantics to identify common patterns.

        Args:
            schema_info (SchemaInfo): SchemaInfo to analyze.
        """
        for table_name, table_info in schema_info.tables.items():
            for column_name, column_info in table_info.columns.items():
                # Check for date patterns
                if any(
                    re.search(pattern, column_name, re.IGNORECASE)
                    for pattern in self.date_patterns
                ):
                    column_info.tags = column_info.tags + ["date"]

                # Check for ID patterns
                if any(
                    re.search(pattern, column_name, re.IGNORECASE)
                    for pattern in self.id_patterns
                ):
                    column_info.tags = column_info.tags + ["id"]

                # Check for name patterns
                if any(
                    re.search(pattern, column_name, re.IGNORECASE)
                    for pattern in self.name_patterns
                ):
                    column_info.tags = column_info.tags + ["name"]

                # Check for quantity patterns
                if any(
                    re.search(pattern, column_name, re.IGNORECASE)
                    for pattern in self.quantity_patterns
                ):
                    column_info.tags = column_info.tags + ["quantity"]

                # Infer data types
                if "INT" in column_info.data_type.upper():
                    column_info.tags = column_info.tags + ["numeric", "integer"]
                elif any(
                    t in column_info.data_type.upper()
                    for t in ["DECIMAL", "FLOAT", "DOUBLE"]
                ):
                    column_info.tags = column_info.tags + ["numeric", "float"]
                elif any(
                    t in column_info.data_type.upper()
                    for t in ["CHAR", "TEXT", "VARCHAR"]
                ):
                    column_info.tags = column_info.tags + ["text"]
                elif any(
                    t in column_info.data_type.upper()
                    for t in ["DATE", "TIME", "TIMESTAMP"]
                ):
                    column_info.tags = column_info.tags + ["date"]
                elif "BOOL" in column_info.data_type.upper():
                    column_info.tags = column_info.tags + ["boolean"]

    def generate_join_conditions(
        self, schema_info: SchemaInfo, tables: List[str]
    ) -> List[str]:
        """
        Generate SQL JOIN conditions for a set of tables.

        Args:
            schema_info (SchemaInfo): SchemaInfo containing schema information.
            tables (List[str]): List of table names to join.

        Returns:
            List of JOIN conditions.
        """
        if len(tables) < 2:
            return []

        # Create a graph of table relationships
        graph = defaultdict(list)
        for src_table, src_col, dst_table, dst_col in schema_info.relationships:
            graph[src_table].append(
                (dst_table, f"{src_table}.{src_col} = {dst_table}.{dst_col}")
            )
            graph[dst_table].append(
                (src_table, f"{src_table}.{src_col} = {dst_table}.{dst_col}")
            )

        # Find joins that connect all tables
        joins = []
        visited = set([tables[0]])
        remaining = set(tables[1:])

        while remaining:
            for src in list(visited):
                for dst, condition in graph[src]:
                    if dst in remaining:
                        joins.append(condition)
                        visited.add(dst)
                        remaining.remove(dst)
                        break

        return joins

    def infer_column_constraints(
        self, schema_info: SchemaInfo
    ) -> Dict[str, Dict[str, str]]:
        """
        Infer constraints for columns based on their names and data types.

        Args:
            schema_info (SchemaInfo): SchemaInfo containing schema information.

        Returns:
            Dictionary mapping table.column to constraint expressions.
        """
        constraints = {}
        for table_name, table_info in schema_info.tables.items():
            constraints[table_name] = {}
            for column_name, column_info in table_info.columns.items():
                # Basic type constraints
                if "INT" in column_info.data_type.upper():
                    if "id" in column_name.lower() or column_info.is_primary_key:
                        constraints[table_name][column_name] = "INTEGER > 0"
                    else:
                        constraints[table_name][column_name] = "INTEGER"
                elif "DATE" in column_info.data_type.upper():
                    constraints[table_name][column_name] = "DATE"
                # Domain-specific constraints
                if "age" in column_name.lower():
                    constraints[table_name][column_name] = "INTEGER BETWEEN 0-120"
                elif "email" in column_name.lower():
                    constraints[table_name][column_name] = "TEXT containing '@'"
                elif "price" in column_name.lower() or "amount" in column_name.lower():
                    constraints[table_name][column_name] = "DECIMAL >= 0"
                elif (
                    "quantity" in column_name.lower() or "count" in column_name.lower()
                ):
                    constraints[table_name][column_name] = "INTEGER >= 0"

        return constraints

    def generate_schema_summary(
        self,
        schema_info: SchemaInfo,
        target_tables: Optional[List[str]] = None,
        detail_level: str = "medium",
        include_sample_data: bool = True,  # Added parameter
    ) -> str:
        """
        Generate a human-readable summary of the schema.

        Args:
            schema_info (SchemaInfo): SchemaInfo containing schema information.
            target_tables (Optional[List[str]]): Optional list of tables to
                include in the summary.
            detail_level (str): Level of detail to include
                ('low', 'medium', 'high').
            include_sample_data (bool): Whether to include sample data for high detail.

        Returns:
            Human-readable schema summary.
        """
        if target_tables is None:
            target_tables = list(schema_info.tables.keys())

        # Filter tables
        tables_to_include = {
            name: info
            for name, info in schema_info.tables.items()
            if name in target_tables
        }

        if not tables_to_include:
            return "No tables found."

        summary = [f"Schema: {schema_info.name}"]

        # Include tables and their columns
        for table_name, table_info in tables_to_include.items():
            summary.append(f"\nTable: {table_name}")
            if table_info.description and detail_level != "low":
                summary.append(f"Description: {table_info.description}")
            if detail_level != "low":
                summary.append(f"Row count (sample): ~{table_info.sample_row_count}")

            summary.append("Columns:")
            for column_name, column_info in table_info.columns.items():
                line = f"  - {column_name} ({column_info.data_type})"
                if column_info.is_primary_key:
                    line += " [PK]"
                if column_info.is_foreign_key:
                    line += (
                        f" [FK -> {column_info.referenced_table}."
                        f"{column_info.referenced_column}]"
                    )
                if not column_info.nullable:
                    line += " [NOT NULL]"
                summary.append(line)

                # Add sample values for high detail level if requested
                if (
                    detail_level == "high"
                    and include_sample_data
                    and column_info.sample_values
                ):
                    sample_str = ", ".join(column_info.sample_values[:3])
                    summary.append(f"    Sample values: {sample_str}")

        # Include relationships for medium and high detail levels
        if detail_level != "low" and schema_info.relationships:
            summary.append("\nRelationships:")
            for src_table, src_col, dst_table, dst_col in schema_info.relationships:
                if src_table in target_tables and dst_table in target_tables:
                    summary.append(
                        f"  - {src_table}.{src_col} -> " f"{dst_table}.{dst_col}"
                    )

        return "\n".join(summary)


if __name__ == "__main__":
    # Example usage
    analyzer = SchemaAnalyzerAgent()

    # Analyze e-commerce schema
    e_commerce_schema = analyzer.analyze_schema("e_commerce")
    print(analyzer.generate_schema_summary(e_commerce_schema))

    # Generate join conditions
    join_conditions = analyzer.generate_join_conditions(
        e_commerce_schema, ["orders", "users", "order_items"]
    )
    print("\nJoin Conditions:")
    for condition in join_conditions:
        print(f"  {condition}")

    # Infer column constraints
    constraints = analyzer.infer_column_constraints(e_commerce_schema)
    print("\nColumn Constraints:")
    for table, cols in constraints.items():
        for col, constraint in cols.items():
            print(f"  {table}.{col}: {constraint}")
