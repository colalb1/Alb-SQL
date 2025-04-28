"""
Schema processing utilities for SQL databases.
"""

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from ..config import SPECIAL_TOKENS


@dataclass
class Column:
    """Represents a database column with its properties."""

    name: str
    type: str
    table: str
    is_primary_key: bool = False
    foreign_key: Optional[Tuple[str, str]] = None  # (table_name, column_name)

    def __str__(self) -> str:
        """String representation of the column."""
        result = f"{self.name} ({self.type})"
        if self.is_primary_key:
            result += " PRIMARY KEY"
        if self.foreign_key:
            result += f" REFERENCES {self.foreign_key[0]}({self.foreign_key[1]})"
        return result

    def to_schema_string(self) -> str:
        """Convert column to schema string format."""
        parts = [SPECIAL_TOKENS["column_token"], self.name, self.type]

        if self.is_primary_key:
            parts.append(SPECIAL_TOKENS["primary_key_token"])

        if self.foreign_key:
            parts.append(SPECIAL_TOKENS["foreign_key_token"])
            parts.append(f"{self.foreign_key[0]}.{self.foreign_key[1]}")

        return " ".join(parts)


@dataclass
class Table:
    """Represents a database table with its columns."""

    name: str
    columns: List[Column]

    def __str__(self) -> str:
        """String representation of the table."""
        cols = ", ".join(str(col) for col in self.columns)
        return f"{self.name} ({cols})"

    def to_schema_string(self) -> str:
        """Convert table to schema string format."""
        result = [f"{SPECIAL_TOKENS['table_token']} {self.name}"]
        for col in self.columns:
            result.append(col.to_schema_string())
        return "\n".join(result)


class SchemaProcessor:
    """Processes database schemas from _tables.json files."""

    def __init__(self, tables_json_path: str):
        """Initialize with path to tables JSON file."""
        self.tables_json_path = tables_json_path
        self.tables: Dict[str, Table] = {}
        self.foreign_keys: Dict[str, List[Tuple[str, str, str, str]]] = {}
        self._load_schema()

    def _load_schema(self) -> None:
        """Load schema from tables JSON file."""
        if not os.path.exists(self.tables_json_path):
            raise FileNotFoundError(
                f"Tables JSON file not found: {self.tables_json_path}"
            )

        with open(self.tables_json_path, "r", encoding="utf-8") as f:
            tables_data = json.load(f)

        # First pass: Create tables and columns without foreign keys
        for db_id, db_info in tables_data.items():
            self.foreign_keys[db_id] = []

            for table_info in db_info["table_names_original"]:
                table_id, table_name = table_info
                self.tables[f"{db_id}.{table_name}"] = Table(
                    name=table_name, columns=[]
                )

            # Process primary keys
            primary_keys: Set[int] = set()
            for pk in db_info.get("primary_keys", []):
                primary_keys.add(pk)

            # Process foreign keys
            for fk in db_info.get("foreign_keys", []):
                source_col_id, target_col_id = fk
                source_table_id = db_info["column_names_original"][source_col_id][0]
                source_col_name = db_info["column_names_original"][source_col_id][1]
                target_table_id = db_info["column_names_original"][target_col_id][0]
                target_col_name = db_info["column_names_original"][target_col_id][1]

                source_table_name = db_info["table_names_original"][source_table_id][1]
                target_table_name = db_info["table_names_original"][target_table_id][1]

                self.foreign_keys[db_id].append(
                    (
                        source_table_name,
                        source_col_name,
                        target_table_name,
                        target_col_name,
                    )
                )

            # Add columns to tables
            for col_id, (table_id, col_name) in enumerate(
                db_info["column_names_original"]
            ):
                if table_id == -1:  # Skip special columns like '*'
                    continue

                table_name = db_info["table_names_original"][table_id][1]
                col_type = db_info["column_types"][col_id]

                column = Column(
                    name=col_name,
                    type=col_type,
                    table=table_name,
                    is_primary_key=col_id in primary_keys,
                )

                self.tables[f"{db_id}.{table_name}"].columns.append(column)

        # Second pass: Add foreign key references to columns
        for db_id, fk_list in self.foreign_keys.items():
            for source_table, source_col, target_table, target_col in fk_list:
                # Find the source column and update its foreign key reference
                for col in self.tables[f"{db_id}.{source_table}"].columns:
                    if col.name == source_col:
                        col.foreign_key = (target_table, target_col)
                        break

    def get_table(self, db_id: str, table_name: str) -> Optional[Table]:
        """Get a table by database ID and table name."""
        key = f"{db_id}.{table_name}"
        return self.tables.get(key)

    def get_all_tables(self, db_id: str) -> List[Table]:
        """Get all tables for a specific database."""
        return [
            table for key, table in self.tables.items() if key.startswith(f"{db_id}.")
        ]

    def format_schema_for_model(self, db_id: str) -> str:
        """Format the schema for input to the model."""
        tables = self.get_all_tables(db_id)
        return "\n".join(table.to_schema_string() for table in tables)
