"""
Data Loading Utilities

This module provides utilities for loading and processing BIRD-SQL dataset files
and other test data needed for the testing framework.
"""

import json
import os
import sqlite3
import zipfile
from typing import Any, Dict, List, Optional


def load_bird_json(file_path: str) -> List[Dict[str, Any]]:
    """
    Load BIRD dataset from a JSON file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        List of samples from the BIRD dataset.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_bird_tables(file_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load BIRD tables schema from a JSON file.

    Args:
        file_path (str): Path to the tables.json file.

    Returns:
        Dictionary mapping database IDs to table schema information.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def extract_bird_databases(zip_path: str, extract_dir: str) -> List[str]:
    """
    Extract database files from a BIRD dataset zip archive.

    Args:
        zip_path (str): Path to the ZIP file containing databases.
        extract_dir (str): Directory to extract files to.

    Returns:
        List of paths to extracted database files.
    """
    # Create extraction directory if it doesn't exist
    os.makedirs(extract_dir, exist_ok=True)

    # Extract files
    extracted_files = []
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for file_info in zip_ref.infolist():
            if file_info.filename.endswith(".sqlite"):
                zip_ref.extract(file_info, extract_dir)
                extracted_files.append(os.path.join(extract_dir, file_info.filename))

    return extracted_files


def get_samples_by_db(samples: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group BIRD samples by database ID.

    Args:
        samples (List[Dict[str, Any]]): List of BIRD samples.

    Returns:
        Dictionary mapping database IDs to lists of samples for that database.
    """
    by_db = {}
    for sample in samples:
        db_id = sample.get("db_id")
        if not db_id:
            continue

        if db_id not in by_db:
            by_db[db_id] = []
        by_db[db_id].append(sample)

    return by_db


def get_sample_with_execution_info(
    sample: Dict[str, Any], db_path: str
) -> Dict[str, Any]:
    """
    Execute a sample's gold SQL and extract execution information.

    Args:
        sample (Dict[str, Any]): BIRD sample with 'sql' field.
        db_path (str): Path to the corresponding SQLite database.

    Returns:
        Sample with added execution information.
    """
    if not os.path.exists(db_path):
        return {
            **sample,
            "execution": {"success": False, "error": "Database file not found"},
        }

    gold_sql = sample.get("sql")
    if not gold_sql:
        return {
            **sample,
            "execution": {"success": False, "error": "No SQL query in sample"},
        }

    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get schema information
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]

        table_schemas = {}
        for table in tables:
            cursor.execute(f"PRAGMA table_info({table});")
            columns = [{"name": row[1], "type": row[2]} for row in cursor.fetchall()]
            table_schemas[table] = columns

        # Execute the query
        start_time = os.times().user
        cursor.execute(gold_sql)
        end_time = os.times().user
        execution_time = end_time - start_time

        # Fetch results (limiting to 100 rows to avoid memory issues)
        results = cursor.fetchmany(100)
        column_names = [description[0] for description in cursor.description]

        # Format results as a list of dictionaries
        formatted_results = []
        for row in results:
            formatted_results.append(
                {column_names[i]: row[i] for i in range(len(column_names))}
            )

        # Close connection
        conn.close()

        # Add execution information to the sample
        execution_info = {
            "success": True,
            "results": formatted_results,
            "row_count": len(formatted_results),
            "schema": table_schemas,
            "execution_time": execution_time,
        }

        return {**sample, "execution": execution_info}

    except sqlite3.Error as e:
        return {**sample, "execution": {"success": False, "error": str(e)}}
    except Exception as e:
        return {
            **sample,
            "execution": {"success": False, "error": f"Unexpected error: {str(e)}"},
        }


def prepare_test_samples(
    json_path: str,
    db_dir: str,
    tables_path: Optional[str] = None,
    sample_limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Prepare test samples with execution information.

    Args:
        json_path (str): Path to the JSON file with samples.
        db_dir (str): Directory containing database files.
        tables_path (Optional[str]): Path to tables.json file.
        sample_limit (Optional[int]): Maximum number of samples to prepare.

    Returns:
        List of samples with execution information.
    """
    # Load samples
    samples = load_bird_json(json_path)

    # Apply sample limit if specified
    if sample_limit is not None:
        samples = samples[:sample_limit]

    # Load tables schema if available
    tables_schema = {}
    if tables_path and os.path.exists(tables_path):
        tables_schema = load_bird_tables(tables_path)

    # Prepare samples with execution information
    prepared_samples = []
    for sample in samples:
        db_id = sample.get("db_id")
        if not db_id:
            continue

        # Find the corresponding database file
        db_path = None
        for filename in os.listdir(db_dir):
            if filename.startswith(db_id) and filename.endswith(".sqlite"):
                db_path = os.path.join(db_dir, filename)
                break

        if not db_path:
            # Skip samples without a database file
            continue

        # Add tables schema if available
        if db_id in tables_schema:
            sample["tables_schema"] = tables_schema[db_id]

        # Get sample with execution information
        prepared_sample = get_sample_with_execution_info(sample, db_path)
        prepared_samples.append(prepared_sample)

    return prepared_samples


def get_test_coverage_stats(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate test coverage statistics for a set of samples.

    Args:
        samples (List[Dict[str, Any]]): List of samples.

    Returns:
        Dictionary with test coverage statistics.
    """
    stats = {
        "total_samples": len(samples),
        "databases": set(),
        "tables": set(),
        "sql_operations": {
            "select": 0,
            "where": 0,
            "group_by": 0,
            "having": 0,
            "order_by": 0,
            "limit": 0,
            "join": 0,
            "union": 0,
            "intersect": 0,
            "except": 0,
            "subquery": 0,
        },
    }

    for sample in samples:
        # Track databases
        if "db_id" in sample:
            stats["databases"].add(sample["db_id"])

        # Track tables
        if "tables" in sample:
            for table in sample["tables"]:
                stats["tables"].add(table)

        # Track SQL operations
        if "sql" in sample:
            sql = sample["sql"].upper()

            if "SELECT" in sql:
                stats["sql_operations"]["select"] += 1
            if "WHERE" in sql:
                stats["sql_operations"]["where"] += 1
            if "GROUP BY" in sql:
                stats["sql_operations"]["group_by"] += 1
            if "HAVING" in sql:
                stats["sql_operations"]["having"] += 1
            if "ORDER BY" in sql:
                stats["sql_operations"]["order_by"] += 1
            if "LIMIT" in sql:
                stats["sql_operations"]["limit"] += 1
            if "JOIN" in sql:
                stats["sql_operations"]["join"] += 1
            if "UNION" in sql:
                stats["sql_operations"]["union"] += 1
            if "INTERSECT" in sql:
                stats["sql_operations"]["intersect"] += 1
            if "EXCEPT" in sql:
                stats["sql_operations"]["except"] += 1
            if "(" in sql and "SELECT" in sql.split("(", 1)[1]:
                stats["sql_operations"]["subquery"] += 1

    # Convert sets to counts
    stats["unique_databases"] = len(stats["databases"])
    stats["unique_tables"] = len(stats["tables"])
    stats["databases"] = list(stats["databases"])
    stats["tables"] = list(stats["tables"])

    return stats


if __name__ == "__main__":
    # Example usage
    print("Data loading utilities for BIRD-SQL dataset")
