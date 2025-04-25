"""
Example script showing how to use the Hugging Face text-to-SQL generator.

This script provides a simple CLI interface for generating SQL queries
from natural language using local Hugging Face models optimized for text-to-SQL tasks.
"""

import argparse
import json
import logging
import os
import sys
import time

# Add the parent directory to the Python path so we can import from the main project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def setup_logger():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate SQL from natural language using local Hugging Face models."
    )
    parser.add_argument("question", help="Natural language question to convert to SQL")
    parser.add_argument(
        "--schema", help="Path to a schema file or schema string", default=None
    )
    parser.add_argument(
        "--model",
        help="Hugging Face model name",
        default="Salesforce/codet5-base-sql",
    )
    parser.add_argument(
        "--temperature", help="Temperature for generation", type=float, default=0.3
    )
    parser.add_argument(
        "--max-tokens", help="Maximum tokens to generate", type=int, default=256
    )
    parser.add_argument(
        "--candidates", help="Number of SQL candidates to generate", type=int, default=1
    )
    parser.add_argument(
        "--cpu", help="Force using CPU even if GPU is available", action="store_true"
    )
    parser.add_argument(
        "--json", help="Return result in JSON format", action="store_true"
    )
    parser.add_argument(
        "--benchmark", help="Run benchmark with N iterations", type=int, default=0
    )

    return parser.parse_args()


def load_schema(schema_path):
    """Load schema from file or return the input if it's a string."""
    if not schema_path:
        return None

    # If schema_path is a file path, load it
    if os.path.exists(schema_path):
        ext = os.path.splitext(schema_path)[1].lower()

        if ext == ".json":
            with open(schema_path, "r") as f:
                schema_data = json.load(f)

            # Format schema data into a string
            schema_str = format_schema_from_json(schema_data)
            return schema_str

        else:  # Assume it's a plain text schema file
            with open(schema_path, "r") as f:
                return f.read()

    # Otherwise assume schema_path is the actual schema text
    return schema_path


def format_schema_from_json(schema_data):
    """Format schema data from JSON into a readable string."""
    schema_parts = []

    # Handle various schema formats
    if isinstance(schema_data, dict):
        # If schema is a dict with tables
        if "tables" in schema_data:
            for table in schema_data["tables"]:
                table_str = f"Table: {table['name']}\n"
                table_str += "Columns:\n"

                for column in table.get("columns", []):
                    column_info = (
                        f"  - {column['name']} ({column.get('type', 'UNKNOWN')})"
                    )
                    if column.get("primary_key"):
                        column_info += " [PK]"
                    if column.get("foreign_key"):
                        fk = column["foreign_key"]
                        column_info += f" [FK -> {fk['table']}.{fk['column']}]"
                    table_str += column_info + "\n"

                schema_parts.append(table_str)

        # Simple dictionary of table_name -> [columns]
        else:
            for table_name, columns in schema_data.items():
                table_str = f"Table: {table_name}\n"
                table_str += "Columns:\n"

                for column in columns:
                    if isinstance(column, dict):
                        column_info = (
                            f"  - {column['name']} ({column.get('type', 'UNKNOWN')})"
                        )
                    else:
                        column_info = f"  - {column}"
                    table_str += column_info + "\n"

                schema_parts.append(table_str)

    # If schema is a list of tables
    elif isinstance(schema_data, list):
        for table in schema_data:
            if isinstance(table, dict):
                table_str = f"Table: {table['name']}\n"
                table_str += "Columns:\n"

                for column in table.get("columns", []):
                    if isinstance(column, dict):
                        column_info = (
                            f"  - {column['name']} ({column.get('type', 'UNKNOWN')})"
                        )
                    else:
                        column_info = f"  - {column}"
                    table_str += column_info + "\n"

                schema_parts.append(table_str)

    return "\n".join(schema_parts)


def main():
    """Main function."""
    setup_logger()
    args = parse_args()

    # Load schema if provided
    schema_context = load_schema(args.schema)

    # Import the appropriate function based on options
    if args.json:
        from core.llm_sql_generator import generate_sql_json_response as generate_func
    elif args.candidates > 1:
        from core.llm_sql_generator import generate_sql_from_llm as generate_func
    else:
        from core.llm_sql_generator import generate_sql_from_text as generate_func

    try:
        # Configure device
        device = "cpu" if args.cpu else None

        print(f"Generating SQL for: {args.question}")
        print(f"Using model: {args.model}")
        print(f"Temperature: {args.temperature}")
        print(f"Using device: {device or 'auto (CUDA if available, else CPU)'}")

        if schema_context:
            print(f"Schema context length: {len(schema_context)} characters")
            print("Schema preview:")
            # Print the first few lines of the schema
            print("\n".join(schema_context.split("\n")[:5]) + "...")
        else:
            print("No schema context provided")

        # Run benchmark if requested
        if args.benchmark > 0:
            print(f"\n=== Running benchmark with {args.benchmark} iterations ===")
            total_time = 0
            for i in range(args.benchmark):
                start_time = time.time()

                if args.json:
                    result = generate_func(
                        prompt=args.question,
                        schema_context=schema_context,
                        model_name=args.model,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        device=device,
                    )
                elif args.candidates > 1:
                    result = generate_func(
                        question=args.question,
                        schema_context=schema_context,
                        model_name=args.model,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        num_candidates=args.candidates,
                        device=device,
                    )
                else:
                    result = generate_func(
                        prompt=args.question,
                        schema_context=schema_context,
                        model_name=args.model,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        device=device,
                    )

                iter_time = time.time() - start_time
                total_time += iter_time
                print(f"Iteration {i + 1}: {iter_time:.4f}s")

            avg_time = total_time / args.benchmark
            print("\nBenchmark results:")
            print(f"  Total time: {total_time:.4f}s")
            print(f"  Average time per query: {avg_time:.4f}s")
            print(f"  Queries per second: {1 / avg_time:.2f}")

            # Show the last result
            if args.json:
                print(f"\nLast result: {json.dumps(result, indent=2)}")
            else:
                print(f"\nLast result: {result}")

            return 0

        # Standard single run
        start_time = time.time()

        # Generate SQL using the appropriate function
        if args.json:
            result = generate_func(
                prompt=args.question,
                schema_context=schema_context,
                model_name=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                device=device,
            )
            # Print results
            print(f"\n=== Generated SQL (in {time.time() - start_time:.4f}s) ===")
            print(json.dumps(result, indent=2))

        elif args.candidates > 1:
            sql_candidates = generate_func(
                question=args.question,
                schema_context=schema_context,
                model_name=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                num_candidates=args.candidates,
                device=device,
            )

            # Handle both string and list return types
            if isinstance(sql_candidates, str):
                sql_candidates = [sql_candidates]

            # Print results
            print(f"\n=== Generated SQL (in {time.time() - start_time:.4f}s) ===")
            for i, sql in enumerate(sql_candidates):
                print(f"\nCandidate {i + 1}:")
                print(f"{sql}")

            # Validate first SQL
            print("\n=== SQL Validation ===")
            syntax_valid = validate_sql_syntax(sql_candidates[0])
            print(f"Syntax check: {'PASS' if syntax_valid else 'FAIL'}")

        else:
            sql = generate_func(
                prompt=args.question,
                schema_context=schema_context,
                model_name=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                device=device,
            )

            # Print results
            print(f"\n=== Generated SQL (in {time.time() - start_time:.4f}s) ===")
            print(sql)

            # Validate SQL
            print("\n=== SQL Validation ===")
            syntax_valid = validate_sql_syntax(sql)
            print(f"Syntax check: {'PASS' if syntax_valid else 'FAIL'}")

    except Exception as e:
        print(f"Error generating SQL: {e}")
        return 1

    return 0


def validate_sql_syntax(sql):
    """
    Simple SQL syntax validator.

    This is a very basic validator that just checks for common SQL syntax issues.
    A more robust validator would use a proper SQL parser.

    Args:
        sql (str): SQL query to validate

    Returns:
        bool: True if syntax appears valid, False otherwise
    """
    # If empty or None, not valid
    if not sql:
        return False

    # Convert to lowercase for easier checks
    sql_lower = sql.lower()

    # Check for basic SQL keywords
    has_select = "select" in sql_lower
    has_from = "from" in sql_lower

    # Check for basic syntax errors
    unbalanced_parens = sql.count("(") != sql.count(")")
    unbalanced_quotes = sql.count("'") % 2 != 0

    # For INSERT, UPDATE, DELETE checks
    if "insert" in sql_lower:
        has_insert_into = "insert into" in sql_lower
        has_values = "values" in sql_lower
        return (
            has_insert_into
            and has_values
            and not unbalanced_parens
            and not unbalanced_quotes
        )

    elif "update" in sql_lower:
        has_set = "set" in sql_lower
        return has_set and not unbalanced_parens and not unbalanced_quotes

    elif "delete" in sql_lower:
        has_delete_from = "delete from" in sql_lower
        return has_delete_from and not unbalanced_parens and not unbalanced_quotes

    # For SELECT checks
    else:
        return (
            has_select and has_from and not unbalanced_parens and not unbalanced_quotes
        )


if __name__ == "__main__":
    sys.exit(main())
