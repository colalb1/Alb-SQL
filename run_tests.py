#!/usr/bin/env python
"""
Test Runner for Alb-SQL Testing Framework

This script runs the testing framework for the Alb-SQL text-to-SQL system.
It provides command-line options for running different types of tests.
"""

import argparse
import os
import sys

import pytest


def main():
    """Run the Alb-SQL testing framework."""
    parser = argparse.ArgumentParser(description="Run Alb-SQL tests")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument(
        "--integration", action="store_true", help="Run integration tests only"
    )
    parser.add_argument(
        "--regression", action="store_true", help="Run regression tests only"
    )
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument(
        "--coverage", action="store_true", help="Generate coverage report"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--file", "-f", type=str, help="Run specific test file")
    parser.add_argument(
        "--dev-data",
        type=str,
        help="Path to BIRD dev dataset",
        default="data/dev_20240627/dev.json",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        help="Path to baseline results",
        default="tests/regression/baseline.json",
    )

    args = parser.parse_args()

    # Set environment variables for test data paths
    os.environ["BIRD_DEV_PATH"] = args.dev_data
    os.environ["BASELINE_RESULTS_PATH"] = args.baseline

    # Build pytest arguments
    pytest_args = []

    # Set verbosity
    if args.verbose:
        pytest_args.append("-v")

    # Add coverage if requested
    if args.coverage:
        pytest_args.extend(["--cov=.", "--cov-report=term", "--cov-report=html"])

    # Determine which tests to run
    if args.file:
        pytest_args.append(args.file)
    elif args.unit:
        pytest_args.append("tests/unit/")
    elif args.integration:
        pytest_args.append("tests/integration/")
    elif args.regression:
        pytest_args.append("tests/regression/")
    elif args.all or not any([args.unit, args.integration, args.regression, args.file]):
        # Run all tests if --all is specified or no specific test type is requested
        pytest_args.append("tests/")

    # Add additional pytest arguments
    pytest_args.extend(["-xvs"])

    print(f"Running tests with arguments: {' '.join(pytest_args)}")

    # Run pytest with the specified arguments
    return pytest.main(pytest_args)


if __name__ == "__main__":
    sys.exit(main())
