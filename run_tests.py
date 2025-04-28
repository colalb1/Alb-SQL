#!/usr/bin/env python
"""
Script to run all tests for the BIRD-SQL system.
"""

import sys
import unittest


def run_tests():
    """Run all tests for the BIRD-SQL system."""
    # Discover and run tests
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover("tests", pattern="test_*.py")

    # Run tests
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)

    # Return exit code based on test results
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
