# Alb-SQL Testing Framework

This directory contains the testing framework for the Alb-SQL text-to-SQL system. The framework is designed to support:

- Unit tests for each component
- Integration tests for the entire pipeline
- Regression tests to track performance over time
- Error analysis to help improve the system

## Structure

- `unit/` - Unit tests for individual components
- `integration/` - Integration tests for the full pipeline
- `regression/` - Regression tests to catch performance drops
- `fixtures/` - Test fixtures and mock data
- `utils/` - Utility functions for testing
- `conftest.py` - Pytest configuration and shared fixtures

## Running Tests

```bash
# Run all tests
pytest

# Run unit tests only
pytest tests/unit/

# Run integration tests only
pytest tests/integration/

# Run regression tests only
pytest tests/regression/

# Run specific test file
pytest tests/unit/test_schema_analyzer.py

# Run with coverage report
pytest --cov=.
```

## Adding New Tests

When adding new tests, follow these conventions:

1. Unit tests should be placed in the `unit/` directory
2. Integration tests should be placed in the `integration/` directory
3. Regression tests should be placed in the `regression/` directory
4. Use the appropriate fixtures from `conftest.py`
5. Add detailed docstrings to test functions
