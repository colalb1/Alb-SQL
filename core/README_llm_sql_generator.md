# Hugging Face LLM SQL Generator for Alb-SQL

This module provides functionality to generate SQL queries from natural language using Hugging Face Transformer models.

## Overview

The `llm_sql_generator.py` module adds support for generating SQL queries using locally hosted Hugging Face models, providing a more customizable and privacy-focused alternative to remote API-based LLMs.

## Features

- Support for various Hugging Face models (e.g., Llama-2, Mistral, CodeLlama)
- Configurable generation parameters (temperature, max tokens, etc.)
- Multiple candidate generation
- GPU/CPU compatibility
- Automatic SQL extraction from model outputs
- Error handling and graceful fallbacks

## Installation

Ensure you have the required dependencies by running:

```bash
pip install transformers torch
```

These dependencies are already included in the project's `requirements.txt`.

## Usage

### Using within Alb-SQL

The Alb-SQL system has been configured to use the Hugging Face SQL generator automatically. You don't need to make any changes to the code to utilize this functionality - the system will use it in the `_generate_sql_candidates` method.

You can adjust the model and parameters by modifying the appropriate variables in the method.

### Direct Function Usage

For direct usage, you can call the `generate_sql_from_llm` function:

```python
from core.llm_sql_generator import generate_sql_from_llm

sql = generate_sql_from_llm(
    question="What is the average salary of employees?",
    schema_context="Table: employees\nColumns: id (INT), name (VARCHAR), salary (DECIMAL)",
    model_name="TheBloke/Llama-2-7B-Chat-GGUF",
    temperature=0.3,
    max_tokens=512,
    num_candidates=1
)

print(f"Generated SQL: {sql}")
```

### Command-line Interface

An example command-line script is provided in `examples/generate_sql_with_hf.py`:

```bash
# Basic usage
python examples/generate_sql_with_hf.py "What is the average salary of employees?"

# With schema file
python examples/generate_sql_with_hf.py "What is the average salary of employees?" --schema my_schema.json

# With custom model and parameters
python examples/generate_sql_with_hf.py "What is the average salary of employees?" \
    --model HuggingFaceH4/zephyr-7b-beta \
    --temperature 0.5 \
    --max-tokens 1024 \
    --candidates 3
```

## Class Usage

For more control, you can use the `HuggingFaceSQLGenerator` class directly:

```python
from core.llm_sql_generator import HuggingFaceSQLGenerator

# Initialize with custom configuration
generator = HuggingFaceSQLGenerator(
    model_name="TheBloke/Llama-2-7B-Chat-GGUF",
    device="cuda:0",  # Specify GPU device if available
    max_tokens=1024,
    temperature=0.7,
    use_pipeline=True,  # Set to False to use AutoModelForCausalLM + AutoTokenizer directly
)

# Generate SQL
sql = generator.generate_sql(
    question="What is the average salary of employees?",
    schema_context="Table: employees\nColumns: id (INT), name (VARCHAR), salary (DECIMAL)",
    num_candidates=3,
    clean_output=True  # Extract SQL from model output
)

print(f"Generated SQL: {sql}")
```

## API Reference

### generate_sql_from_llm

```python
def generate_sql_from_llm(
    question: str,
    schema_context: Optional[str] = None,
    model_name: str = "TheBloke/Llama-2-7B-Chat-GGUF",
    max_tokens: int = 1024,
    temperature: float = 0.7,
    num_candidates: int = 1,
    clean_output: bool = True,
    device: str = None,
    use_pipeline: bool = True,
    **kwargs,
) -> Union[str, List[str]]
```

Generates SQL from a natural language question using a Hugging Face model.

**Parameters:**
- `question`: Natural language question.
- `schema_context`: Database schema or context (optional).
- `model_name`: Name of the Hugging Face model to use.
- `max_tokens`: Maximum tokens for generation.
- `temperature`: Temperature for text generation (higher = more random).
- `num_candidates`: Number of SQL candidates to generate.
- `clean_output`: Whether to clean and extract SQL from model output.
- `device`: Device to run the model on (e.g., 'cuda:0', 'cpu').
- `use_pipeline`: Whether to use the transformers pipeline API.
- `**kwargs`: Additional arguments to pass to the model.

**Returns:**
- Generated SQL query or list of candidate queries.

### HuggingFaceSQLGenerator

The class that handles SQL generation using Hugging Face models.

**Main Methods:**

- `__init__`: Initialize the generator with model configuration.
- `generate_sql`: Generate SQL from a natural language question.
- `_build_prompt`: Build a prompt for the LLM.
- `_generate_text`: Generate text using the LLM.
- `_extract_sql`: Extract SQL query from model output.

## Recommended Models

The following Hugging Face models are recommended for SQL generation:

- `TheBloke/Llama-2-7B-Chat-GGUF`: Good balance of performance and resource usage
- `TheBloke/CodeLlama-7B-GGUF`: Specialized for code generation
- `HuggingFaceH4/zephyr-7b-beta`: Strong performance on text-to-SQL tasks
- `mistralai/Mistral-7B-Instruct-v0.2`: Strong general purpose model

## Troubleshooting

Common issues:

1. **CUDA out of memory**: Reduce the model size or use `device="cpu"`.
2. **Model not found**: Verify the model name and check your internet connection.
3. **Poor SQL quality**: Try adjusting the temperature (lower for more deterministic outputs), or use a different model.
4. **Slow performance**: Consider using a smaller model or quantized version.
