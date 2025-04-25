# LLM SQL Generator

The LLM SQL Generator module provides functionality to translate natural language queries into SQL statements using local Hugging Face transformer models fine-tuned on text-to-SQL tasks.

## Overview

This module replaces the previous cloud-based LLM API approach with a local Hugging Face model. The implementation focuses on:

1. Using lightweight, performant seq2seq models (T5/BART based) fine-tuned on text-to-SQL tasks
2. Supporting GPU acceleration with NVIDIA hardware
3. Maintaining the same API interface for backward compatibility
4. Optimizing for concurrent use in web backends (FastAPI, Flask)

## Model Selection

The system works with seq2seq models fine-tuned on SQL generation tasks. Recommended models include:

- **Salesforce/codet5-base-sql** (default) - A CodeT5 model fine-tuned on SQL generation tasks
- **tscholak/1rpp-sql-base** - A T5-based model trained on Spider dataset
- **mrm8488/t5-base-finetuned-wikisql** - T5 model for simpler SQL queries

Models are chosen for their balance of performance, size, and accuracy. The system prefers smaller models (< 1GB) for faster response times in production environments.

## Key Components

### HuggingFaceSQLGenerator

The core class that manages model initialization and SQL generation:

```python
generator = HuggingFaceSQLGenerator(
    model_name="Salesforce/codet5-base-sql",
    device="cuda",  # or specific GPU: "cuda:0"
    max_tokens=256,
    temperature=0.3,
    batch_size=1
)

sql = generator.generate_sql(
    question="Find all users who registered in the last month",
    schema_context="Table: users\nColumns: id, name, email, registration_date"
)
```

### Public Functions

#### generate_sql_from_text

A stateless function designed for use in concurrent web backends:

```python
from core.llm_sql_generator import generate_sql_from_text

sql = generate_sql_from_text(
    prompt="What are the top 5 selling products?",
    schema_context="Table: products\nColumns: id, name, price, sales_count",
    model_name="Salesforce/codet5-base-sql",
    device="cuda"
)
```

#### generate_sql_json_response

Returns SQL in a JSON format compatible with the original API:

```python
from core.llm_sql_generator import generate_sql_json_response

response = generate_sql_json_response(
    prompt="Find users who placed more than 3 orders",
    schema_context="Table: users\nColumns: id, name\nTable: orders\nColumns: id, user_id, date"
)

# Returns: {"sql": "SELECT u.name FROM users u JOIN orders o ON u.id = o.user_id GROUP BY u.id HAVING COUNT(*) > 3"}
```

#### generate_sql_from_llm (Backward Compatibility)

Maintains compatibility with the previous API:

```python
from core.llm_sql_generator import generate_sql_from_llm

sql = generate_sql_from_llm(
    question="What's the average order value?",
    schema_context="Table: orders\nColumns: id, total_value, customer_id",
    model_name="Salesforce/codet5-base-sql"
)
```

## GPU Optimization

The implementation includes several optimizations for NVIDIA GPUs:

1. **Automatic GPU detection** - Uses CUDA when available and falls back to CPU
2. **Half-precision inference** (FP16) - For faster GPU execution
3. **Multi-GPU support** with `device_map="auto"` if multiple GPUs are detected
4. **Batch processing** capability for handling multiple queries concurrently

## Scaling and Performance Considerations

- **Memory Usage**: The default models require ~1-2GB of VRAM/RAM
- **Latency**: ~200-500ms per query on GPU, ~1-2s on CPU
- **Concurrent Requests**: Multiple queries can be processed in parallel on multi-core CPUs or GPUs
- **Tokenizer Cache**: The implementation maintains tokenizer efficiency across requests

## Schema Context Handling

The module accepts database schema as a string context which helps the model generate more accurate SQL:

```
Schema: 
Table: users
Columns: id (INT), name (VARCHAR), email (VARCHAR), signup_date (DATE)

Table: orders
Columns: id (INT), user_id (INT), total (DECIMAL), order_date (DATE)

Relationship: orders.user_id -> users.id
```

The system handles schema truncation to fit within model context limits while prioritizing question content.

## Error Handling

The implementation includes robust error handling:

- CUDA out-of-memory detection with fallback options
- Prompt truncation for oversized inputs
- Graceful error messages embedded in SQL comments
- Exception capture to prevent API failures

## Usage in Web Backends

Example FastAPI integration:

```python
from fastapi import FastAPI
from pydantic import BaseModel
from core.llm_sql_generator import generate_sql_json_response

app = FastAPI()

class SQLRequest(BaseModel):
    question: str
    schema: str = None

@app.post("/generate-sql")
async def generate_sql(request: SQLRequest):
    return generate_sql_json_response(
        prompt=request.question,
        schema_context=request.schema
    )
```

## Testing

The module includes comprehensive unit tests that can be run without requiring actual model downloads, using mock objects to simulate model behavior.
