# Alb-SQL

## Project Structure

```
bird_sql/
├── data/
│   ├── loader.py       # SQLDataset implementation
│   └── schemas.py      # SchemaProcessor
├── model/
│   ├── modeling.py     # BirdSQLModel class
│   └── tokenization.py # Custom tokenizer setup
├── training/
│   ├── trainer.py      # TrainingPipeline
│   └── metrics.py      # EvaluationMetrics
└── utils/
    ├── validation.py   # SQLValidator
    └── database.py     # SQL execution helpers
```

## Usage

### Training a Model

```python
from bird_sql.data.loader import SQLDataset
from bird_sql.model.tokenization import SQLTokenizer
from bird_sql.training.trainer import TrainingPipeline

# Load tokenizer
tokenizer = SQLTokenizer.from_pretrained("t5-base")

# Load datasets
train_dataset = SQLDataset(
    base_path="path/to/train/data",
    split="train",
    tokenizer=tokenizer,
)

dev_dataset = SQLDataset(
    base_path="path/to/dev/data",
    split="dev",
    tokenizer=tokenizer,
)

# Create training pipeline
pipeline = TrainingPipeline(
    model_name_or_path="t5-base",
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    tokenizer=tokenizer,
    output_dir="./checkpoints",
    batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    warmup_ratio=0.03,
    num_epochs=10,
    fp16=True,
)

# Train model
metrics = pipeline.train()
```

### Evaluating a Model

```python
from bird_sql.data.loader import SQLDataset
from bird_sql.model.tokenization import SQLTokenizer
from bird_sql.training.trainer import TrainingPipeline

# Load tokenizer
tokenizer = SQLTokenizer.from_pretrained("./checkpoints/best_model")

# Load dataset
eval_dataset = SQLDataset(
    base_path="path/to/dev/data",
    split="dev",
    tokenizer=tokenizer,
)

# Create training pipeline
pipeline = TrainingPipeline.from_pretrained(
    model_path="./checkpoints/best_model",
    eval_dataset=eval_dataset,
)

# Evaluate model
metrics = pipeline.evaluate()
```

### Generating SQL Queries

```python
from bird_sql.model.modeling import BirdSQLModel
from bird_sql.model.tokenization import SQLTokenizer
from bird_sql.data.schemas import SchemaProcessor

# Load tokenizer and model
tokenizer = SQLTokenizer.from_pretrained("./checkpoints/best_model")
model = BirdSQLModel.from_pretrained("./checkpoints/best_model", tokenizer=tokenizer)

# Load schema processor
schema_processor = SchemaProcessor("path/to/tables.json")

# Get schema for a database
db_id = "database_id"
schema_str = schema_processor.format_schema_for_model(db_id)

# Generate SQL query
question = "What is the average salary of employees in the IT department?"
sql_query = model.generate_sql(question, schema_str)
```

### Command Line Interface

```bash
# Train a model
python -m bird_sql.main train --model t5-base --train-data path/to/train/data --dev-data path/to/dev/data --output-dir ./checkpoints

# Evaluate a model
python -m bird_sql.main evaluate --model ./checkpoints/best_model --data path/to/dev/data
```