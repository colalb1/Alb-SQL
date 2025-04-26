"""
LLM SQL Generator Module

This module provides functionality to generate SQL queries from natural language
using Hugging Face Transformer models optimized for sequence-to-sequence tasks.
"""

import logging
import os  # Added for environment variables
import re
from typing import Dict, List, Optional, Union

import torch
from dotenv import load_dotenv  # Added for loading .env file
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class HuggingFaceSQLGenerator:
    """
    A class to generate SQL queries from natural language using Hugging Face seq2seq models.
    Optimized for T5/BART models fine-tuned on text-to-SQL tasks.
    """

    def __init__(
        self,
        model_name: str = "defog/sqlcoder",
        device: str = None,
        max_tokens: int = 256,
        temperature: float = 0.3,
        batch_size: int = 1,
        **kwargs,
    ):
        """
        Initialize the HuggingFaceSQLGenerator with a seq2seq model optimized for SQL generation.

        Args:
            model_name (str): Name of the Hugging Face model to use (seq2seq model fine-tuned on SQL).
                Recommended models: 'Salesforce/codet5-base-sql', 'tscholak/1rpp-sql-base'.
            device (str, optional): Device to run the model on, e.g., 'cuda:0', 'cpu'.
                If None, will use CUDA if available, else CPU.
            max_tokens (int): Maximum tokens for the model's output.
            temperature (float): Temperature for text generation (higher = more random).
            batch_size (int): Batch size for inference. Larger values may improve throughput
                              when processing multiple queries.
            **kwargs: Additional arguments to pass to the model or tokenizer load functions.
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.batch_size = batch_size
        self.hf_token = os.getenv("HF_TOKEN")  # Load token from environment

        # Determine device with more detailed GPU detection
        if device is None:
            if torch.cuda.is_available():
                # Automatically use all available GPUs
                self.device = "cuda"
                logger.info(
                    f"CUDA available with {torch.cuda.device_count()} device(s)"
                )
                logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = "cpu"
                logger.info("CUDA not available, using CPU")
        else:
            self.device = device

        logger.info(f"Initializing HuggingFaceSQLGenerator with model {model_name}")
        logger.info(f"Using device: {self.device}")

        # Set device_map to auto for efficient multi-GPU usage
        device_map = (
            "auto"
            if self.device == "cuda" and torch.cuda.device_count() > 1
            else self.device
        )

        try:
            # Load tokenizer and model
            logger.info("Loading tokenizer...")
            # Pass token if available
            tokenizer_kwargs = (
                {**kwargs, "token": self.hf_token} if self.hf_token else kwargs
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, **tokenizer_kwargs
            )

            logger.info(f"Loading model with device_map={device_map}...")
            # Pass token if available
            model_kwargs = (
                {**kwargs, "token": self.hf_token} if self.hf_token else kwargs
            )
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                device_map=device_map,
                torch_dtype=(
                    torch.float16 if self.device == "cuda" else torch.float32
                ),  # Use fp16 for GPU
                **model_kwargs,
            )

            logger.info("Model and tokenizer initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise RuntimeError(
                f"Failed to initialize HuggingFace model {model_name}: {e}"
            )

    def generate_sql(
        self,
        question: str,
        schema_context: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_candidates: int = 1,
        clean_output: bool = True,
        **kwargs,
    ) -> Union[str, List[str]]:
        """
        Generate SQL from a natural language question.

        Args:
            question (str): Natural language question.
            schema_context (Optional[str]): Database schema or context.
            max_tokens (Optional[int]): Maximum tokens for generation (overrides default).
            temperature (Optional[float]): Temperature for generation (overrides default).
            num_candidates (int): Number of SQL candidates to generate.
            clean_output (bool): Whether to clean and extract SQL from model output.
            **kwargs: Additional arguments to pass to the generation function.

        Returns:
            Union[str, List[str]]: Generated SQL query or list of candidate queries.
        """
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature or self.temperature

        # Construct the prompt
        prompt = self._build_prompt(question, schema_context)

        # Truncate if needed (model specific - T5/BART typically have 512-1024 token limits)
        max_input_length = 512  # Conservative for most models
        if len(prompt.split()) > max_input_length:
            logger.warning(
                f"Prompt may exceed model's context window ({len(prompt.split())} tokens). Truncating..."
            )
            # For seq2seq models, we need to keep most relevant parts
            # This is a simple approach; more sophisticated truncation could be implemented
            prompt = self._truncate_prompt(prompt, max_input_length)

        # Get model outputs
        try:
            outputs = self._generate_text(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                num_candidates=num_candidates,
                **kwargs,
            )

            # Process outputs
            if clean_output:
                if isinstance(outputs, list):
                    return [self._extract_sql(output) for output in outputs]
                else:
                    return self._extract_sql(outputs)
            else:
                return outputs

        except Exception as e:
            logger.error(f"Error generating SQL: {e}")
            if isinstance(e, torch.cuda.OutOfMemoryError):
                logger.error(
                    "CUDA out of memory. Try reducing the prompt size or batch size."
                )

            # Return a placeholder on error
            error_msg = f"/* Error generating SQL: {str(e)} */"
            return [error_msg] if num_candidates > 1 else error_msg

    def _build_prompt(self, question: str, schema_context: Optional[str] = None) -> str:
        """
        Build a prompt for the seq2seq model.

        Args:
            question (str): Natural language question.
            schema_context (Optional[str]): Database schema or context.

        Returns:
            str: Constructed prompt.
        """
        # For seq2seq models fine-tuned on SQL tasks, we need to format
        # according to the model's training data format

        # Default format for many text-to-SQL models: schema followed by question
        if schema_context:
            return f"Schema: {schema_context}\nQuestion: {question}"

        return f"Question: {question}"

    def _truncate_prompt(self, prompt: str, max_length: int) -> str:
        """
        Truncate prompt to fit model's context window.

        Args:
            prompt (str): Input prompt.
            max_length (int): Maximum number of tokens to keep.

        Returns:
            str: Truncated prompt.
        """
        words = prompt.split()

        if len(words) <= max_length:
            return prompt

        # For schema + question format, keep beginning (schema) and end (question)
        if "Question:" in prompt:
            # Split into schema and question parts
            parts = prompt.split("Question:")
            schema_part = parts[0]
            question_part = "Question:" + parts[1]

            schema_words = schema_part.split()
            question_words = question_part.split()

            # If question is already too long, truncate it
            if len(question_words) >= max_length:
                return " ".join(question_words[:max_length])

            # Allocate tokens between schema and question
            schema_tokens = max_length - len(question_words)
            return " ".join(schema_words[:schema_tokens]) + " " + question_part

        # Simple truncation for other formats - keep beginning and end
        beginning = words[: max_length // 2]
        end = words[-(max_length // 2) :]
        return " ".join(beginning + end)

    def _generate_text(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        num_candidates: int = 1,
        **kwargs,
    ) -> Union[str, List[str]]:
        """
        Generate text using the seq2seq model.

        Args:
            prompt (str): Input prompt.
            max_tokens (int): Maximum tokens to generate.
            temperature (float): Temperature for text generation.
            num_candidates (int): Number of candidates to generate.
            **kwargs: Additional arguments for the generation function.

        Returns:
            Union[str, List[str]]: Generated text or list of generated texts.
        """
        logger.info(
            f"Generating SQL with prompt of length {len(prompt.split())} tokens"
        )

        # Tokenize input
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Build generation config
        generation_config = {
            "max_length": (
                self.tokenizer.model_max_length
                if hasattr(self.tokenizer, "model_max_length")
                else 512
            ),
            "max_new_tokens": max_tokens,
            "min_length": 10,  # Avoid empty or too short responses
            "temperature": temperature,
            "num_return_sequences": num_candidates,
            "do_sample": temperature > 0,
            **kwargs,
        }

        # Generate with no_grad for inference
        with torch.no_grad():
            # Apply half-precision for faster inference on GPU if supported
            if self.device == "cuda":
                with torch.cuda.amp.autocast():
                    outputs = self.model.generate(**inputs, **generation_config)
            else:
                outputs = self.model.generate(**inputs, **generation_config)

        # Decode outputs
        decoded_outputs = [
            self.tokenizer.decode(
                output, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            for output in outputs
        ]

        if num_candidates > 1:
            return decoded_outputs
        else:
            return decoded_outputs[0]

    def _extract_sql(self, text: str) -> str:
        """
        Extract SQL query from model output.

        Args:
            text (str): Text containing SQL query.

        Returns:
            str: Extracted SQL query.
        """
        # Look for SQL inside code blocks
        sql_pattern = r"```(?:sql)?(.*?)```"
        matches = re.findall(sql_pattern, text, re.DOTALL)

        if matches:
            return matches[0].strip()

        # If no code blocks, attempt to find SQL statements directly
        sql_keywords = [
            r"SELECT.*?FROM",
            r"INSERT INTO",
            r"UPDATE.*?SET",
            r"DELETE FROM",
            r"CREATE TABLE",
            r"ALTER TABLE",
            r"DROP TABLE",
        ]

        for keyword_pattern in sql_keywords:
            matches = re.findall(keyword_pattern, text, re.IGNORECASE | re.DOTALL)
            if matches:
                # Extract from the first keyword to the end (or to the next non-SQL text)
                start_idx = text.lower().find(matches[0].lower())
                # Handle end boundary - find first non-SQL content after the keyword
                extracted_sql = text[start_idx:]

                # Try to find a sensible end point
                end_markers = ["\n\n", "```", "Note:", "Explanation:"]
                for marker in end_markers:
                    if marker in extracted_sql:
                        extracted_sql = extracted_sql.split(marker)[0]

                return extracted_sql.strip()

        # If still no match, return original text as a fallback
        logger.warning("Could not extract SQL pattern, returning original text")
        return text.strip()


def generate_sql_from_text(
    prompt: str,
    schema_context: Optional[str] = None,
    model_name: str = "Salesforce/codet5-base-sql",
    max_tokens: int = 256,
    temperature: float = 0.3,
    device: str = None,
    **kwargs,
) -> str:
    """
    Generate SQL from a natural language question.

    This stateless function is optimized for use in concurrent environments
    like web backends (FastAPI, Flask). It loads the model if needed and
    performs inference in a thread-safe manner.

    Args:
        prompt (str): Natural language question or prompt.
        schema_context (Optional[str]): Database schema or context to include with the prompt.
        model_name (str): Name of the Hugging Face seq2seq model to use.
            Recommended models: 'Salesforce/codet5-base-sql', 'tscholak/1rpp-sql-base'.
        max_tokens (int): Maximum tokens to generate in the response.
        temperature (float): Temperature for text generation. Lower values produce
                            more deterministic outputs.
        device (str, optional): Device to run the model on. If None, will use CUDA if
                               available, else CPU. Can be 'cuda', 'cpu', or specific GPU
                               like 'cuda:0'.
        **kwargs: Additional arguments to pass to the model initialization or generation.

    Returns:
        str: Generated SQL query as a string.
    """
    generator = HuggingFaceSQLGenerator(
        model_name=model_name,
        device=device,
        max_tokens=max_tokens,
        temperature=temperature,
        **kwargs,
    )

    sql = generator.generate_sql(
        question=prompt,
        schema_context=schema_context,
        max_tokens=max_tokens,
        temperature=temperature,
        **kwargs,
    )

    return sql


def generate_sql_json_response(
    prompt: str,
    schema_context: Optional[str] = None,
    model_name: str = "Salesforce/codet5-base-sql",
    max_tokens: int = 256,
    temperature: float = 0.3,
    device: str = None,
    **kwargs,
) -> Dict[str, str]:
    """
    Generate SQL and return in JSON format compatible with the existing API.

    Args:
        prompt (str): Natural language question or prompt.
        schema_context (Optional[str]): Database schema or context to include with the prompt.
        model_name (str): Name of the Hugging Face seq2seq model to use.
        max_tokens (int): Maximum tokens to generate in the response.
        temperature (float): Temperature for text generation.
        device (str, optional): Device to run the model on.
        **kwargs: Additional arguments to pass to the model initialization or generation.

    Returns:
        Dict[str, str]: JSON-compatible dictionary with the generated SQL query.
    """
    sql = generate_sql_from_text(
        prompt=prompt,
        schema_context=schema_context,
        model_name=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        device=device,
        **kwargs,
    )

    return {"sql": sql}


# For backward compatibility with existing code
def generate_sql_from_llm(
    question: str,
    schema_context: Optional[str] = None,
    model_name: str = "Salesforce/codet5-base-sql",
    max_tokens: int = 256,
    temperature: float = 0.3,
    num_candidates: int = 1,
    clean_output: bool = True,
    device: str = None,
    **kwargs,
) -> Union[str, List[str]]:
    """
    Generate SQL from a natural language question using a Hugging Face seq2seq model.

    This function maintains compatibility with the existing API while using
    the new seq2seq-based implementation underneath.

    Args:
        question (str): Natural language question.
        schema_context (Optional[str]): Database schema or context.
        model_name (str): Name of the Hugging Face model to use.
        max_tokens (int): Maximum tokens for generation.
        temperature (float): Temperature for text generation.
        num_candidates (int): Number of SQL candidates to generate.
        clean_output (bool): Whether to clean and extract SQL from model output.
        device (str, optional): Device to run the model on e.g., 'cuda:0', 'cpu'.
        **kwargs: Additional arguments to pass to the model initialization or generation.

    Returns:
        Union[str, List[str]]: Generated SQL query or list of candidate queries.
    """
    generator = HuggingFaceSQLGenerator(
        model_name=model_name,
        device=device,
        max_tokens=max_tokens,
        temperature=temperature,
        **kwargs,
    )

    return generator.generate_sql(
        question=question,
        schema_context=schema_context,
        max_tokens=max_tokens,
        temperature=temperature,
        num_candidates=num_candidates,
        clean_output=clean_output,
        **kwargs,
    )


# Example usage
if __name__ == "__main__":
    # Example for direct function call
    question = "What is the average price of products in the Electronics category?"
    schema = """
    Table: products
    Columns: id (INT), name (VARCHAR), price (DECIMAL), category_id (INT)

    Table: categories
    Columns: id (INT), name (VARCHAR)

    Relationship: products.category_id -> categories.id
    """

    # Generate SQL using the new function
    sql = generate_sql_from_text(
        prompt=question,
        schema_context=schema,
        model_name="Salesforce/codet5-base-sql",
        temperature=0.3,
    )

    print(f"Question: {question}")
    print(f"Generated SQL: {sql}")
