"""
LLM SQL Generator Module

This module provides functionality to generate SQL queries from natural language
using Hugging Face Transformer models.
"""

import logging
import re
from typing import List, Optional, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class HuggingFaceSQLGenerator:
    """
    A class to generate SQL queries from natural language using Hugging Face models.
    """

    def __init__(
        self,
        model_name: str = "TheBloke/Llama-2-7B-Chat-GGUF",
        device: str = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        use_pipeline: bool = True,
        **kwargs,
    ):
        """
        Initialize the HuggingFaceSQLGenerator.

        Args:
            model_name (str): Name of the Hugging Face model to use.
            device (str, optional): Device to run the model on, e.g. 'cuda:0', 'cpu'.
                                    If None, will use CUDA if available, else CPU.
            max_tokens (int): Maximum tokens for the model's output.
            temperature (float): Temperature for text generation (higher = more random).
            use_pipeline (bool): Whether to use the pipeline API (True) or
                                 load model and tokenizer separately (False).
            **kwargs: Additional arguments to pass to the model or pipeline.
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.use_pipeline = use_pipeline
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Initializing HuggingFaceSQLGenerator with model {model_name}")
        logger.info(f"Using device: {self.device}")

        try:
            if use_pipeline:
                logger.info("Using transformers pipeline API")
                self.model = pipeline(
                    "text-generation",
                    model=model_name,
                    device=self.device,
                    **kwargs,
                )
                self.tokenizer = None  # Not directly accessible in pipeline mode
            else:
                logger.info("Loading model and tokenizer separately")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name, device_map=self.device, **kwargs
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

        # Get model outputs
        try:
            if len(prompt) > 6000:  # A conservative limit
                logger.warning(
                    "Prompt may exceed model's context window. Truncating..."
                )
                # Simple truncation strategy - keep beginning and end of prompt
                head = prompt[:3000]
                tail = prompt[-3000:]
                prompt = f"{head}\n...\n{tail}"

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
                    "CUDA out of memory. Try reducing the prompt size or using CPU."
                )

            # Return a placeholder on error
            error_msg = f"/* Error generating SQL: {str(e)} */"
            return [error_msg] if num_candidates > 1 else error_msg

    def _build_prompt(self, question: str, schema_context: Optional[str] = None) -> str:
        """
        Build a prompt for the LLM.

        Args:
            question (str): Natural language question.
            schema_context (Optional[str]): Database schema or context.

        Returns:
            str: Constructed prompt.
        """
        # Build basic prompt
        prompt_parts = [
            "You are a SQL expert. Generate a SQL query for the following question.",
            "QUESTION: " + question,
        ]

        # Add schema context if provided
        if schema_context:
            prompt_parts.insert(1, "SCHEMA:\n" + schema_context)

        # Add guidance on output format
        prompt_parts.append(
            "Provide only the SQL query without any explanations. Ensure the SQL is valid and optimized."
        )

        return "\n\n".join(prompt_parts)

    def _generate_text(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        num_candidates: int = 1,
        **kwargs,
    ) -> Union[str, List[str]]:
        """
        Generate text using the LLM.

        Args:
            prompt (str): Input prompt.
            max_tokens (int): Maximum tokens to generate.
            temperature (float): Temperature for text generation.
            num_candidates (int): Number of candidates to generate.
            **kwargs: Additional arguments for the generation function.

        Returns:
            Union[str, List[str]]: Generated text or list of generated texts.
        """
        logger.info(f"Generating SQL with prompt of length {len(prompt)}")

        if self.use_pipeline:
            # Handle pipeline API
            generation_config = {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "num_return_sequences": num_candidates,
                "do_sample": temperature > 0,
                **kwargs,
            }

            outputs = self.model(
                prompt,
                **generation_config,
            )

            # Extract generated text from outputs
            if num_candidates > 1:
                return [
                    output["generated_text"][len(prompt) :].strip()
                    for output in outputs
                ]
            else:
                return outputs[0]["generated_text"][len(prompt) :].strip()

        else:
            # Handle direct model + tokenizer API
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            generation_config = {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "num_return_sequences": num_candidates,
                "do_sample": temperature > 0,
                "pad_token_id": self.tokenizer.eos_token_id,  # For models without pad token
                **kwargs,
            }

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **generation_config,
                )

            # Decode outputs
            decoded_outputs = [
                self.tokenizer.decode(output, skip_special_tokens=True)
                for output in outputs
            ]

            # Remove the prompt from each output
            prompt_len = len(prompt)
            processed_outputs = [
                output[prompt_len:].strip() for output in decoded_outputs
            ]

            if num_candidates > 1:
                return processed_outputs
            else:
                return processed_outputs[0]

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
) -> Union[str, List[str]]:
    """
    Generate SQL from a natural language question using a Hugging Face model.

    This is a convenient wrapper around the HuggingFaceSQLGenerator class.

    Args:
        question (str): Natural language question.
        schema_context (Optional[str]): Database schema or context.
        model_name (str): Name of the Hugging Face model to use.
        max_tokens (int): Maximum tokens for generation.
        temperature (float): Temperature for text generation.
        num_candidates (int): Number of SQL candidates to generate.
        clean_output (bool): Whether to clean and extract SQL from model output.
        device (str, optional): Device to run the model on, e.g. 'cuda:0', 'cpu'.
        use_pipeline (bool): Whether to use the pipeline API (True) or
                             load model and tokenizer separately (False).
        **kwargs: Additional arguments to pass to the model initialization or generation.

    Returns:
        Union[str, List[str]]: Generated SQL query or list of candidate queries.
    """
    generator = HuggingFaceSQLGenerator(
        model_name=model_name,
        device=device,
        max_tokens=max_tokens,
        temperature=temperature,
        use_pipeline=use_pipeline,
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

    sql = generate_sql_from_llm(
        question=question,
        schema_context=schema,
        model_name="TheBloke/Llama-2-7B-Chat-GGUF",
        temperature=0.3,
    )

    print(f"Question: {question}")
    print(f"Generated SQL: {sql}")
