"""
Tokenization utilities for SQL generation models.
"""

from typing import Dict, List, Optional, Union

from transformers import AutoTokenizer, PreTrainedTokenizer

from ..config import MAX_INPUT_LENGTH, MAX_OUTPUT_LENGTH, SPECIAL_TOKENS


class SQLTokenizer:
    """Wrapper around Hugging Face tokenizers with SQL-specific functionality."""

    def __init__(
        self,
        model_name_or_path: str,
        max_input_length: int = MAX_INPUT_LENGTH,
        max_output_length: int = MAX_OUTPUT_LENGTH,
        use_fast: bool = True,
    ):
        """
        Initialize the SQL tokenizer.

        Args:
            model_name_or_path: Name or path of the pre-trained model
            max_input_length: Maximum input sequence length
            max_output_length: Maximum output sequence length
            use_fast: Whether to use the fast tokenizer implementation
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, use_fast=use_fast
        )
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

        # Add special tokens if they don't exist
        self._add_special_tokens()

    def _add_special_tokens(self) -> None:
        """Add SQL-specific special tokens to the tokenizer."""
        special_tokens = list(SPECIAL_TOKENS.values())

        # Check if we need to add any tokens
        tokens_to_add = []
        for token in special_tokens:
            if token not in self.tokenizer.get_vocab():
                tokens_to_add.append(token)

        if tokens_to_add:
            special_tokens_dict = {"additional_special_tokens": tokens_to_add}
            self.tokenizer.add_special_tokens(special_tokens_dict)

    def encode_input(
        self,
        question: str,
        schema: str,
        padding: Union[bool, str] = "max_length",
        truncation: bool = True,
        return_tensors: Optional[str] = "pt",
    ) -> Dict:
        """
        Encode input for the model.

        Args:
            question: Natural language question
            schema: Database schema string
            padding: Padding strategy
            truncation: Whether to truncate sequences
            return_tensors: Return format for tensors

        Returns:
            Encoded input
        """
        # Format input
        input_text = f"Question: {question} | Schema: {schema}"

        # Tokenize
        return self.tokenizer(
            input_text,
            padding=padding,
            max_length=self.max_input_length,
            truncation=truncation,
            return_tensors=return_tensors,
        )

    def encode_output(
        self,
        sql_query: str,
        padding: Union[bool, str] = "max_length",
        truncation: bool = True,
        return_tensors: Optional[str] = "pt",
    ) -> Dict:
        """
        Encode output for the model.

        Args:
            sql_query: SQL query
            padding: Padding strategy
            truncation: Whether to truncate sequences
            return_tensors: Return format for tensors

        Returns:
            Encoded output
        """
        # Format output
        output_text = f"SQL: {sql_query}"

        # Tokenize as target
        with self.tokenizer.as_target_tokenizer():
            return self.tokenizer(
                output_text,
                padding=padding,
                max_length=self.max_output_length,
                truncation=truncation,
                return_tensors=return_tensors,
            )

    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ) -> str:
        """
        Decode token IDs to text.

        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens
            clean_up_tokenization_spaces: Whether to clean up tokenization spaces

        Returns:
            Decoded text
        """
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )

    def batch_decode(
        self,
        sequences: List[List[int]],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ) -> List[str]:
        """
        Decode multiple sequences of token IDs to text.

        Args:
            sequences: List of token ID sequences
            skip_special_tokens: Whether to skip special tokens
            clean_up_tokenization_spaces: Whether to clean up tokenization spaces

        Returns:
            List of decoded texts
        """
        return self.tokenizer.batch_decode(
            sequences,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )

    def extract_sql_query(self, text: str) -> str:
        """
        Extract the SQL query from decoded text.

        Args:
            text: Decoded text from the model

        Returns:
            Extracted SQL query
        """
        # Remove the "SQL: " prefix if present
        if text.startswith("SQL:"):
            text = text[4:].strip()

        return text

    def get_tokenizer(self) -> PreTrainedTokenizer:
        """Get the underlying Hugging Face tokenizer."""
        return self.tokenizer

    def save_pretrained(self, save_directory: str) -> None:
        """
        Save the tokenizer to a directory.

        Args:
            save_directory: Directory to save the tokenizer
        """
        self.tokenizer.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        max_input_length: int = MAX_INPUT_LENGTH,
        max_output_length: int = MAX_OUTPUT_LENGTH,
        use_fast: bool = True,
        **kwargs,
    ) -> "SQLTokenizer":
        """
        Load a tokenizer from a pre-trained model.

        Args:
            pretrained_model_name_or_path: Name or path of the pre-trained model
            max_input_length: Maximum input sequence length
            max_output_length: Maximum output sequence length
            use_fast: Whether to use the fast tokenizer implementation
            **kwargs: Additional arguments to pass to the tokenizer

        Returns:
            SQLTokenizer instance
        """
        return cls(
            model_name_or_path=pretrained_model_name_or_path,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
            use_fast=use_fast,
        )
