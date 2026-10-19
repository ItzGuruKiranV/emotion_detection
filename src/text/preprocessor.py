"""
Text Preprocessing and Sanitation Module.
"""

import re
from src.utils.validation import validate_text_input
from src.utils.logger import get_logger

logger = get_logger("text.preprocessor")


def preprocess_text(raw_text: str) -> str:
    """
    Clean, normalize, and validate input text for NLP sequence classification.
    """
    text = validate_text_input(raw_text)

    # Normalize excessive whitespace
    text = re.sub(r"\s+", " ", text)

    # Standardize quotation marks
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")

    return text.strip()
