"""Converter for Qwen 3 models."""

from .llama_converter import LlamaConverter
from ..models.qwen_model import QwenForCausalLM


class QwenConverter(LlamaConverter):
    """Handle conversion of Qwen 3 models to Core ML."""

    model_cls = QwenForCausalLM
