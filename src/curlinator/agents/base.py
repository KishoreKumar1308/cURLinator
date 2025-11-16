"""Base agent class for all cURLinator agents"""

from abc import ABC, abstractmethod
from typing import Any, Optional
import logging

from llama_index.core.llms import LLM
from llama_index.llms.openai import OpenAI
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.gemini import Gemini

from curlinator.config import get_settings
from curlinator.api.utils.llm_validation import validate_llm_config

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Base class for all agents in cURLinator"""

    def __init__(
        self,
        llm: Optional[LLM] = None,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the base agent.

        Args:
            llm: Language model to use. If None, uses default from settings.
            verbose: Whether to enable verbose logging.
        """
        self.settings = get_settings()
        self.verbose = verbose
        self.llm = llm or self._create_default_llm()

    def _create_default_llm(self) -> LLM:
        """
        Create default LLM based on settings.

        Only initializes LLM if a valid API key is available.
        Raises ValueError if no valid API key is found.
        """
        provider = self.settings.default_llm_provider

        if provider == "openai":
            if not validate_llm_config("openai", self.settings.openai_api_key):
                raise ValueError(
                    "No valid OpenAI API key found. Please set OPENAI_API_KEY environment variable "
                    "with a real API key (not a test placeholder)."
                )
            kwargs = {
                "model": self.settings.default_model_openai,
                "api_key": self.settings.openai_api_key,
            }
            # Add custom API base if provided
            if self.settings.openai_api_base:
                kwargs["api_base"] = self.settings.openai_api_base
            return OpenAI(**kwargs)

        elif provider == "anthropic":
            if not validate_llm_config("anthropic", self.settings.anthropic_api_key):
                raise ValueError(
                    "No valid Anthropic API key found. Please set ANTHROPIC_API_KEY environment variable "
                    "with a real API key (not a test placeholder)."
                )
            return Anthropic(
                model=self.settings.default_model_anthropic,
                api_key=self.settings.anthropic_api_key,
            )

        elif provider == "gemini":
            if not validate_llm_config("gemini", self.settings.gemini_api_key):
                raise ValueError(
                    "No valid Gemini API key found. Please set GEMINI_API_KEY environment variable "
                    "with a real API key (not a test placeholder)."
                )
            return Gemini(
                model=self.settings.default_model_gemini,
                api_key=self.settings.gemini_api_key,
            )

        else:
            raise ValueError(f"Unknown LLM provider: {provider}")

    @abstractmethod
    async def execute(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the agent's main task"""
        pass

    def _log(self, message: str) -> None:
        """Log message if verbose mode is enabled"""
        if self.verbose:
            print(f"[{self.__class__.__name__}] {message}")

