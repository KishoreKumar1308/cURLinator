"""Base agent class for all cURLinator agents"""

import logging
from abc import ABC, abstractmethod
from typing import Any

from llama_index.core.llms import LLM
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.gemini import Gemini
from llama_index.llms.openai import OpenAI

from curlinator.config import get_settings
from curlinator.utils.llm_validation import validate_llm_config

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Base class for all agents in cURLinator"""

    def __init__(
        self,
        llm: LLM | None = None,
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

    def _create_default_llm(self) -> LLM | None:
        """
        Create default LLM based on settings.

        Only initializes LLM if a valid API key is available.
        Returns None if no valid API key is found (instead of raising an error).
        This allows the agent to be created but will fail gracefully when trying to execute.
        """
        provider = self.settings.default_llm_provider

        if provider == "openai":
            if not validate_llm_config("openai", self.settings.openai_api_key):
                logger.warning(
                    "⚠️  No valid OpenAI API key found. LLM will not be initialized. "
                    "Set OPENAI_API_KEY environment variable with a real API key to "
                    "enable LLM functionality."
                )
                return None
            try:
                kwargs = {
                    "model": self.settings.default_model_openai,
                    "api_key": self.settings.openai_api_key,
                }
                # Add custom API base if provided
                if self.settings.openai_api_base:
                    kwargs["api_base"] = self.settings.openai_api_base
                llm = OpenAI(**kwargs)
                logger.info(f"✅ Initialized OpenAI LLM: {self.settings.default_model_openai}")
                return llm
            except Exception as e:
                logger.error(f"❌ Failed to initialize OpenAI LLM: {e}")
                return None

        elif provider == "anthropic":
            if not validate_llm_config("anthropic", self.settings.anthropic_api_key):
                logger.warning(
                    "⚠️  No valid Anthropic API key found. LLM will not be initialized. "
                    "Set ANTHROPIC_API_KEY environment variable with a real API key to "
                    "enable LLM functionality."
                )
                return None
            try:
                llm = Anthropic(
                    model=self.settings.default_model_anthropic,
                    api_key=self.settings.anthropic_api_key,
                )
                logger.info(
                    f"✅ Initialized Anthropic LLM: {self.settings.default_model_anthropic}"
                )
                return llm
            except Exception as e:
                logger.error(f"❌ Failed to initialize Anthropic LLM: {e}")
                return None

        elif provider == "gemini":
            if not validate_llm_config("gemini", self.settings.gemini_api_key):
                logger.warning(
                    "⚠️  No valid Gemini API key found. LLM will not be initialized. "
                    "Set GEMINI_API_KEY environment variable with a real API key to "
                    "enable LLM functionality."
                )
                return None
            try:
                llm = Gemini(
                    model=self.settings.default_model_gemini,
                    api_key=self.settings.gemini_api_key,
                )
                logger.info(f"✅ Initialized Gemini LLM: {self.settings.default_model_gemini}")
                return llm
            except Exception as e:
                logger.error(f"❌ Failed to initialize Gemini LLM: {e}")
                return None

        else:
            logger.error(f"❌ Unknown LLM provider: {provider}")
            return None

    @abstractmethod
    async def execute(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the agent's main task"""
        pass

    def _log(self, message: str) -> None:
        """Log message if verbose mode is enabled"""
        if self.verbose:
            print(f"[{self.__class__.__name__}] {message}")
