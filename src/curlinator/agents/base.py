"""Base agent class for all cURLinator agents"""

from abc import ABC, abstractmethod
from typing import Any, Optional

from llama_index.core.llms import LLM
from llama_index.llms.openai import OpenAI
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.gemini import Gemini


from curlinator.config import get_settings


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
        """Create default LLM based on settings"""
        if self.settings.default_llm_provider == "openai":
            kwargs = {
                "model": self.settings.default_model_openai,
                "api_key": self.settings.openai_api_key,
            }
            # Add custom API base if provided
            if self.settings.openai_api_base:
                kwargs["api_base"] = self.settings.openai_api_base
            return OpenAI(**kwargs)
        elif self.settings.default_llm_provider == "anthropic":
            return Anthropic(
                model=self.settings.default_model_anthropic,
                api_key=self.settings.anthropic_api_key,
            )
        elif self.settings.default_llm_provider == "gemini":
            return Gemini(
                model=self.settings.default_model_gemini,
                api_key=self.settings.gemini_api_key,
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.settings.default_llm_provider}")

    @abstractmethod
    async def execute(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the agent's main task"""
        pass

    def _log(self, message: str) -> None:
        """Log message if verbose mode is enabled"""
        if self.verbose:
            print(f"[{self.__class__.__name__}] {message}")

