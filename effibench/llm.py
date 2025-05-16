from functools import cache
from dotenv import load_dotenv

from openai import OpenAI
from effibench.utils import retry

load_dotenv()

THINK_TAG, THINK_END_TAG = "<think>", "</think>"

class LanguageModelClient:
    def __init__(self, *args, **kwargs) -> None:
        self.client = OpenAI(*args, **kwargs)

    @staticmethod
    def is_openai_reasoning_model(model: str) -> bool:
        """Returns True if the model is an OpenAI reasoning model."""
        return (model.startswith("o1") and not model.startswith("o1-preview")) or model.startswith("o3") or model.startswith("o4")
    
    @staticmethod
    def is_think_model(model: str) -> bool:
        """Returns True if the model is a DeepSeek reasoning model."""
        return model.startswith("deepseek-r1") or model.startswith("deepseek-reasoner") or model in ("qwq-32b",) or "reasoning" in model
    
    @retry(max_retries=6, backoff_factor=2, error_types=(Exception,))
    def generate(
        self,
        prompt: str | None = None,
        messages: list[dict] | None = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_completion_tokens: int | None = None,
        reasoning_effort: str | None = "high",
        timeout: int = 1200,
        **kwargs
    ) -> tuple[str, list[dict]]:
        response = None
        try:
            # Handle both prompt and messages formats
            if messages is None:
                if prompt is None:
                    raise ValueError("Either prompt or messages must be provided")
                messages = [{"role": "user", "content": prompt}]
            
            # Process messages for OpenAI reasoning models
            if self.is_openai_reasoning_model(model):
                messages = [
                    {"role": "developer" if msg["role"] == "system" else msg["role"], "content": msg["content"]}
                    for msg in messages
                ]
            
            # Prepare API request parameters
            params = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "stream": False,
                "timeout": timeout,
                **kwargs
            }
            
            # Add optional parameters if provided
            if max_completion_tokens is not None:
                params["max_completion_tokens"] = max_completion_tokens
            if reasoning_effort is not None and self.is_openai_reasoning_model(model):
                params["reasoning_effort"] = reasoning_effort

            # Make API request
            response = self.client.chat.completions.create(**params)

            # Process response
            content = (response.choices[0].message.content or "").strip()
            
            if content == "":
                raise RuntimeError("Empty response")
            
            if self.is_think_model(model):
                content = self._postprocess_deepseek_thinking(content)

            # Create updated messages list with assistant's response
            updated_messages = messages + [{"role": "assistant", "content": content}]
            
            return content, updated_messages
        
        except Exception as e:
            raise RuntimeError(f"[{model}] Failed. Response: {response}. {e}") from e

    def _postprocess_deepseek_thinking(self, content: str) -> str:
        """Extract content following the last thinking block in DeepSeek responses."""
        if not content.startswith(THINK_TAG):
            return content
        
        if not THINK_END_TAG in content:
            raise ValueError(f"Reasoning response is missing {THINK_END_TAG}")
        
        depth, i = 0, 0
        while i < len(content):
            if content.startswith(THINK_TAG, i):
                depth += 1
                i += len(THINK_TAG)
            elif content.startswith(THINK_END_TAG, i):
                depth -= 1
                i += len(THINK_END_TAG)
                if depth == 0:
                    return content[i:].strip()
            else:
                i += 1
        return ""

@cache
def get_lm_client() -> LanguageModelClient:
    """Returns a cached language model client instance."""
    return LanguageModelClient()