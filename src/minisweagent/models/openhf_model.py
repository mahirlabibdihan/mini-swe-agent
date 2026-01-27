import json
import logging
import os
from typing import Any, Literal

import requests
from pydantic import BaseModel
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.models.utils.cache_control import set_cache_control

from minisweagent.models.openrouter_model import OpenRouterModelConfig, OpenRouterAPIError, OpenRouterAuthenticationError, OpenRouterRateLimitError, OpenRouterModel
logger = logging.getLogger("openrouter_model")


class OpenHFContextLengthExceededError(Exception):
    """Custom exception for OpenRouter authentication errors."""
    pass

class OpenHFModel(OpenRouterModel):
    def __init__(self, **kwargs):
        self.config = OpenRouterModelConfig(**kwargs)
        self.cost = 0.0
        self.n_calls = 0
        self._api_url = f"{os.getenv("HUGGING_FACE_API_SERVER", "")}/v1/chat/completions"
        self._api_key = "your-api-key-here"

    @retry(
        reraise=True,
        stop=stop_after_attempt(int(os.getenv("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", "10"))),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=retry_if_not_exception_type(
            (
                OpenRouterAuthenticationError,
                KeyboardInterrupt,
                OpenHFContextLengthExceededError,
            )
        ),
    )
    
    def _query(self, messages: list[dict[str, str]], **kwargs):
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.config.model_name,
            "messages": messages,
            **(self.config.model_kwargs | kwargs),
        }

        try:
            response = requests.post(self._api_url, headers=headers, data=json.dumps(payload), timeout=100)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 401:
                error_msg = "Authentication failed. You can permanently set your API key with `mini-extra config set OPENROUTER_API_KEY YOUR_KEY`."
                raise OpenRouterAuthenticationError(error_msg) from e
            elif response.status_code == 429:
                raise OpenRouterRateLimitError("Rate limit exceeded") from e
            elif response.status_code == 400:
                if "please reduce the length of the input messages" in response.text.lower():
                    raise OpenHFContextLengthExceededError("Context length exceeded") from e
                else:
                    raise OpenRouterAPIError(f"Request failed: {e}") from e
            elif response.status_code == 413:
                raise OpenHFContextLengthExceededError("Context length exceeded") from e
            else:
                raise OpenRouterAPIError(f"HTTP {response.status_code}: {response.text}") from e
        except requests.exceptions.RequestException as e:
            raise OpenRouterAPIError(f"Request failed: {e}") from e

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        response = self._query([{"role": msg["role"], "content": msg["content"]} for msg in messages], **kwargs)

        cost = 0.0

        self.n_calls += 1
        self.cost += cost
        GLOBAL_MODEL_STATS.add(cost)

        return {
            "content": response["choices"][0]["message"]["content"] or "",
            "extra": {
                "response": response,  # already is json
            },
        }