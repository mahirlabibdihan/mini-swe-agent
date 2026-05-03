import json
import logging
import os
import threading
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal
from minisweagent.utils.log import instance_logger
import litellm
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

# litellm._turn_on_debug()
logger = logging.getLogger("litellm_model")
litellm.model_cost["Qwen/Qwen2.5-7B-Instruct"] = {
    "input_cost_per_token": 0,
    "output_cost_per_token": 0
}
litellm.model_cost["google/gemma-4-E4B-it"] = {
    "input_cost_per_token": 0,
    "output_cost_per_token": 0
}
# litellm._turn_on_debug()

class LitellmModelConfig(BaseModel):
    model_name: str
    model_kwargs: dict[str, Any] = {}
    litellm_model_registry: Path | str | None = os.getenv("LITELLM_MODEL_REGISTRY_PATH")
    set_cache_control: Literal["default_end"] | None = None
    """Set explicit cache control markers, for example for Anthropic models"""
    cost_tracking: Literal["default", "ignore_errors"] = os.getenv("MSWEA_COST_TRACKING", "default")
    """Cost tracking mode for this model. Can be "default" or "ignore_errors" (ignore errors/missing cost info)"""


class LitellmModel:
    _api_call_lock = threading.RLock()

    def __init__(self, *, config_class: Callable = LitellmModelConfig, **kwargs):
        self.config = config_class(**kwargs)
        self.cost = 0.0
        self.n_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0 
        if self.config.litellm_model_registry and Path(self.config.litellm_model_registry).is_file():
            litellm.utils.register_model(json.loads(Path(self.config.litellm_model_registry).read_text()))

    def _redact_sensitive(self, data: dict[str, Any]) -> dict[str, Any]:
        redacted = dict(data)
        for key in ("api_key", "authorization", "Authorization", "x-api-key", "X-API-Key"):
            if key in redacted:
                redacted[key] = "***"
        return redacted

    def _extract_response_details(self, exc: Exception) -> dict[str, Any]:
        details: dict[str, Any] = {}
        response = getattr(exc, "response", None)
        if response is None:
            return details

        status_code = getattr(response, "status_code", None)
        if status_code is not None:
            details["status_code"] = status_code

        text = getattr(response, "text", None)
        if isinstance(text, str) and text:
            details["response_text"] = text[:4000]

        try:
            json_body = response.json()
            details["response_json"] = json_body
        except Exception:
            pass

        return details

    def _log_query_exception(self, exc: Exception, *, messages: list[dict[str, Any]], kwargs: dict[str, Any]) -> None:
        request_context = {
            "model_name": self.config.model_name,
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "message_count": len(messages),
            "last_role": messages[-1].get("role") if messages else None,
            "last_content_preview": str(messages[-1].get("content", ""))[:500] if messages else None,
            "kwargs": self._redact_sensitive(kwargs),
            "model_kwargs": self._redact_sensitive(self.config.model_kwargs),
            "traceback": traceback.format_exc(),
        }
        request_context |= self._extract_response_details(exc)
        instance_logger.exception("LiteLLM query failed: %s", json.dumps(request_context, default=str))

    @retry(
        reraise=True,
        stop=stop_after_attempt(int(os.getenv("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", "10"))),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        before_sleep=before_sleep_log(logger, logging.DEBUG),
        retry=retry_if_not_exception_type(
            (
                litellm.exceptions.UnsupportedParamsError,
                litellm.exceptions.NotFoundError,
                litellm.exceptions.PermissionDeniedError,
                litellm.exceptions.ContextWindowExceededError,
                litellm.exceptions.APIError,
                litellm.exceptions.AuthenticationError,
                litellm.exceptions.BadRequestError,
                KeyboardInterrupt,
            )
        ),
    )
    def _query(self, messages: list[dict[str, str]], **kwargs):
        try:
            with self._api_call_lock:
                return litellm.completion(
                    model=self.config.model_name, messages=messages, **(self.config.model_kwargs | kwargs)
                )
        except litellm.exceptions.AuthenticationError as e:
            e.message += " You can permanently set your API key with `mini-extra config set KEY VALUE`."
            self._log_query_exception(e, messages=messages, kwargs=kwargs)
            raise e
        except Exception as e:
            self._log_query_exception(e, messages=messages, kwargs=kwargs)
            raise

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        if self.config.set_cache_control:
            messages = set_cache_control(messages, mode=self.config.set_cache_control)
        response = self._query([{"role": msg["role"], "content": msg["content"]} for msg in messages], **kwargs)
        try:
            cost = litellm.cost_calculator.completion_cost(response, model=self.config.model_name)
            if cost <= 0.0:
                raise ValueError(f"Cost must be > 0.0, got {cost}")
        except Exception as e:
            cost = 0.0
            if self.config.cost_tracking != "ignore_errors":
                msg = (
                    f"Error calculating cost for model {self.config.model_name}: {e}, perhaps it's not registered? "
                    "You can ignore this issue from your config file with cost_tracking: 'ignore_errors' or "
                    "globally with export MSWEA_COST_TRACKING='ignore_errors'. "
                    "Alternatively check the 'Cost tracking' section in the documentation at "
                    "https://klieret.short.gy/mini-local-models. "
                    " Still stuck? Please open a github issue at https://github.com/SWE-agent/mini-swe-agent/issues/new/choose!"
                )
                logger.critical(msg)
                raise RuntimeError(msg) from e
        self.n_calls += 1
        self.cost += cost
        usage = getattr(response, "usage", None)
        if usage is not None:
            if isinstance(usage, dict):
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
            else:
                prompt_tokens = getattr(usage, "prompt_tokens", 0)
                completion_tokens = getattr(usage, "completion_tokens", 0)

            if isinstance(prompt_tokens, int | float):
                self.input_tokens += int(prompt_tokens)
            if isinstance(completion_tokens, int | float):
                self.output_tokens += int(completion_tokens)
        GLOBAL_MODEL_STATS.add(cost)
        return {
            "content": response.choices[0].message.content or "",  # type: ignore
            "extra": {
                "response": response.model_dump(),
            },
        }

    def get_template_vars(self) -> dict[str, Any]:
        return self.config.model_dump() | {
            "n_model_calls": self.n_calls,
            "model_cost": self.cost,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }
