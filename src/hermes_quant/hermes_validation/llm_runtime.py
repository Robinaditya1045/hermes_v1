from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Callable
from typing import Any

import requests

from hermes_quant.hermes.schemas.market_state import MarketState

logger = logging.getLogger(__name__)

TOOLS_SPEC: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_market_state",
            "description": "Return the validated MarketState contract for a given symbol.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Ticker symbol (e.g. AAPL, NVDA, MSFT, GOOGL)",
                    }
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_quant_prediction",
            "description": "Return quant model probability and expected return for a symbol.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Ticker symbol of the asset",
                    }
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_fundamentals",
            "description": "Return fundamental financial metrics for a symbol.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Ticker symbol of the asset",
                    }
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recent_news",
            "description": "Return recent news sentiment and event impact for a symbol.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Ticker symbol of the asset",
                    }
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_market_regime",
            "description": "Return current market regime classification and confidence.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Ticker symbol of the asset",
                    }
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_risk_assessment",
            "description": "Return volatility metrics, downside risk, and max drawdown.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Ticker symbol of the asset",
                    }
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_portfolio",
            "description": "Return current paper portfolio cash, positions, and exposures.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_trade_history",
            "description": "Return prior paper trades, fills, and outcomes.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
]

SYSTEM_PROMPT = """You are Hermes, the AI reasoning agent of Hermes Quant.

Your responsibilities:
1. Interpret the provided MarketState and examine all quantitative and risk signals.
2. When necessary, consult available read-only tools to gather additional evidence.
3. Compare conflicting signals, identify market risks, and reason through a trade thesis.
4. Synthesize your analysis into a final trade decision.

Constraints:
- You DO NOT own final risk limits, sizing formulas, or trade execution.
- Your requested_notional is advisory only.
- Base your thesis and risks ONLY on real evidence. Do not invent unavailable data.

When you are ready to conclude, output ONLY a valid JSON object matching this schema:
{
  "asset": "<string symbol>",
  "action": "BUY" | "SELL" | "HOLD",
  "confidence": <float between 0.0 and 1.0>,
  "thesis": ["<reason 1>", ...],
  "risks": ["<risk 1>", ...],
  "evidence_used": ["<evidence source 1>", ...],
  "requested_notional": <float >= 0.0>
}
Do not enclose the JSON in conversational commentary.
"""


class LLMHermesRuntime:
    """Real LLM runtime using an OpenAI-compatible endpoint (Ollama, OpenRouter, OpenAI, vLLM)."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout_seconds: float = 180.0,
        max_turns: int = 12,
    ) -> None:
        self.base_url = (
            base_url or os.environ.get("LLM_BASE_URL") or "http://127.0.0.1:11434/v1"
        ).rstrip("/")
        self.model = model or os.environ.get("LLM_MODEL") or "llama3.2:1b"
        self.api_key = (
            api_key or os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or "ollama"
        )
        self.timeout_seconds = timeout_seconds or 180.0
        self.max_turns = max_turns
        self.runtime_available = True
        self.token_usage: dict[str, int] | None = None

    def _extract_json(self, text: str) -> dict[str, Any] | None:
        text = text.strip()
        # Look for ```json ... ``` blocks first
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Look for standalone JSON object { ... }
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    def decide(
        self,
        market_state: MarketState,
        call_tool: Callable[[str, dict[str, Any]], Any],
    ) -> dict[str, Any]:
        """Execute reasoning loop with tool calling and return structured HermesDecision dict."""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Evaluate this MarketState for {market_state.asset} "
                    "and decide on a position:\n\n"
                    f"```json\n{market_state.model_dump_json()}\n```\n\n"
                    "Inspect signals or call tools if needed, "
                    "then output your final trade decision JSON."
                ),
            },
        ]

        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        turns = 0
        content = ""
        while turns < self.max_turns:
            turns += 1
            payload = {
                "model": self.model,
                "messages": messages,
                "tools": TOOLS_SPEC,
                "tool_choice": "auto",
                "temperature": 0.0,
            }

            try:
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout_seconds,
                )
                resp.raise_for_status()
                data = resp.json()
            except requests.RequestException as exc:
                logger.error("LLM request failed: %s", exc)
                return {
                    "asset": market_state.asset,
                    "action": "HOLD",
                    "confidence": 0.0,
                    "thesis": [f"LLM request error: {exc}"],
                    "risks": ["Model execution failure"],
                    "evidence_used": ["market_state"],
                    "requested_notional": 0.0,
                }

            # Record token usage
            usage = data.get("usage", {})
            total_prompt_tokens += usage.get("prompt_tokens", 0)
            total_completion_tokens += usage.get("completion_tokens", 0)
            total_tokens += usage.get("total_tokens", 0)
            self.token_usage = {
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_tokens": total_tokens,
            }

            choices = data.get("choices", [])
            if not choices:
                break
            choice = choices[0]
            msg = choice.get("message", {})
            tool_calls = msg.get("tool_calls")

            # Case 1: Model wants to execute tool calls
            if tool_calls:
                messages.append(msg)
                for tc in tool_calls:
                    func = tc.get("function", {})
                    name = func.get("name", "")
                    raw_args = func.get("arguments", "{}")
                    if isinstance(raw_args, str):
                        try:
                            parsed_args = json.loads(raw_args)
                        except json.JSONDecodeError:
                            parsed_args = {"raw": raw_args}
                    else:
                        parsed_args = raw_args

                    tool_call_id = tc.get("id") or f"call_{name}_{turns}"

                    try:
                        res = call_tool(name, parsed_args)
                        if hasattr(res, "model_dump_json"):
                            content_str = res.model_dump_json()
                        elif isinstance(res, (dict, list)):
                            content_str = json.dumps(res)
                        else:
                            content_str = str(res)
                    except Exception as exc:
                        content_str = json.dumps({"error": str(exc)})

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "name": name,
                            "content": content_str,
                        }
                    )
                continue

            # Case 2: Model returned text content with final decision
            content = msg.get("content", "")
            decision_dict = self._extract_json(content)
            if decision_dict is not None:
                return decision_dict

            # Case 3: Invalid formatting received, request repair from model
            messages.append(msg)
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your response did not contain a valid JSON object matching the schema. "
                        "Please provide ONLY the final JSON object with fields: "
                        "asset, action (BUY/SELL/HOLD), confidence, thesis (list), "
                        "risks (list), evidence_used (list), requested_notional (float)."
                    ),
                }
            )
        return {
            "asset": market_state.asset,
            "action": "HOLD",
            "confidence": 0.0,
            "thesis": ["Hermes LLM failed to produce valid structured output within maximum turns"],
            "risks": ["Unstructured output failure"],
            "evidence_used": ["market_state"],
            "requested_notional": 0.0,
            "raw_output": content,
        }
