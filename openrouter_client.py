#!/usr/bin/env python3
"""OpenRouter API client for unified access to multiple LLM providers.

This module provides a client for the OpenRouter API, which offers
unified access to models from Anthropic, OpenAI, Google, Meta, and more.
"""

import os
import json
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import sys
import requests

from dotenv import load_dotenv

load_dotenv()

# OpenRouter API configuration
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_POLLING_URL = "https://openrouter.ai/api/v1/generation"

# Default models available through OpenRouter
OPENROUTER_MODELS = {
    "claude-3.5-sonnet": "anthropic/claude-3.5-sonnet",
    "claude-3-opus": "anthropic/claude-3-opus",
    "claude-3-haiku": "anthropic/claude-3-haiku",
    "gpt-4o": "openai/gpt-4o",
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "gemini-pro": "google/gemini-pro",
    "gemini-pro-vision": "google/gemini-pro-vision",
    "llama-3.1-70b": "meta-llama/llama-3.1-70b-instruct",
    "llama-3.1-405b": "meta-llama/llama-3.1-405b-instruct",
    "grok-beta": "x-ai/grok-beta",
    "kimi-k2.5": "moonshotai/kimi-k2.5",
}


def get_openrouter_api_key() -> Optional[str]:
    """Get the OpenRouter API key from environment."""
    return os.getenv("OPENROUTER_API_KEY")


def get_openrouter_model() -> str:
    """Get the configured OpenRouter model."""
    model = os.getenv("OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet")
    # Check if it's a shorthand alias
    return OPENROUTER_MODELS.get(model, model)


@dataclass
class OpenRouterConfig:
    """Configuration for OpenRouter API."""
    api_key: str
    model: str = "anthropic/claude-3.5-sonnet"
    timeout: int = 300
    max_tokens: int = 8192
    temperature: float = 1.0
    top_p: float = 1.0
    
    # Optional parameters
    site_url: Optional[str] = None
    site_name: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> "OpenRouterConfig":
        """Create config from environment variables."""
        api_key = get_openrouter_api_key()
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not set. "
                "Add OPENROUTER_API_KEY to your .env file."
            )
        
        config = cls(
            api_key=api_key,
            model=get_openrouter_model(),
            timeout=int(os.getenv("OPENROUTER_TIMEOUT", "300")),
            max_tokens=int(os.getenv("OPENROUTER_MAX_TOKENS", "8192")),
            temperature=float(os.getenv("OPENROUTER_TEMPERATURE", "1.0")),
            top_p=float(os.getenv("OPENROUTER_TOP_P", "1.0")),
            site_url=os.getenv("OPENROUTER_SITE_URL"),
            site_name=os.getenv("OPENROUTER_SITE_NAME"),
        )
        return config


class OpenRouterClient:
    """Client for OpenRouter API."""
    
    def __init__(self, config: Optional[OpenRouterConfig] = None):
        """Initialize the OpenRouter client.
        
        Args:
            config: OpenRouterConfig instance. If None, loads from env.
        """
        self.config = config or OpenRouterConfig.from_env()
        self.session = requests.Session()
    
    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for OpenRouter API requests."""
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.config.site_url or "https://github.com/PortNumber53/Coding-Guy",
            "X-Title": self.config.site_name or "Coding-Guy",
        }
        return headers
    
    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]] = None,
        stream: bool = True,
        tool_choice: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send a chat completion request to OpenRouter.
        
        Args:
            messages: List of message dicts with 'role' and 'content'.
            tools: Optional list of tool definitions.
            stream: Whether to stream the response.
            tool_choice: Optional tool choice strategy.
            
        Returns:
            Response dict with 'choices' containing messages.
        """
        payload = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "stream": stream,
        }
        
        if tools:
            payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice
            
        headers = self._get_headers()
        
        try:
            response = self.session.post(
                OPENROUTER_API_URL,
                headers=headers,
                json=payload,
                stream=stream,
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            
            if stream:
                return self._process_stream(response)
            else:
                return response.json()["choices"][0]["message"]
                
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                raise ValueError("Invalid OpenRouter API key. Check your OPENROUTER_API_KEY.")
            elif e.response.status_code == 402:
                raise ValueError("Insufficient credits on OpenRouter account.")
            elif e.response.status_code == 429:
                raise RuntimeError("Rate limit exceeded. Please wait before retrying.")
            raise
    
    def _process_stream(self, response) -> Dict[str, Any]:
        """Process a streaming response from OpenRouter."""
        content_parts = []
        tool_calls_by_index = {}
        
        for line in response.iter_lines():
            if not line:
                continue
                
            decoded = line.decode("utf-8")
            if not decoded.startswith("data: "):
                continue
                
            data_str = decoded[len("data: "):].strip()
            if data_str == "[DONE]":
                break
                
            try:
                chunk = json.loads(data_str)
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                
                # Text content
                if delta.get("content"):
                    print(delta["content"], end="", flush=True, file=sys.stderr)
                    content_parts.append(delta["content"])
                
                # Tool call deltas
                for tc_delta in delta.get("tool_calls", []):
                    idx = tc_delta["index"]
                    if idx not in tool_calls_by_index:
                        tool_calls_by_index[idx] = {
                            "id": tc_delta.get("id", ""),
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    tc = tool_calls_by_index[idx]
                    if tc_delta.get("id"):
                        tc["id"] = tc_delta["id"]
                    fn = tc_delta.get("function", {})
                    if fn.get("name"):
                        tc["function"]["name"] = fn["name"]
                    if fn.get("arguments"):
                        tc["function"]["arguments"] += fn["arguments"]
                        
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
        
        content = "".join(content_parts)
        if content:
            print(file=sys.stderr)
        
        message = {"role": "assistant"}
        if content:
            message["content"] = content
        if tool_calls_by_index:
            message["tool_calls"] = [
                tool_calls_by_index[i] for i in sorted(tool_calls_by_index)
            ]
        
        return message
    
    def get_generation_info(self, generation_id: str) -> Optional[Dict]:
        """Get generation info for a completed request.
        
        Args:
            generation_id: The generation ID from a response.
            
        Returns:
            Generation info dict or None.
        """
        headers = self._get_headers()
        try:
            response = self.session.get(
                f"{OPENROUTER_POLLING_URL}?id={generation_id}",
                headers=headers,
                timeout=30,
            )
            response.raise_for_status()
            return response.json().get("data")
        except requests.exceptions.RequestException:
            return None
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """List all available models from OpenRouter.
        
        Returns:
            List of model info dicts.
        """
        headers = self._get_headers()
        try:
            response = self.session.get(
                "https://openrouter.ai/api/v1/models",
                headers=headers,
                timeout=30,
            )
            response.raise_for_status()
            return response.json().get("data", [])
        except requests.exceptions.RequestException as e:
            print(f"Error fetching models: {e}", file=sys.stderr)
            return []


def create_openrouter_client(
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> OpenRouterClient:
    """Create an OpenRouter client from environment or arguments.
    
    Args:
        api_key: Optional API key. If None, uses OPENROUTER_API_KEY env var.
        model: Optional model name. If None, uses OPENROUTER_MODEL env var.
        
    Returns:
        Configured OpenRouterClient instance.
    """
    key = api_key or get_openrouter_api_key()
    if not key:
        raise ValueError(
            "OpenRouter API key required. Set OPENROUTER_API_KEY in your .env file."
        )
    
    config = OpenRouterConfig(
        api_key=key,
        model=model or get_openrouter_model(),
    )
    
    return OpenRouterClient(config)
