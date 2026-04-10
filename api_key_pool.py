"""API Key Pool module for managing multiple LLM API keys.

This module provides a pool of API keys with independent usage tracking
and rate limiting. The pool selects the key with least current usage that
hasn't hit rate limits recently.
"""

import time
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from collections import deque


@dataclass
class UsageRecord:
    """A single API usage record."""
    timestamp: float
    tokens_used: int = 0
    success: bool = True


@dataclass
class RateLimitEvent:
    """A rate limit hit event."""
    timestamp: float
    status_code: int = 429


@dataclass
class APIKey:
    """Represents a single API key with its usage state.
    
    Tracks current usage, rate limit history, and provides scoring
    for key selection.
    """
    key: str
    name: str  # Display name for the key e.g., "key_0", "key_1"
    
    # Usage tracking
    total_requests: int = 0
    total_tokens: int = 0
    current_usage_window: List[UsageRecord] = field(default_factory=list)
    
    # Rate limit tracking
    rate_limit_events: deque = field(default_factory=lambda: deque(maxlen=10))
    consecutive_429s: int = 0
    last_rate_limit_time: Optional[float] = None
    
    # Cooldown state
    in_cooldown: bool = False
    cooldown_until: float = 0.0
    
    # Configuration
    cooldown_duration: float = 60.0  # Seconds to cool down after rate limit
    usage_window_seconds: float = 60.0  # Time window for active usage tracking
    
    _lock: threading.Lock = field(default_factory=threading.Lock)
    
    def __post_init__(self):
        self.current_usage_window = []
        self.rate_limit_events = deque(maxlen=10)
    
    @property
    def is_available(self) -> bool:
        """Check if this key is available for use (not in cooldown)."""
        with self._lock:
            if self.in_cooldown:
                if time.time() >= self.cooldown_until:
                    self.in_cooldown = False
                    self.consecutive_429s = 0
                    return True
                return False
            return True
    
    @property
    def current_usage_score(self) -> float:
        """Calculate current usage score - lower is better.
        
        Based on number of requests in the usage window.
        """
        with self._lock:
            now = time.time()
            # Clean old records
            self.current_usage_window = [
                r for r in self.current_usage_window
                if now - r.timestamp < self.usage_window_seconds
            ]
            return float(len(self.current_usage_window))
    
    @property
    def rate_limit_penalty(self) -> float:
        """Calculate penalty due to recent rate limit hits.
        
        Higher penalty for recent rate limits or multiple consecutive hits.
        """
        with self._lock:
            if not self.rate_limit_events:
                return 0.0
            
            penalty = 0.0
            now = time.time()
            
            # Penalty for recent rate limit events (exponential decay)
            for event in self.rate_limit_events:
                age = now - event.timestamp
                if age < 300:  # Within 5 minutes
                    penalty += 100.0 * (1 - age / 300)
            
            # Additional penalty for consecutive 429s
            penalty += self.consecutive_429s * 50.0
            
            return penalty
    
    @property
    def selection_score(self) -> float:
        """Calculate overall selection score - lower is better.
        
        Combines usage score and rate limit penalty.
        """
        return self.current_usage_score + self.rate_limit_penalty
    
    def record_usage(self, tokens_used: int = 0, success: bool = True) -> None:
        """Record an API call made with this key."""
        with self._lock:
            self.total_requests += 1
            self.total_tokens += tokens_used
            self.current_usage_window.append(UsageRecord(
                timestamp=time.time(),
                tokens_used=tokens_used,
                success=success
            ))
            
            if success:
                self.consecutive_429s = 0
    
    def record_rate_limit_hit(self, status_code: int = 429) -> None:
        """Record a rate limit error for this key."""
        with self._lock:
            self.rate_limit_events.append(RateLimitEvent(
                timestamp=time.time(),
                status_code=status_code
            ))
            self.consecutive_429s += 1
            self.last_rate_limit_time = time.time()
            
            # Enter cooldown if multiple consecutive 429s
            if self.consecutive_429s >= 2:
                self.in_cooldown = True
                # Exponential backoff: 60s, 120s, 240s, etc.
                cooldown = self.cooldown_duration * (2 ** (self.consecutive_429s - 2))
                self.cooldown_until = time.time() + min(cooldown, 600)  # Max 10 min
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for this key."""
        with self._lock:
            return {
                "name": self.name,
                "total_requests": self.total_requests,
                "total_tokens": self.total_tokens,
                "current_usage_window": len(self.current_usage_window),
                "rate_limit_events": len(self.rate_limit_events),
                "consecutive_429s": self.consecutive_429s,
                "in_cooldown": self.in_cooldown,
                "cooldown_until": self.cooldown_until if self.in_cooldown else None,
                "selection_score": self.selection_score,
            }


class APIKeyPoolManager:
    """Manager for a pool of API keys with intelligent selection.
    
    Selects keys based on:
    1. Least current usage (in the usage window)
    2. No recent rate limit hits
    3. Not in cooldown period
    """
    
    def __init__(self, keys: List[str], cooldown_duration: float = 60.0):
        """Initialize the key pool.
        
        Args:
            keys: List of API key strings.
            cooldown_duration: Seconds to cooldown after rate limit hit.
        """
        self.keys: List[APIKey] = []
        self._lock = threading.RLock()
        
        for i, key in enumerate(keys):
            if key and key.strip():  # Skip empty keys
                self.keys.append(APIKey(
                    key=key.strip(),
                    name=f"key_{i}",
                    cooldown_duration=cooldown_duration
                ))
        
        if not self.keys:
            raise ValueError("At least one valid API key is required")
        
        self._round_robin_index = 0
    
    @property
    def available_keys(self) -> List[APIKey]:
        """Get list of keys that are not in cooldown."""
        with self._lock:
            return [k for k in self.keys if k.is_available]
    
    def select_key(self) -> Optional[APIKey]:
        """Select the best key based on usage and rate limit status.
        
        Returns:
            The selected APIKey or None if no keys available.
        """
        with self._lock:
            available = self.available_keys
            
            if not available:
                # All keys in cooldown - reset the one with least penalty
                min_penalty_key = min(self.keys, key=lambda k: k.rate_limit_penalty)
                min_penalty_key.in_cooldown = False
                min_penalty_key.consecutive_429s = 0
                available = [min_penalty_key]
            
            # Select key with lowest selection score
            best_key = min(available, key=lambda k: k.selection_score)
            return best_key
    
    def acquire_key(self) -> Optional[APIKey]:
        """Acquire a key for use (marks it as in-use).
        
        Returns:
            The acquired APIKey or None if none available.
        """
        return self.select_key()
    
    def record_usage(self, key: APIKey, tokens_used: int = 0, success: bool = True) -> None:
        """Record usage for a key."""
        key.record_usage(tokens_used=tokens_used, success=success)
    
    def record_rate_limit_hit(self, key: APIKey, status_code: int = 429) -> None:
        """Record a rate limit hit for a key."""
        key.record_rate_limit_hit(status_code=status_code)
    
    def get_all_stats(self) -> List[Dict[str, Any]]:
        """Get statistics for all keys."""
        with self._lock:
            return [key.get_stats() for key in self.keys]
    
    def get_pool_summary(self) -> Dict[str, Any]:
        """Get a summary of the pool state."""
        with self._lock:
            total_requests = sum(k.total_requests for k in self.keys)
            total_tokens = sum(k.total_tokens for k in self.keys)
            available_count = len(self.available_keys)
            in_cooldown_count = len(self.keys) - available_count
            
            return {
                "total_keys": len(self.keys),
                "available_keys": available_count,
                "in_cooldown": in_cooldown_count,
                "total_requests": total_requests,
                "total_tokens": total_tokens,
            }


# Global pool manager instance
_global_pool: Optional[APIKeyPoolManager] = None


def get_global_pool() -> Optional[APIKeyPoolManager]:
    """Get the global API key pool manager."""
    return _global_pool


def set_global_pool(pool: APIKeyPoolManager) -> None:
    """Set the global API key pool manager."""
    global _global_pool
    _global_pool = pool


def init_key_pool(
    keys: List[str],
    cooldown_duration: float = 60.0
) -> APIKeyPoolManager:
    """Initialize the global API key pool.
    
    Args:
        keys: List of API key strings.
        cooldown_duration: Seconds to cooldown after rate limit.
    
    Returns:
        The configured APIKeyPoolManager instance.
    """
    pool = APIKeyPoolManager(keys=keys, cooldown_duration=cooldown_duration)
    set_global_pool(pool)
    return pool


def parse_api_keys_from_env() -> List[str]:
    """Parse API keys from environment variables.
    
    Supports:
    - Single key: NVIDIA_API_KEY
    - Multiple keys: NVIDIA_API_KEYS (comma-separated)
    - Numbered keys: NVIDIA_API_KEY_0, NVIDIA_API_KEY_1, etc.
    
    Returns:
        List of API keys.
    """
    import os
    
    keys = []
    
    # Check for numbered keys first (NVIDIA_API_KEY_0, NVIDIA_API_KEY_1, ...)
    i = 0
    while True:
        key = os.getenv(f"NVIDIA_API_KEY_{i}")
        if not key:
            break
        keys.append(key.strip())
        i += 1
    
    # Check for comma-separated list
    if not keys:
        multi_key = os.getenv("NVIDIA_API_KEYS")
        if multi_key:
            keys = [k.strip() for k in multi_key.split(",") if k.strip()]
    
    # Fall back to single key
    if not keys:
        single_key = os.getenv("NVIDIA_API_KEY")
        if single_key:
            keys = [single_key.strip()]
    
    return keys
