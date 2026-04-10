"""Rate limiter module to introduce artificial delays between LLM API calls.

This module provides rate limiting functionality to minimize 429 (Rate Limit)
errors by throttling API requests and respecting rate limits.
"""

import time
import threading
from typing import Optional
from collections import deque


class TokenBucketRateLimiter:
    """Token bucket rate limiter for controlling API request rates.
    
    This implementation uses a token bucket algorithm where tokens are added
    at a fixed rate and each request consumes a token. If no tokens are
    available, the request waits until tokens are available.
    """
    
    def __init__(self, requests_per_second: float = 1.0, burst_size: int = 1):
        """Initialize the rate limiter.
        
        Args:
            requests_per_second: Maximum number of requests per second.
            burst_size: Maximum burst of requests allowed.
        """
        self.requests_per_second = requests_per_second
        self.burst_size = burst_size
        self.tokens = burst_size
        self.last_update = time.time()
        self._lock = threading.Lock()
    
    def _add_tokens(self) -> None:
        """Add tokens to the bucket based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_update
        self.tokens = min(
            self.burst_size,
            self.tokens + elapsed * self.requests_per_second
        )
        self.last_update = now
    
    def wait_if_needed(self) -> float:
        """Wait if rate limiting is needed.
        
        Returns:
            The time waited in seconds.
        """
        with self._lock:
            self._add_tokens()
            
            if self.tokens >= 1:
                self.tokens -= 1
                return 0.0
            
            # Calculate wait time needed
            wait_time = (1 - self.tokens) / self.requests_per_second
            self.tokens = 0
        
        # Wait outside the lock
        time.sleep(wait_time)
        return wait_time
    
    def acquire(self) -> None:
        """Acquire a token, blocking until one is available."""
        self.wait_if_needed()


class FixedDelayRateLimiter:
    """Simple rate limiter that enforces a fixed delay between requests."""
    
    def __init__(self, min_delay_seconds: float = 1.0):
        """Initialize with a minimum delay between requests.
        
        Args:
            min_delay_seconds: Minimum delay in seconds between API calls.
        """
        self.min_delay_seconds = min_delay_seconds
        self._last_request_time: Optional[float] = None
        self._lock = threading.Lock()
    
    def wait_if_needed(self) -> float:
        """Wait if needed to maintain minimum delay between requests.
        
        Returns:
            The time waited in seconds.
        """
        with self._lock:
            if self._last_request_time is None:
                self._last_request_time = time.time()
                return 0.0
            
            elapsed = time.time() - self._last_request_time
            wait_time = max(0, self.min_delay_seconds - elapsed)
            
            if wait_time > 0:
                time.sleep(wait_time)
            
            self._last_request_time = time.time()
            return wait_time
    
    def acquire(self) -> None:
        """Acquire permission to make a request, blocking if necessary."""
        self.wait_if_needed()


class AdaptiveRateLimiter:
    """Adaptive rate limiter that adjusts based on observed rate limits.
    
    This limiter tracks 429 errors and response times to dynamically
    adjust the delay between requests.
    """
    
    def __init__(
        self,
        initial_delay: float = 0.5,
        min_delay: float = 0.1,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        recovery_factor: float = 0.9,
        success_window: int = 10
    ):
        """Initialize the adaptive rate limiter.
        
        Args:
            initial_delay: Starting delay between requests.
            min_delay: Minimum allowed delay.
            max_delay: Maximum allowed delay.
            backoff_factor: Multiplier for delay after 429 error.
            recovery_factor: Multiplier for delay after successful request.
            success_window: Number of successes before reducing delay.
        """
        self.current_delay = initial_delay
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.recovery_factor = recovery_factor
        self.success_window = success_window
        
        self._last_request_time: Optional[float] = None
        self._consecutive_successes = 0
        self._lock = threading.Lock()
    
    def record_success(self) -> None:
        """Record a successful request."""
        with self._lock:
            self._consecutive_successes += 1
            if self._consecutive_successes >= self.success_window:
                self.current_delay = max(
                    self.min_delay,
                    self.current_delay * self.recovery_factor
                )
                self._consecutive_successes = 0
    
    def record_rate_limit_hit(self) -> None:
        """Record a 429 rate limit error."""
        with self._lock:
            self.current_delay = min(
                self.max_delay,
                self.current_delay * self.backoff_factor
            )
            self._consecutive_successes = 0
    
    def wait_if_needed(self) -> float:
        """Wait if needed based on current delay.
        
        Returns:
            The time waited in seconds.
        """
        with self._lock:
            if self._last_request_time is None:
                self._last_request_time = time.time()
                return 0.0
            
            elapsed = time.time() - self._last_request_time
            wait_time = max(0, self.current_delay - elapsed)
            
            if wait_time > 0:
                time.sleep(wait_time)
            
            self._last_request_time = time.time()
            return wait_time
    
    def acquire(self, expect_success: bool = True) -> None:
        """Acquire permission to make a request.
        
        Args:
            expect_success: If True, records success after acquiring.
        """
        self.wait_if_needed()
        if expect_success:
            self.record_success()


class RateLimitManager:
    """Manager for rate limiting with multiple strategies.
    
    This class provides a unified interface for rate limiting with
    configurable strategies.
    """
    
    STRATEGIES = {
        'none': None,
        'fixed': FixedDelayRateLimiter,
        'token_bucket': TokenBucketRateLimiter,
        'adaptive': AdaptiveRateLimiter,
    }
    
    def __init__(self, strategy: str = 'adaptive', **kwargs):
        """Initialize the rate limit manager.
        
        Args:
            strategy: The rate limiting strategy to use ('none', 'fixed', 
                     'token_bucket', 'adaptive').
            **kwargs: Arguments to pass to the rate limiter constructor.
        """
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy}. "
                           f"Available: {list(self.STRATEGIES.keys())}")
        
        if strategy == 'none' or self.STRATEGIES[strategy] is None:
            self._limiter = None
        else:
            self._limiter = self.STRATEGIES[strategy](**kwargs)
    
    def wait_if_needed(self) -> float:
        """Wait if rate limiting is needed.
        
        Returns:
            Time waited in seconds (0.0 if rate limiting is disabled).
        """
        if self._limiter is None:
            return 0.0
        return self._limiter.wait_if_needed()
    
    def acquire(self) -> None:
        """Acquire permission for an API call."""
        self.wait_if_needed()
    
    def record_success(self) -> None:
        """Record a successful API call (for adaptive limiters)."""
        if isinstance(self._limiter, AdaptiveRateLimiter):
            self._limiter.record_success()
    
    def record_rate_limit_hit(self) -> None:
        """Record a rate limit error (for adaptive limiters)."""
        if isinstance(self._limiter, AdaptiveRateLimiter):
            self._limiter.record_rate_limit_hit()


# Global rate limiter instance for the application
_default_limiter: Optional[RateLimitManager] = None


def get_global_limiter() -> Optional[RateLimitManager]:
    """Get the global rate limiter instance."""
    return _default_limiter


def set_global_limiter(limiter: RateLimitManager) -> None:
    """Set the global rate limiter instance."""
    global _default_limiter
    _default_limiter = limiter


def init_global_limiter(
    strategy: str = 'adaptive',
    initial_delay: float = 0.5,
    min_delay: float = 0.1,
    max_delay: float = 60.0,
    min_requests_per_second: float = 0.5
) -> RateLimitManager:
    """Initialize the global rate limiter.
    
    Args:
        strategy: The rate limiting strategy ('none', 'fixed', 'token_bucket', 'adaptive').
        initial_delay: Initial delay between requests (for fixed/adaptive).
        min_delay: Minimum delay between requests.
        max_delay: Maximum delay between requests.
        min_requests_per_second: Minimum requests per second (for token_bucket).
    
    Returns:
        The configured RateLimitManager instance.
    """
    limiter = None
    
    if strategy == 'adaptive':
        limiter = RateLimitManager(
            strategy='adaptive',
            initial_delay=initial_delay,
            min_delay=min_delay,
            max_delay=max_delay
        )
    elif strategy == 'fixed':
        limiter = RateLimitManager(strategy='fixed', min_delay_seconds=initial_delay)
    elif strategy == 'token_bucket':
        limiter = RateLimitManager(
            strategy='token_bucket',
            requests_per_second=min_requests_per_second,
            burst_size=1
        )
    else:
        limiter = RateLimitManager(strategy='none')
    
    set_global_limiter(limiter)
    return limiter
