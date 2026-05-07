"""Tool search integration — parameter inference, iterative re-search, and outcome logging.

This module connects the semantic tool search engine to Coding Guy's agent loop:

  1. Parameter Inference: Uses the LLM to extract tool parameters from task
     context when selecting a tool. Given a task like "edit line 42 of main.py",
     it can infer patch_file(path='main.py', patches=[...]).

  2. Iterative Re-search: If a selected tool fails, automatically triggers
     re-search with an adjusted query based on the error feedback.

  3. Outcome Logging: Logs tool invocation outcomes (success/failure) to
     continuously refine tool relevance scores.

  4. Tool Selection for Task: High-level API that combines search + parameter
     inference into a single call for the agent loop.
"""

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Outcome logging — records tool success/failure for relevance refinement
# ---------------------------------------------------------------------------

_OUTCOMES_DB_PATH = os.path.join(
    os.path.expanduser("~"), ".cache", "coding-guy", "tool_outcomes.json"
)


class ToolOutcomeLogger:
    """Logs and aggregates tool invocation outcomes for relevance refinement.

    Tracks per-tool success rates and common failure modes. This data can
    be used to boost or demote tools in search rankings.
    """

    def __init__(self, db_path: str = _OUTCOMES_DB_PATH):
        self.db_path = db_path
        self._data: dict = self._load()

    def _load(self) -> dict:
        """Load outcome data from disk."""
        try:
            if os.path.exists(self.db_path):
                with open(self.db_path) as f:
                    return json.load(f)
        except Exception:
            pass
        return {"tools": {}, "last_updated": 0}

    def _save(self):
        """Save outcome data to disk."""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            self._data["last_updated"] = time.time()
            with open(self.db_path, "w") as f:
                json.dump(self._data, f, indent=2)
        except Exception as e:
            print(f"[ToolSearch] Warning: failed to save outcome data: {e}", file=sys.stderr)

    def record(self, tool_name: str, success: bool, error_type: str = "",
               query: str = "", latency_ms: float = 0):
        """Record a tool invocation outcome."""
        if tool_name not in self._data["tools"]:
            self._data["tools"][tool_name] = {
                "successes": 0,
                "failures": 0,
                "error_types": {},
                "query_associations": {},
            }

        tool_data = self._data["tools"][tool_name]
        if success:
            tool_data["successes"] += 1
        else:
            tool_data["failures"] += 1
            if error_type:
                counts = tool_data.get("error_types", {})
                counts[error_type] = counts.get(error_type, 0) + 1
                tool_data["error_types"] = counts

        # Track which queries led to this tool being used
        if query:
            assoc = tool_data.get("query_associations", {})
            # Store a truncated hash of the query to avoid unbounded growth
            q_key = query[:50]
            entry = assoc.get(q_key, {"successes": 0, "failures": 0})
            if success:
                entry["successes"] += 1
            else:
                entry["failures"] += 1
            assoc[q_key] = entry
            tool_data["query_associations"] = assoc

        # Periodic save (every 10 invocations)
        total = sum(t.get("successes", 0) + t.get("failures", 0) for t in self._data["tools"].values())
        if total % 10 == 0:
            self._save()

    def get_success_rate(self, tool_name: str) -> float:
        """Get the success rate (0.0-1.0) for a tool. Returns 0.5 if no data."""
        tool_data = self._data["tools"].get(tool_name, {})
        successes = tool_data.get("successes", 0)
        failures = tool_data.get("failures", 0)
        total = successes + failures
        if total == 0:
            return 0.5  # Neutral — no data
        return successes / total

    def get_relevance_boost(self, tool_name: str) -> float:
        """Get a relevance boost/demotion factor based on historical performance.

        Returns a multiplier:
          - >1.0 for tools with high success rates (boost)
          - <1.0 for tools with low success rates (demotion)
          - 1.0 for tools with no data or neutral performance
        """
        rate = self.get_success_rate(tool_name)
        total = sum(
            t.get("successes", 0) + t.get("failures", 0)
            for t in self._data["tools"].values()
        )
        tool_data = self._data["tools"].get(tool_name, {})
        tool_total = tool_data.get("successes", 0) + tool_data.get("failures", 0)

        if tool_total < 3:
            return 1.0  # Not enough data

        # Scale boost based on deviation from 0.5 and sample size confidence
        deviation = rate - 0.5
        # Confidence factor: more samples → more weight, but capped
        confidence = min(1.0, tool_total / 20)
        boost = 1.0 + deviation * confidence * 0.4

        # Clamp to [0.5, 1.5]
        return max(0.5, min(1.5, boost))

    def get_common_errors(self, tool_name: str, top_n: int = 3) -> list[tuple[str, int]]:
        """Get the most common error types for a tool."""
        tool_data = self._data["tools"].get(tool_name, {})
        errors = tool_data.get("error_types", {})
        sorted_errors = sorted(errors.items(), key=lambda x: x[1], reverse=True)
        return sorted_errors[:top_n]

    def save(self):
        """Force-save outcome data."""
        self._save()


# ---------------------------------------------------------------------------
# Global outcome logger
# ---------------------------------------------------------------------------

_outcome_logger: ToolOutcomeLogger | None = None


def get_outcome_logger() -> ToolOutcomeLogger:
    """Get or create the global outcome logger."""
    global _outcome_logger
    if _outcome_logger is None:
        _outcome_logger = ToolOutcomeLogger()
    return _outcome_logger


# ---------------------------------------------------------------------------
# Iterative re-search on tool failure
# ---------------------------------------------------------------------------

_MAX_RESEARCH_ATTEMPTS = 3


class ToolSearchResult:
    """Result of a tool search with re-search capability.

    Wraps the initial search results and provides an iterative re-search
    mechanism: if the selected tool fails, call re_search() with the error
    to find alternative tools.
    """

    def __init__(self, query: str, results: list[dict], search_engine,
                 outcome_logger: ToolOutcomeLogger | None = None):
        self.query = query
        self.results = results
        self._search_engine = search_engine
        self._outcome_logger = outcome_logger or get_outcome_logger()
        self._attempt = 0
        self._tried_tools: set[str] = set()
        self._error_history: list[tuple[str, str]] = []  # (tool_name, error_type)
        self._current_index = 0

    @property
    def top_result(self) -> dict | None:
        """Get the current top untried result."""
        for r in self.results:
            if r["name"] not in self._tried_tools:
                return r
        return None

    def mark_used(self, tool_name: str, success: bool, error_type: str = ""):
        """Mark a tool as used (success or failure)."""
        self._tried_tools.add(tool_name)
        self._outcome_logger.record(tool_name, success, error_type, self.query)
        if not success:
            self._error_history.append((tool_name, error_type))

    def re_search(self, error_feedback: str = "") -> list[dict]:
        """Re-search with adjusted query based on error feedback.

        If a tool failed, include the error info in the search query to
        find better alternatives. Also demotes previously failed tools.

        Returns new list of results (excluding already-tried tools).
        """
        self._attempt += 1
        if self._attempt > _MAX_RESEARCH_ATTEMPTS:
            from tool_search import log_warn
            log_warn(f"Max re-search attempts ({_MAX_RESEARCH_ATTEMPTS}) reached for: {self.query}")
            return []

        # Build adjusted query
        parts = [self.query]
        if error_feedback:
            parts.append(f"NOT {error_feedback}")

        # Add context about what tools already failed
        for tool_name, err_type in self._error_history:
            parts.append(f"instead of {tool_name}")

        adjusted_query = " ".join(parts)

        from tool_search import log_info
        log_info(f"Re-search (attempt {self._attempt}): {adjusted_query}")

        new_results = self._search_engine.search(
            adjusted_query, top_k=len(self.results) + 5,
            include_descriptions=True
        )

        # Apply outcome-based relevance boosts
        for r in new_results:
            boost = self._outcome_logger.get_relevance_boost(r["name"])
            r["score"] = min(1.0, r["score"] * boost)

        # Filter out already-tried tools
        new_results = [r for r in new_results if r["name"] not in self._tried_tools]

        # Re-sort by adjusted score
        new_results.sort(key=lambda x: x["score"], reverse=True)
        self.results = new_results

        return new_results


# ---------------------------------------------------------------------------
# Parameter inference using LLM
# ---------------------------------------------------------------------------

_PARAMETER_INFERENCE_PROMPT = """\
You are a tool parameter inference assistant. Given a task description and a selected tool,
extract the appropriate parameters for the tool call.

Tool: {tool_name}
Description: {tool_description}
Parameters schema: {parameters_schema}

Task context: {task_context}

Return a JSON object with the tool parameters. Only include parameters that can be
confidently inferred from the task context. Use null for parameters that cannot be inferred.

Example response format:
{{"path": "src/main.py", "pattern": "def hello"}}

Parameters:"""


def infer_tool_parameters(tool_name: str, task_context: str,
                          api_key: str = "", api_url: str = "",
                          model: str = "") -> dict | None:
    """Use the LLM to infer tool parameters from task context.

    This is called when the agent needs help determining what parameters to
    pass to a selected tool. It uses a small LLM call to avoid consuming
    the main conversation's context.

    Args:
        tool_name: The tool that was selected
        task_context: Description of the current task
        api_key: API key for the LLM call (optional, uses env vars if not set)
        api_url: API endpoint URL (optional)
        model: Model to use (optional)

    Returns:
        Dict of inferred parameters, or None if inference failed.
    """
    from tools import TOOL_DEFINITIONS
    from tool_registry import get_tool_entry

    # Find the tool's parameter schema
    tool_def = None
    for tdef in TOOL_DEFINITIONS:
        if tdef["function"]["name"] == tool_name:
            tool_def = tdef
            break

    if not tool_def:
        return None

    entry = get_tool_entry(tool_name)
    description = entry.get("description", "") if entry else tool_def["function"].get("description", "")
    params_schema = json.dumps(tool_def["function"].get("parameters", {}))

    prompt = _PARAMETER_INFERENCE_PROMPT.format(
        tool_name=tool_name,
        tool_description=description,
        parameters_schema=params_schema,
        task_context=task_context[:2000],  # Limit context size
    )

    # Try using ask_ollama for local inference (fast, no API cost)
    try:
        from tools import ask_ollama
        result_str = ask_ollama(prompt=prompt, model=model or "gemma4:e4b")
        result = json.loads(result_str)
        if "response" in result:
            # Try to parse the response as JSON
            response_text = result["response"].strip()
            # Strip markdown code fence if present
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                lines = [l for l in lines if not l.startswith("```")]
                response_text = "\n".join(lines)
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                # Try to find JSON in the response
                import re
                json_match = re.search(r'\{[^{}]+\}', response_text)
                if json_match:
                    return json.loads(json_match.group())
    except Exception as e:
        from tool_search import log_debug
        log_debug(f"Parameter inference via Ollama failed: {e}")

    return None




def infer_tool_parameters_simple(tool_name: str, task_context: str) -> dict | None:
    """Simple rule-based parameter inference without LLM.

    Uses pattern matching on the task context to extract common parameters.
    This is faster and cheaper than LLM-based inference but less capable.
    """
    import re
    from tools import TOOL_DEFINITIONS

    # Find the tool's parameters
    tool_def = None
    for tdef in TOOL_DEFINITIONS:
        if tdef["function"]["name"] == tool_name:
            tool_def = tdef
            break
    if not tool_def:
        return None

    params = tool_def["function"].get("parameters", {})
    properties = params.get("properties", {})
    required = params.get("required", [])
    inferred = {}

    for param_name, param_schema in properties.items():
        param_type = param_schema.get("type", "string")
        param_desc = param_schema.get("description", "").lower()

        # Pattern-based inference for common parameter types
        if param_name == "path" or "path" in param_desc:
            # Try to extract file path from task context
            path_patterns = [
                r'(?:file|path|at|in|of|from)\s+[`"\']?(\S+\.\w+)[`"\']?',
                r'[`"\'](/?\w+(?:/\w+)*\.\w+)[`"\']',
                r'(\w+/\w+\.\w+)',
            ]
            for pattern in path_patterns:
                match = re.search(pattern, task_context)
                if match:
                    inferred[param_name] = match.group(1)
                    break

        elif param_name == "pattern" or "pattern" in param_desc or "regex" in param_desc:
            # Extract search patterns
            match = re.search(r'(?:find|search|grep|look for)\s+["\']?([^"\'\n]+?)["\']?\s', task_context, re.IGNORECASE)
            if match:
                inferred[param_name] = match.group(1).strip()

        elif param_name == "url" or "url" in param_desc:
            match = re.search(r'https?://\S+', task_context)
            if match:
                inferred[param_name] = match.group(0)

        elif param_name == "command" or "command" in param_desc:
            match = re.search(r'(?:run|execute|command):\s*`([^`]+)`', task_context)
            if match:
                inferred[param_name] = match.group(1)

        elif param_name == "question" or "question" in param_desc:
            match = re.search(r'(?:ask|question|wonder|need to know)[:\s]+(.+?)(?:\.|$)', task_context, re.IGNORECASE)
            if match:
                inferred[param_name] = match.group(1).strip()

    return inferred if inferred else None


# ---------------------------------------------------------------------------
# High-level API: select tools for a task
# ---------------------------------------------------------------------------

def select_tools_for_task(task_description: str, top_k: int = 5,
                          search_engine=None) -> list[dict]:
    """Select the most relevant tools for a task description.

    This is the primary API for the agent loop to call. It combines
    semantic search with outcome-based ranking and parameter hints.

    Args:
        task_description: The task to find tools for
        top_k: Number of tools to return
        search_engine: SemanticToolSearch instance (uses global if None)

    Returns:
        List of tool recommendation dicts:
        [{"name": ..., "score": ..., "description": ..., "param_hint": {...}}, ...]
    """
    from tool_search import get_tool_search
    from tool_search import log_info, log_score

    engine = search_engine or get_tool_search()
    if engine is None:
        return []

    # Search for matching tools
    results = engine.search(task_description, top_k=top_k, include_descriptions=True)

    # Apply outcome-based relevance boosts
    logger = get_outcome_logger()
    for r in results:
        boost = logger.get_relevance_boost(r["name"])
        r["score"] = round(min(1.0, r["score"] * boost), 4)

        # Add simple parameter hints
        param_hint = infer_tool_parameters_simple(r["name"], task_description)
        if param_hint:
            r["param_hint"] = param_hint

    # Re-sort after boosting
    results.sort(key=lambda x: x["score"], reverse=True)

    if results:
        log_info(f"Selected {len(results)} tools for task: {task_description[:80]}...")
        for r in results:
            boost = logger.get_relevance_boost(r["name"])
            boost_str = f" (boost={boost:.2f})" if boost != 1.0 else ""
            log_score(f"  {r['name']}: score={r['score']:.3f}{boost_str}")

    return results


# ---------------------------------------------------------------------------
# Architect→Programmer handoff integration
# ---------------------------------------------------------------------------

def architect_handoff(task_description: str, required_capabilities: list[str],
                      search_engine=None) -> dict:
    """Architect→Programmer handoff: find tools matching required capabilities.

    When the architect sub-agent specifies what capabilities are needed for
    a task (e.g. ["file editing", "web scraping", "docker management"]),
    this function finds the best-matching tools and returns a structured
    capability→tool mapping that the programmer sub-agent can use.

    Args:
        task_description: The overall task description
        required_capabilities: List of capability descriptions from the architect
        search_engine: SemanticToolSearch instance (uses global if None)

    Returns:
        Dict with:
          - "task": task description
          - "capabilities": list of capability→tools mappings
          - "all_tools": all unique tools found across capabilities
          - "fallback_used": whether keyword fallback was used
    """
    from tool_search import get_tool_search
    engine = search_engine or get_tool_search()

    if engine is None:
        # No search engine — return static mapping using keyword search
        from tool_registry import get_registry
        registry = get_registry()
        tool_map = []
        all_tools = []
        seen = set()
        for cap in required_capabilities:
            matches = keyword_search(cap, registry, top_k=5)
            tools = [{"name": name, "score": round(score, 3)} for name, score in matches]
            tool_map.append({"capability": cap, "tools": tools})
            for name, score in matches:
                if name not in seen:
                    all_tools.append({"name": name, "score": round(score, 3), "for_capability": cap})
                    seen.add(name)
        return {
            "task": task_description,
            "capabilities": tool_map,
            "all_tools": all_tools,
            "fallback_used": True,
        }

    # Use semantic search with capability-based matching
    results = engine.search_for_capabilities(required_capabilities, top_k=5)

    # Group results by matched capability
    cap_map = {}
    for r in results:
        for cap in r.get("matched_capabilities", []):
            if cap not in cap_map:
                cap_map[cap] = []
            cap_map[cap].append({"name": r["name"], "score": r["score"]})

    capabilities = [
        {"capability": cap, "tools": cap_map.get(cap, [])}
        for cap in required_capabilities
    ]

    return {
        "task": task_description,
        "capabilities": capabilities,
        "all_tools": results,
        "fallback_used": False,
    }


def create_tool_search_result(task_description: str, search_engine=None) -> ToolSearchResult:
    """Create a ToolSearchResult for iterative re-search on failure.

    Args:
        task_description: The task to search tools for
        search_engine: SemanticToolSearch instance (uses global if None)

    Returns:
        ToolSearchResult that can be used for iterative tool selection.
    """
    from tool_search import get_tool_search

    engine = search_engine or get_tool_search()
    if engine is None:
        return ToolSearchResult(task_description, [], None)

    results = engine.search(task_description, top_k=15, include_descriptions=True)
    return ToolSearchResult(task_description, results, engine)
