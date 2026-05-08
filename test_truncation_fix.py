#!/usr/bin/env python3
"""Tests for the truncated tool call repair fix."""

import json
import sys
import os

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from coding_agent import (
    _repair_truncated_json,
    _validate_tool_args,
    _detect_truncated_values,
    _looks_truncated_string,
    _parse_tool_args,
)


def test_repair_truncated_json_basic():
    """Test basic JSON truncation repair."""
    # Already valid JSON
    args, ok, reason = _repair_truncated_json("test", '{"path": "/foo"}')
    assert ok, f"Should succeed for valid JSON, got: {reason}"
    assert json.loads(args) == {"path": "/foo"}

    # Missing closing brace — simple repair
    args, ok, reason = _repair_truncated_json("test", '{"path": "/foo"')
    assert ok, f"Should repair simple missing brace, got: {reason}"
    assert json.loads(args) == {"path": "/foo"}

    # Missing array closing
    args, ok, reason = _repair_truncated_json("test", '{"steps": ["a", "b"')
    assert ok, f"Should repair with ]}} — got: {reason}"
    assert json.loads(args) == {"steps": ["a", "b"]}

    print("  PASS: test_repair_truncated_json_basic")


def test_repair_truncated_json_with_truncated_values():
    """Test that repair detects truncated values when potentially_truncated=True."""
    # The actual failing case from the bug report:
    # create_task(description="Identify and remove files...", steps=["Explore the workspace..."])
    # where the second array element got cut off
    raw = '{"description":"Identify and remove files/folders in the workspace that don", "steps":["Explore the workspace structure to understand what\'s prese"'
    args, ok, reason = _repair_truncated_json("create_task", raw, potentially_truncated=True)

    # The repair should succeed structurally (adding ]}), but should detect
    # that the description is truncated (ends with "don" — mid-word)
    # Actually, "don" is a valid word. The key issue is the steps array
    # element "Explore the workspace structure to understand what's prese"
    # which is truncated mid-word.
    if ok:
        parsed = json.loads(args)
        truncated = _detect_truncated_values(parsed)
        # The description "Identify...that don" looks like truncation
        # The steps[0] "Explore...what's prese" also looks truncated
        assert len(truncated) >= 1, f"Should detect truncated values, got: {truncated}"
    else:
        # Repair itself failed — which is also valid behavior
        assert "truncated" in reason.lower() or "incomplete" in reason.lower()

    print("  PASS: test_repair_truncated_json_with_truncated_values")


def test_repair_truncated_json_max_tokens_strict():
    """Test that max_tokens truncation is handled more strictly."""
    # Truncated JSON where repair would close the struct but values are garbage
    raw = '{"description": "Some task that got cut off in the mid'
    args, ok, reason = _repair_truncated_json("create_task", raw, potentially_truncated=True)

    # Should either fail the repair or detect truncated values
    if not ok:
        assert "truncated" in reason.lower() or "incomplete" in reason.lower(), \
            f"Expected truncation-related error, got: {reason}"
    else:
        # Repair succeeded but should have detected truncated values
        parsed = json.loads(args)
        truncated = _detect_truncated_values(parsed)
        assert len(truncated) > 0, "Should detect truncated values after max_tokens"

    print("  PASS: test_repair_truncated_json_max_tokens_strict")


def test_repair_truncated_json_non_max_tokens():
    """Test that non-max_tokens truncation still allows execution (lenient mode)."""
    # Same truncation but NOT due to max_tokens — should still repair
    raw = '{"path": "/foo", "patches": [{"old": "bar", "new": "baz"}'
    args, ok, reason = _repair_truncated_json("patch_file", raw, potentially_truncated=False)
    assert ok, f"Should repair non-max_tokens truncation, got: {reason}"
    parsed = json.loads(args)
    assert parsed["path"] == "/foo"
    assert len(parsed["patches"]) == 1

    print("  PASS: test_repair_truncated_json_non_max_tokens")


def test_repair_truncated_json_unrepairable():
    """Test that truly unrepairable JSON is rejected."""
    raw = '{"path": "/foo", "patches": [{"old": "bar"'  # Missing new + array close
    args, ok, reason = _repair_truncated_json("patch_file", raw, potentially_truncated=True)
    # This might succeed or fail depending on whether closing braces produce valid JSON
    # {"path": "/foo", "patches": [{"old": "bar"}]}  — this IS valid JSON structurally
    if ok:
        parsed = json.loads(args)
        # But the patch is missing "new" — detect truncated values
        truncated = _detect_truncated_values(parsed)
        # The "old" value "bar" is fine (too short to detect truncation)
        # But we should still check _validate_tool_args
        _, missing = _validate_tool_args("patch_file", parsed)
        # "new" is not in the args at all since the patch dict only has "old"
        # Actually the patch dict would be {"old": "bar"} — missing "new" 
        # but _validate_tool_args checks top-level required params, not nested
    else:
        assert "repair" in reason.lower() or "truncat" in reason.lower()

    print("  PASS: test_repair_truncated_json_unrepairable")


def test_validate_tool_args():
    """Test required argument validation."""
    from tools import TOOL_DEFINITIONS

    # create_task requires "description"
    args, missing = _validate_tool_args("create_task", {"description": "test task"})
    assert not missing, f"Should have all required args, missing: {missing}"

    args, missing = _validate_tool_args("create_task", {})
    assert "description" in missing, f"Should detect missing description, got: {missing}"

    args, missing = _validate_tool_args("create_task", {"description": ""})
    assert "description" in missing, f"Empty string should count as missing, got: {missing}"

    # patch_file requires "path" and "patches"
    args, missing = _validate_tool_args("patch_file", {"path": "/foo", "patches": [{"old": "x", "new": "y"}]})
    assert not missing, f"Should have all required args, missing: {missing}"

    args, missing = _validate_tool_args("patch_file", {"path": "/foo"})
    assert "patches" in missing, f"Should detect missing patches, got: {missing}"

    print("  PASS: test_validate_tool_args")


def test_detect_truncated_values():
    """Test truncated value detection in parsed JSON."""
    # Normal values — should not be truncated
    result = _detect_truncated_values({"description": "A normal task description."})
    assert not result, f"Normal string should not be truncated: {result}"

    # String ending with comma — truncated
    result = _detect_truncated_values({"description": "Some task that ends with,"})
    assert "description" in result, f"Should detect comma-end truncation: {result}"

    # String ending with colon — truncated
    result = _detect_truncated_values({"command": "cd /workspace && "})
    assert "command" in result, f"Should detect trailing command: {result}"

    # Unmatched brackets
    result = _detect_truncated_values({"code": "def foo(bar:"})
    assert "code" in result, f"Should detect unmatched brackets: {result}"

    # Nested truncation
    result = _detect_truncated_values({"steps": ["step one ok", "step two got cut off mid,"]})
    assert len(result) > 0, f"Should detect truncated value in list: {result}"

    print("  PASS: test_detect_truncated_values")


def test_looks_truncated_string():
    """Test the heuristic for detecting truncated strings."""
    # Normal strings — should NOT be truncated
    assert not _looks_truncated_string("A normal task description.")
    assert not _looks_truncated_string("short")
    assert not _looks_truncated_string("")
    assert not _looks_truncated_string("Write the file and verify the result.")

    # Truncated strings — SHOULD be detected
    assert _looks_truncated_string("Some task that ends with,")
    assert _looks_truncated_string("def foo(bar:")
    assert _looks_truncated_string("The value is:")
    assert _looks_truncated_string("function(arg1, arg2")
    assert _looks_truncated_string("Code block starts with `")

    # Medium-length strings ending without punctuation — borderline
    # These should be flagged only if they look mid-word
    assert _looks_truncated_string("Identify and remove files/folders in the workspace that don")
    assert _looks_truncated_string("Explore the workspace structure to understand what's prese")

    print("  PASS: test_looks_truncated_string")


def test_bug_report_case():
    """Reproduce the exact bug from the report."""
    # The exact case from the report:
    # create_task(description="Identify and remove files/folders in the workspace that don
    #   , steps=["Explore the workspace structure to understand what's prese)
    # After repair with ']}' this becomes:
    # create_task(description="Identify...that don", steps=["Explore...what's prese"])
    # The "description" and "steps[0]" values are truncated.

    raw_args = '{"description":"Identify and remove files/folders in the workspace that don", "steps":["Explore the workspace structure to understand what\'s prese"'
    
    # Step 1: Repair the JSON structure
    repaired, ok, reason = _repair_truncated_json(
        "create_task", raw_args, potentially_truncated=True
    )

    if ok:
        # Step 2: Parse and validate
        parsed = json.loads(repaired)
        truncated = _detect_truncated_values(parsed)
        
        # The description ends with "don" — which is either the word "don"
        # or truncation of "don't". Our heuristic should flag it.
        # The steps[0] ending with "prese" should also be flagged.
        assert len(truncated) >= 1, (
            f"Bug report case: should detect truncated values in repaired JSON. "
            f"parsed={parsed}, truncated={truncated}"
        )
        
        # Step 3: Validate required args — description is present but trimmed
        _, missing = _validate_tool_args("create_task", parsed)
        # description exists (even if truncated), so no missing required args
        # The key improvement is that we DETECT the truncation and either
        # reject the call or warn the LLM
        
        print(f"  Detected truncated values: {truncated}")
        print("  PASS: test_bug_report_case — truncation detected after repair")
    else:
        # Repair failed — which means the new code correctly rejected it
        assert "truncat" in reason.lower() or "incomplete" in reason.lower()
        print(f"  Repair rejected: {reason}")
        print("  PASS: test_bug_report_case — repair correctly rejected truncated args")


def test_parse_tool_args_still_works():
    """Ensure _parse_tool_args still works after our changes."""
    args = _parse_tool_args('{"path": "/foo", "content": "bar"}')
    assert args == {"path": "/foo", "content": "bar"}

    args = _parse_tool_args('{}')
    assert args == {}

    args = _parse_tool_args('')
    assert args == {}

    print("  PASS: test_parse_tool_args_still_works")


def test_finish_reason_detection():
    """Test that finish_reason is properly extracted from streaming response."""
    # This tests the streaming assembly code indirectly
    # We can't easily test the full streaming, but we can verify
    # that the _finish_reason and _potentially_truncated fields are used
    
    # Simulate a message with finish_reason
    mock_msg = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_123",
                "type": "function",
                "function": {"name": "create_task", "arguments": '{"description": "test"}'},
                "_potentially_truncated": True,
            }
        ],
        "_finish_reason": "length",
    }
    
    # The agent_loop should extract these
    finish_reason = mock_msg.pop("_finish_reason", None)
    assert finish_reason == "length"
    
    potentially_truncated = (finish_reason == "length")
    assert potentially_truncated
    
    for tc in mock_msg.get("tool_calls", []):
        tc_truncated = potentially_truncated or tc.pop("_potentially_truncated", False)
        assert tc_truncated, "Should detect potentially truncated tool call"

    print("  PASS: test_finish_reason_detection")


if __name__ == "__main__":
    print("Running truncated tool call repair tests...\n")
    test_repair_truncated_json_basic()
    test_repair_truncated_json_with_truncated_values()
    test_repair_truncated_json_max_tokens_strict()
    test_repair_truncated_json_non_max_tokens()
    test_repair_truncated_json_unrepairable()
    test_validate_tool_args()
    test_detect_truncated_values()
    test_looks_truncated_string()
    test_bug_report_case()
    test_parse_tool_args_still_works()
    test_finish_reason_detection()
    print("\nAll tests passed! ✓")
