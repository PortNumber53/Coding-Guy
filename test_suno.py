#!/usr/bin/env python3
"""Test script for Suno API integration.

Run this to verify the Suno client module loads correctly:
    python test_suno.py --validate

To test with actual API calls (requires SUNO_API_KEY):
    SUNO_API_KEY=your-key python test_suno.py --test-generate
"""

import sys
import argparse
from suno_client import (
    SunoClient,
    SongStatus,
    SunoAuthError,
    suno_generate_song,
    suno_get_job_status,
    suno_get_song_data,
    suno_list_songs,
    suno_delete_song,
)


def validate_module():
    """Validate the module imports correctly."""
    print("Validating Suno module...")
    
    # Check exports
    assert SunoClient, "SunoClient not exported"
    assert SongStatus, "SongStatus not exported"
    assert SunoAuthError, "SunoAuthError not exported"
    assert suno_generate_song, "suno_generate_song not exported"
    assert suno_get_job_status, "suno_get_job_status not exported"
    assert suno_get_song_data, "suno_get_song_data not exported"
    assert suno_list_songs, "suno_list_songs not exported"
    assert suno_delete_song, "suno_delete_song not exported"
    
    # Check SongStatus enum values
    statuses = [SongStatus.PENDING, SongStatus.PROCESSING, 
                SongStatus.COMPLETED, SongStatus.FAILED, SongStatus.CANCELLED]
    for status in statuses:
        assert isinstance(status.value, str), f"Status {status} has invalid value"
    
    print("✓ Module validation passed!")
    print("\nExported items:")
    print("  - SunoClient")
    print("  - SongStatus (PENDING, PROCESSING, COMPLETED, FAILED, CANCELLED)")
    print("  - SunoAuthError")
    print("  - suno_generate_song")
    print("  - suno_get_job_status")
    print("  - suno_get_song_data")
    print("  - suno_list_songs")
    print("  - suno_delete_song")
    return True


def test_mock_client():
    """Test client initialization with no API key."""
    print("\nTesting mock client initialization (no API key)...")
    try:
        client = SunoClient(api_key=None)
        print("✗ Should have raised SunoAuthError")
        return False
    except SunoAuthError as e:
        print(f"✓ Correctly raised SunoAuthError: {e}")
        return True


def test_api_calls():
    """Test actual API calls (requires SUNO_API_KEY)."""
    print("\nTesting actual API calls...")
    print("This requires SUNO_API_KEY to be set in environment\n")
    
    # Test generate_song
    print("1. Testing suno_generate_song...")
    result = suno_generate_song(
        lyrics="Code all night, code all day\nBuild the future, find your way",
        style="Electronic pop with energetic beat",
        title="Coder's Anthem",
        wait_for_completion=False
    )
    print(f"   Result: {result[:200]}...")
    
    # Test list_songs
    print("\n2. Testing suno_list_songs...")
    result = suno_list_songs(limit=5)
    print(f"   Result: {result[:200]}...")
    
    print("\n✓ API calls completed")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Test Suno API integration"
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Validate module imports and structure"
    )
    parser.add_argument(
        "--test-generate", action="store_true",
        help="Test actual API calls (requires SUNO_API_KEY)"
    )
    args = parser.parse_args()
    
    if not args.validate and not args.test_generate:
        # Run validation by default
        args.validate = True
    
    success = True
    
    if args.validate:
        try:
            success = success and validate_module()
            success = success and test_mock_client()
        except Exception as e:
            print(f"✗ Validation failed: {e}")
            success = False
    
    if args.test_generate:
        try:
            success = success and test_api_calls()
        except Exception as e:
            print(f"✗ API test failed: {e}")
            success = False
    
    if success:
        print("\n=== All tests passed! ===")
        return 0
    else:
        print("\n=== Tests failed ===")
        return 1


if __name__ == "__main__":
    sys.exit(main())
