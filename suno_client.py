"""Suno API client for AI music generation.

This module provides a complete client for the Suno API, allowing:
- Generate songs with custom lyrics and style
- Check generation status
- Retrieve song data and metadata
- List user's generated songs
- Delete songs
"""

import json
import os
import time
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

import requests
from dotenv import load_dotenv

load_dotenv()

# API Configuration
DEFAULT_BASE_URL = "https://api.suno.com/api/v1"
SUNO_API_KEY = os.getenv("SUNO_API_KEY")


class SongStatus(Enum):
    """Status of a song generation job."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class SongMetadata:
    """Metadata for a generated song."""
    title: str
    artist: str
    duration_seconds: int
    lyrics_snippet: str
    style: str
    model_version: str
    created_at: datetime
    tags: list = field(default_factory=list)


@dataclass
class SongData:
    """Complete data for a generated song."""
    song_id: str
    status: SongStatus
    metadata: Optional[SongMetadata] = None
    audio_urls: dict = field(default_factory=dict)
    cover_image_url: Optional[str] = None


@dataclass
class GenerationJob:
    """A song generation job."""
    job_id: str
    status: SongStatus
    created_at: datetime
    updated_at: datetime
    song_id: Optional[str] = None
    error_message: Optional[str] = None
    progress_percent: int = 0


class SunoAPIError(Exception):
    """Base exception for Suno API errors."""
    pass


class SunoAuthError(SunoAPIError):
    """Authentication error."""
    pass


class SunoRateLimitError(SunoAPIError):
    """Rate limit exceeded."""
    pass


class SunoClient:
    """Client for the Suno API.
    
    Example usage:
        client = SunoClient()
        
        # Generate a song
        job = client.generate_song(
            lyrics="Rise up, coding mind,\nBuild the future line by line",
            style="Upbeat electronic pop",
            title="Code Dreams"
        )
        
        # Wait for completion and get the song
        song = client.wait_for_completion(job.job_id)
    """
    
    def __init__(self, api_key: Optional[str] = None, base_url: str = DEFAULT_BASE_URL):
        """Initialize the Suno client.
        
        Args:
            api_key: Suno API key. If not provided, reads from SUNO_API_KEY env var.
            base_url: Base URL for the Suno API.
        """
        self.api_key = api_key or SUNO_API_KEY
        self.base_url = base_url.rstrip("/")
        
        if not self.api_key:
            raise SunoAuthError(
                "Suno API key not configured. Set SUNO_API_KEY in .env file."
            )
        
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        })
    
    def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        **kwargs
    ) -> dict:
        """Make an HTTP request to the Suno API.
        
        Args:
            method: HTTP method (GET, POST, DELETE, etc.)
            endpoint: API endpoint (e.g., '/jobs')
            **kwargs: Additional arguments for requests
            
        Returns:
            JSON response as dict
            
        Raises:
            SunoAuthError: If authentication fails
            SunoRateLimitError: If rate limit is exceeded
            SunoAPIError: For other API errors
        """
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = self.session.request(method, url, timeout=60, **kwargs)
            
            if response.status_code == 401:
                raise SunoAuthError("Invalid API key. Check your SUNO_API_KEY configuration.")
            elif response.status_code == 429:
                raise SunoRateLimitError("Rate limit exceeded. Please wait before making more requests.")
            elif response.status_code >= 500:
                raise SunoAPIError(f"Server error {response.status_code}: {response.text}")
            
            response.raise_for_status()
            
            # Handle empty responses
            if not response.text:
                return {}
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            raise SunoAPIError(f"Request failed: {str(e)}")
    
    def generate_song(
        self,
        lyrics: str,
        style: str,
        title: Optional[str] = None,
        model_version: str = "v3",
        audio_quality: str = "high",
        seed: Optional[int] = None,
        make_instrumental: bool = False
    ) -> GenerationJob:
        """Submit a new song generation request.
        
        Args:
            lyrics: The lyrics to sing
            style: Musical style/genre description
            title: Optional song title
            model_version: Model version to use (default: v3)
            audio_quality: Audio quality (standard or high)
            seed: Optional seed for reproducibility
            make_instrumental: Generate instrumental without vocals
            
        Returns:
            GenerationJob with job_id for polling status
        """
        payload = {
            "lyrics": lyrics,
            "style": style,
            "model_version": model_version,
            "audio_quality": audio_quality,
            "make_instrumental": make_instrumental
        }
        
        if title:
            payload["title"] = title
        if seed is not None:
            payload["seed"] = seed
        
        data = self._make_request("POST", "/songs/generate", json=payload)
        
        job = GenerationJob(
            job_id=data.get("job_id"),
            status=SongStatus(data.get("status", "pending")),
            created_at=datetime.fromisoformat(data.get("created_at", datetime.now(timezone.utc).isoformat())),
            updated_at=datetime.fromisoformat(data.get("updated_at", datetime.now(timezone.utc).isoformat())),
        )
        
        return job
    
    def get_job_status(self, job_id: str) -> GenerationJob:
        """Check the status of a generation job.
        
        Args:
            job_id: The job ID from generate_song
            
        Returns:
            GenerationJob with current status
        """
        data = self._make_request("GET", f"/jobs/{job_id}")
        
        job = GenerationJob(
            job_id=data.get("job_id", job_id),
            status=SongStatus(data.get("status", "pending")),
            created_at=datetime.fromisoformat(data.get("created_at")),
            updated_at=datetime.fromisoformat(data.get("updated_at")),
            song_id=data.get("song_id"),
            error_message=data.get("error_message"),
            progress_percent=data.get("progress_percent", 0)
        )
        
        return job
    
    def get_song(self, song_id: str) -> SongData:
        """Retrieve a generated song's data.
        
        Args:
            song_id: The song ID from a completed job
            
        Returns:
            SongData with metadata and URLs
        """
        data = self._make_request("GET", f"/songs/{song_id}")
        
        metadata = None
        if "metadata" in data:
            metadata = SongMetadata(
                title=data["metadata"].get("title", "Untitled"),
                artist=data["metadata"].get("artist", "Unknown"),
                duration_seconds=data["metadata"].get("duration_seconds", 0),
                lyrics_snippet=data["metadata"].get("lyrics_snippet", ""),
                style=data["metadata"].get("style", ""),
                model_version=data["metadata"].get("model_version", "v3"),
                created_at=datetime.fromisoformat(data["metadata"].get("created_at")),
                tags=data["metadata"].get("tags", [])
            )
        
        return SongData(
            song_id=data.get("song_id", song_id),
            status=SongStatus(data.get("status", "pending")),
            metadata=metadata,
            audio_urls=data.get("audio_urls", {}),
            cover_image_url=data.get("cover_image_url")
        )
    
    def list_songs(
        self,
        limit: int = 20,
        offset: int = 0,
        status: Optional[str] = None,
        sort_by: str = "created_at",
        sort_direction: str = "desc"
    ) -> dict:
        """List user's generated songs.
        
        Args:
            limit: Number of results per page (max 100)
            offset: Pagination offset
            status: Filter by status (pending, processing, completed, failed)
            sort_by: Field to sort by (created_at, title)
            sort_direction: Sort direction (asc or desc)
            
        Returns:
            Dict with total count and list of SongData
        """
        params = {
            "limit": min(limit, 100),
            "offset": offset,
            "sort_by": sort_by,
            "sort_direction": sort_direction
        }
        if status:
            params["status"] = status
        
        data = self._make_request("GET", "/songs", params=params)
        
        songs = []
        for song_data in data.get("songs", []):
            songs.append(SongData(
                song_id=song_data.get("song_id"),
                status=SongStatus(song_data.get("status", "pending")),
                audio_urls=song_data.get("audio_urls", {}),
                cover_image_url=song_data.get("cover_image_url")
            ))
        
        return {
            "total": data.get("total", 0),
            "limit": len(songs),
            "offset": offset,
            "songs": songs
        }
    
    def delete_song(self, song_id: str) -> bool:
        """Delete a generated song.
        
        Args:
            song_id: The song ID to delete
            
        Returns:
            True if successful
        """
        self._make_request("DELETE", f"/songs/{song_id}")
        return True
    
    def wait_for_completion(
        self,
        job_id: str,
        timeout: Optional[int] = None,
        poll_interval: int = 5
    ) -> Optional[SongData]:
        """Poll a job until it completes or fails.
        
        Args:
            job_id: The job ID to poll
            timeout: Maximum seconds to wait (None for no timeout)
            poll_interval: Seconds between status checks
            
        Returns:
            SongData if successful, None if timeout
            
        Raises:
            SunoAPIError: If the job fails
        """
        start_time = time.monotonic()
        
        while True:
            job = self.get_job_status(job_id)
            
            if job.status == SongStatus.COMPLETED and job.song_id:
                return self.get_song(job.song_id)
            
            if job.status == SongStatus.FAILED:
                raise SunoAPIError(f"Job failed: {job.error_message or 'Unknown error'}")
            
            if job.status == SongStatus.CANCELLED:
                raise SunoAPIError("Job was cancelled")
            
            if timeout and (time.monotonic() - start_time) > timeout:
                return None
            
            time.sleep(poll_interval)
    
    def generate_songs_batch(
        self,
        variations: list[dict],
        poll_interval: int = 5,
        timeout: int = 600
    ) -> list[SongData]:
        """Generate multiple songs in parallel and wait for completion.
        
        Args:
            variations: List of dicts with keys: lyrics, style, title, etc.
            poll_interval: Seconds between status checks
            timeout: Maximum time to wait for all jobs in seconds (default: 600)
            
        Returns:
            List of SongData for completed songs
            
        Raises:
            TimeoutError: If jobs don't complete within the timeout period
        """
        start_time = time.monotonic()
        # Submit all jobs
        jobs = []
        for params in variations:
            job = self.generate_song(**params)
            jobs.append(job)
        
        # Poll all jobs
        completed = [None] * len(jobs)
        pending = set(range(len(jobs)))
        
        while pending:
            # Check timeout
            elapsed = time.monotonic() - start_time
            if elapsed > timeout:
                # Build error message with status of pending jobs
                job_statuses = []
                for i in pending:
                    job = jobs[i]
                    try:
                        status = self.get_job_status(job.job_id)
                        job_statuses.append(f"Job {job.job_id}: {status.value}")
                    except Exception:
                        job_statuses.append(f"Job {job.job_id}: unknown")
                
                raise TimeoutError(
                    f"Batch song generation timed out after {timeout}s. "
                    f"Pending: {', '.join(job_statuses)}"
                )
            
            time.sleep(poll_interval)
            
            newly_done = set()
            for idx in pending:
                job = self.get_job_status(jobs[idx].job_id)
                
                if job.status == SongStatus.COMPLETED and job.song_id:
                    completed[idx] = self.get_song(job.song_id)
                    newly_done.add(idx)
                elif job.status == SongStatus.FAILED:
                    completed[idx] = SongData(
                        song_id="",
                        status=SongStatus.FAILED,
                        metadata=SongMetadata(
                            title=variations[idx].get("title", "Failed"),
                            artist="Unknown",
                            duration_seconds=0,
                            lyrics_snippet="",
                            style=variations[idx].get("style", ""),
                            model_version="v3",
                            created_at=datetime.now(timezone.utc)
                        )
                    )
                    newly_done.add(idx)
            
            pending -= newly_done
        
        return completed


# Convenience functions for tool integration

def _get_client() -> SunoClient:
    """Get a configured Suno client."""
    return SunoClient()


def suno_generate_song(
    lyrics: str,
    style: str,
    title: Optional[str] = None,
    wait_for_completion: bool = False,
    timeout: int = 300
) -> str:
    """Generate a song using Suno AI.
    
    Args:
        lyrics: The lyrics to sing
        style: Musical style/genre (e.g., "Upbeat electronic pop", "Acoustic folk ballad")
        title: Optional song title
        wait_for_completion: If True, poll until complete and return full song data
        timeout: Maximum seconds to wait for completion (if wait_for_completion=True)
        
    Returns:
        JSON string with job ID or full song data
    """
    try:
        client = _get_client()
        job = client.generate_song(lyrics=lyrics, style=style, title=title)
        
        if wait_for_completion:
            song = client.wait_for_completion(job.job_id, timeout=timeout)
            if song:
                return json.dumps({
                    "job_id": job.job_id,
                    "status": song.status.value,
                    "song_id": song.song_id,
                    "title": song.metadata.title if song.metadata else title,
                    "artist": song.metadata.artist if song.metadata else "Unknown",
                    "duration_seconds": song.metadata.duration_seconds if song.metadata else 0,
                    "audio_urls": song.audio_urls,
                    "cover_image_url": song.cover_image_url,
                    "message": "Song generated successfully"
                })
            else:
                return json.dumps({
                    "job_id": job.job_id,
                    "status": "timeout",
                    "message": "Generation timed out, use suno_get_job_status to check progress"
                })
        else:
            return json.dumps({
                "job_id": job.job_id,
                "status": job.status.value,
                "message": "Song generation started. Use suno_get_job_status to check progress."
            })
    
    except SunoAuthError as e:
        return json.dumps({"error": str(e)})
    except SunoRateLimitError as e:
        return json.dumps({"error": str(e), "retry_after": 60})
    except SunoAPIError as e:
        return json.dumps({"error": f"Suno API error: {str(e)}"})
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {str(e)}"})


def suno_get_job_status(job_id: str) -> str:
    """Check the status of a song generation job.
    
    Args:
        job_id: The job ID from suno_generate_song
        
    Returns:
        JSON string with job status
    """
    try:
        client = _get_client()
        job = client.get_job_status(job_id)
        
        result = {
            "job_id": job.job_id,
            "status": job.status.value,
            "progress_percent": job.progress_percent,
            "created_at": job.created_at.isoformat(),
            "updated_at": job.updated_at.isoformat(),
        }
        
        if job.song_id:
            result["song_id"] = job.song_id
        if job.error_message:
            result["error_message"] = job.error_message
        if job.status == SongStatus.COMPLETED and job.song_id:
            result["message"] = f"Job complete. Use suno_get_song_data('{job.song_id}') to retrieve song details."
        elif job.status == SongStatus.PENDING:
            result["message"] = "Job is pending in queue."
        elif job.status == SongStatus.PROCESSING:
            result["message"] = f"Job is processing ({job.progress_percent}% complete)."
        elif job.status == SongStatus.FAILED:
            result["message"] = f"Job failed: {job.error_message}"
        
        return json.dumps(result)
    
    except SunoAPIError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {str(e)}"})


def suno_get_song_data(song_id: str) -> str:
    """Retrieve complete data for a generated song.
    
    Args:
        song_id: The song ID (from completed job status)
        
    Returns:
        JSON string with song metadata and URLs
    """
    try:
        client = _get_client()
        song = client.get_song(song_id)
        
        result = {
            "song_id": song.song_id,
            "status": song.status.value,
            "audio_urls": song.audio_urls,
            "cover_image_url": song.cover_image_url,
        }
        
        if song.metadata:
            result.update({
                "title": song.metadata.title,
                "artist": song.metadata.artist,
                "duration_seconds": song.metadata.duration_seconds,
                "lyrics_snippet": song.metadata.lyrics_snippet,
                "style": song.metadata.style,
                "model_version": song.metadata.model_version,
                "created_at": song.metadata.created_at.isoformat(),
                "tags": song.metadata.tags,
            })
        
        return json.dumps(result)
    
    except SunoAPIError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {str(e)}"})


def suno_list_songs(
    limit: int = 20,
    offset: int = 0,
    status: Optional[str] = None
) -> str:
    """List user's generated songs.
    
    Args:
        limit: Number of results to return (max 100)
        offset: Pagination offset
        status: Filter by status (pending, processing, completed, failed)
        
    Returns:
        JSON string with song list
    """
    try:
        client = _get_client()
        result = client.list_songs(limit=limit, offset=offset, status=status)
        
        songs = []
        for song in result["songs"]:
            songs.append({
                "song_id": song.song_id,
                "status": song.status.value,
                "audio_urls": song.audio_urls,
                "cover_image_url": song.cover_image_url,
            })
        
        return json.dumps({
            "total": result["total"],
            "limit": result["limit"],
            "offset": result["offset"],
            "songs": songs
        })
    
    except SunoAPIError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {str(e)}"})


def suno_delete_song(song_id: str) -> str:
    """Delete a generated song.
    
    Args:
        song_id: The song ID to delete
        
    Returns:
        JSON string with deletion status
    """
    try:
        client = _get_client()
        client.delete_song(song_id)
        return json.dumps({
            "song_id": song_id,
            "deleted": True,
            "message": "Song deleted successfully"
        })
    
    except SunoAPIError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {str(e)}"})
