"""
Async Task Manager for batch ingest operations.

Provides a simple in-memory task queue that:
- Accepts batch ingest requests and returns a task_id immediately
- Processes files in the background
- Allows polling for task status/completion
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class IngestTask:
    """Represents a batch ingest task."""
    task_id: str
    client_id: str
    job_id: str
    status: TaskStatus = TaskStatus.QUEUED
    total_files: int = 0
    processed_files: int = 0
    failed_files: int = 0
    skipped_files: int = 0
    created_at: str = ""
    completed_at: Optional[str] = None
    error: Optional[str] = None
    file_paths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "client_id": self.client_id,
            "job_id": self.job_id,
            "status": self.status.value,
            "total_files": self.total_files,
            "processed_files": self.processed_files,
            "failed_files": self.failed_files,
            "skipped_files": self.skipped_files,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "error": self.error,
        }


class TaskManager:
    """In-memory task manager for async batch processing."""

    def __init__(self):
        self.tasks: dict[str, IngestTask] = {}

    def create_task(self, client_id: str, job_id: str, file_paths: list[str]) -> IngestTask:
        """Create a new ingest task and return it."""
        task = IngestTask(
            task_id=str(uuid.uuid4()),
            client_id=client_id,
            job_id=job_id,
            total_files=len(file_paths),
            created_at=datetime.now().isoformat(),
            file_paths=file_paths,
        )
        self.tasks[task.task_id] = task
        logger.info(f"Task created: {task.task_id} (client={client_id}, job={job_id}, files={len(file_paths)})")
        return task

    def get_task(self, task_id: str) -> Optional[IngestTask]:
        """Get a task by ID."""
        return self.tasks.get(task_id)

    def get_tasks_for_client(self, client_id: str) -> list[IngestTask]:
        """Get all tasks for a client."""
        return [t for t in self.tasks.values() if t.client_id == client_id]

    async def run_ingest_task(self, task: IngestTask, model: str = "ollama/llama3") -> None:
        """
        Run the ingest task in the background.

        Called by: server.py background task after creating the task.
        """
        from matching_engine.database import ProfileDatabase
        from matching_engine.vector_store import VectorStore
        from matching_engine.scanner import scan_and_ingest

        task.status = TaskStatus.IN_PROGRESS
        logger.info(f"Task {task.task_id}: starting ingest ({task.total_files} files)")

        try:
            # Create a temporary directory concept — files are already saved
            # We'll ingest from the upload directory
            upload_dir = Path("data/uploads") / task.client_id / task.job_id
            
            db = ProfileDatabase()
            vs = VectorStore()

            result = await scan_and_ingest(
                resumes_dir=str(upload_dir),
                db=db,
                vector_store=vs,
                client_id=task.client_id,
                job_id=task.job_id,
                model=model,
                temperature=0.1,
            )

            task.processed_files = result["new_count"]
            task.failed_files = result["failed_count"]
            task.skipped_files = result["skipped_count"]
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now().isoformat()

            logger.info(
                f"Task {task.task_id}: completed "
                f"(new={task.processed_files}, failed={task.failed_files}, skipped={task.skipped_files})"
            )

            db.close()

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.now().isoformat()
            logger.error(f"Task {task.task_id}: FAILED — {e}")


# Global task manager instance
task_manager = TaskManager()
