"""SQLite-backed response cache for evaluation runner."""

import hashlib
import os
import sqlite3
from typing import Dict, Optional


class ResponseCache:
    """SQLite-backed cache for model responses.

    Caches responses by (model, mode, item_id, question_sha256) to avoid
    redundant API calls during evaluation runs.
    """

    def __init__(self, cache_dir: str):
        """Initialize cache with SQLite database.

        Args:
            cache_dir: Directory for cache database file.
        """
        os.makedirs(cache_dir, exist_ok=True)
        self.db_path = os.path.join(cache_dir, "response_cache.db")
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_db()

    def _init_db(self):
        """Create cache table if it doesn't exist."""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS responses (
                model TEXT NOT NULL,
                mode TEXT NOT NULL,
                item_id TEXT NOT NULL,
                question_sha256 TEXT NOT NULL,
                response TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                PRIMARY KEY (model, mode, item_id, question_sha256)
            )
        """)
        self.conn.commit()

    def get(self, model: str, mode: str, item_id: str, question: str) -> Optional[str]:
        """Retrieve cached response.

        Args:
            model: Model name.
            mode: Evaluation mode.
            item_id: Item identifier.
            question: Question text.

        Returns:
            Cached response or None if not found.
        """
        question_hash = hashlib.sha256(question.encode('utf-8')).hexdigest()
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT response FROM responses
            WHERE model = ? AND mode = ? AND item_id = ? AND question_sha256 = ?
            """,
            (model, mode, item_id, question_hash)
        )
        row = cursor.fetchone()
        return row[0] if row else None

    def put(self, model: str, mode: str, item_id: str, question: str, response: str) -> None:
        """Store response in cache.

        Args:
            model: Model name.
            mode: Evaluation mode.
            item_id: Item identifier.
            question: Question text.
            response: Model response to cache.
        """
        question_hash = hashlib.sha256(question.encode('utf-8')).hexdigest()
        from datetime import datetime
        timestamp = datetime.now().isoformat()

        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO responses
            (model, mode, item_id, question_sha256, response, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (model, mode, item_id, question_hash, response, timestamp)
        )
        self.conn.commit()

    def invalidate(self, model: str, mode: str, item_id: Optional[str] = None) -> int:
        """Delete cache entries.

        Args:
            model: Model name.
            mode: Evaluation mode.
            item_id: Optional item identifier. If None, deletes all for model+mode.

        Returns:
            Number of entries deleted.
        """
        cursor = self.conn.cursor()
        if item_id is None:
            cursor.execute(
                "DELETE FROM responses WHERE model = ? AND mode = ?",
                (model, mode)
            )
        else:
            cursor.execute(
                "DELETE FROM responses WHERE model = ? AND mode = ? AND item_id = ?",
                (model, mode, item_id)
            )
        self.conn.commit()
        return cursor.rowcount

    def stats(self) -> Dict[str, int]:
        """Get cache statistics.

        Returns:
            Dict with total_entries and models counts.
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM responses")
        total = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT model) FROM responses")
        models = cursor.fetchone()[0]

        return {"total_entries": total, "models": models}

    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
