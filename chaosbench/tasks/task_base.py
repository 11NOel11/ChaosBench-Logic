"""Base protocol for benchmark task types."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

from chaosbench.data.schemas import Question


class TaskProtocol(Protocol):
    """Protocol that all task types must implement."""

    task_family: str

    def generate_items(self) -> List[Question]:
        """Generate question items for this task.

        Returns:
            List of Question objects.
        """
        ...

    def score(self, predictions: Dict[str, str]) -> Dict[str, Any]:
        """Score model predictions for this task.

        Args:
            predictions: Dict mapping item_id to predicted label.

        Returns:
            Dict of metric names to values.
        """
        ...


@dataclass
class TaskConfig:
    """Configuration for a task instance.

    Attributes:
        task_family: Task type name.
        system_ids: Which systems to generate items for.
        seed: Random seed for deterministic generation.
        params: Additional task-specific parameters.
    """

    task_family: str
    system_ids: List[str] = field(default_factory=list)
    seed: int = 42
    params: Dict[str, Any] = field(default_factory=dict)
    system_pool: Optional[str] = None
    target_count: Optional[int] = None
