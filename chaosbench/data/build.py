"""Dataset generation pipeline utilities."""

import hashlib
import json
from typing import Any, Dict, List

from chaosbench.data.schemas import DatasetConfig


def compute_dataset_hash(items: List[Dict[str, Any]]) -> str:
    """Compute a deterministic hash of dataset items.

    Args:
        items: List of dataset item dicts.

    Returns:
        SHA-256 hex digest string.
    """
    serialized = json.dumps(items, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def make_dataset_config(
    version: str,
    splits: Dict[str, str],
    items: List[Dict[str, Any]],
) -> DatasetConfig:
    """Create a DatasetConfig with computed content hash.

    Args:
        version: Version string.
        splits: Dict mapping split names to file paths.
        items: All dataset items for hash computation.

    Returns:
        DatasetConfig with content_hash populated.
    """
    config = DatasetConfig(version=version, splits=splits)
    content = json.dumps(items, sort_keys=True, ensure_ascii=True)
    config.compute_hash(content)
    return config
