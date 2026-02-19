"""Registry for dysts chaotic systems with programmatic access.

This module provides utilities for enumerating and accessing dynamical systems
from the dysts package, which provides implementations of continuous-time flows.

Example:
    >>> systems = list_dysts_systems(subset=5)
    >>> for sys in systems:
    ...     print(f"{sys.system_id}: {sys.class_name} (dim={sys.dimension})")
    >>>
    >>> trajectory = get_dysts_trajectory("Lorenz", n_points=1000)
    >>> print(trajectory.shape)  # (1000, 3)
"""

import inspect
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class DystsSystemInfo:
    """Metadata for a single dysts system.

    Attributes:
        class_name: Name of the dysts flow class (e.g., "Lorenz").
        system_id: Unique identifier in format "dysts_{lowercase_class_name}".
        dimension: Phase space dimension of the system.
        parameters: Dict of parameter names to default values.
        equations: String representation of the governing equations.
        description: Human-readable description of the system.
    """

    class_name: str
    system_id: str
    dimension: int
    parameters: Dict[str, float] = field(default_factory=dict)
    equations: str = ""
    description: str = ""


# Lazy-loaded registry mapping system_id -> DystsSystemInfo
_DYSTS_SYSTEM_MAP: Optional[Dict[str, DystsSystemInfo]] = None


def _is_flow_class(obj, base_class) -> bool:
    """Check if obj is a flow class (not the base class itself).

    Args:
        obj: Object to check.
        base_class: Base flow class from dysts.

    Returns:
        True if obj is a concrete flow subclass.
    """
    return (
        inspect.isclass(obj)
        and issubclass(obj, base_class)
        and obj is not base_class
        and not obj.__name__.startswith("_")
    )


def _extract_parameters(model) -> Dict[str, float]:
    """Extract parameter dict from a dysts model instance.

    Args:
        model: Instantiated dysts flow object.

    Returns:
        Dict of parameter names to float values.
    """
    params = {}
    # Try common attributes used by dysts
    for attr_name in ["params", "default_params", "_params"]:
        if hasattr(model, attr_name):
            raw_params = getattr(model, attr_name)
            if isinstance(raw_params, dict):
                params = {k: float(v) for k, v in raw_params.items()}
                break

    # Fallback: try to get from class attributes
    if not params and hasattr(model, "__dict__"):
        for key, value in model.__dict__.items():
            if not key.startswith("_") and isinstance(value, (int, float)):
                params[key] = float(value)

    return params


def _extract_equations(model, class_name: str) -> str:
    """Extract equations string from a dysts model.

    Args:
        model: Instantiated dysts flow object.
        class_name: Class name for fallback.

    Returns:
        String representation of equations.
    """
    # Try repr first
    try:
        repr_str = repr(model)
        if repr_str and not repr_str.startswith("<"):
            return repr_str
    except Exception:
        pass

    # Try equations attribute
    for attr_name in ["equations", "equation", "_equations"]:
        if hasattr(model, attr_name):
            eq = getattr(model, attr_name)
            if isinstance(eq, str) and eq:
                return eq

    # Fallback to class docstring or name
    if model.__doc__:
        # Get first line of docstring
        first_line = model.__doc__.strip().split("\n")[0]
        if first_line and len(first_line) < 200:
            return first_line

    return f"{class_name} system"


def _extract_description(model, class_name: str) -> str:
    """Extract description from a dysts model.

    Args:
        model: Instantiated dysts flow object.
        class_name: Class name for fallback.

    Returns:
        Description string.
    """
    if model.__doc__:
        doc = model.__doc__.strip()
        # Take first paragraph or first 200 chars
        paragraphs = doc.split("\n\n")
        if paragraphs:
            desc = paragraphs[0].replace("\n", " ").strip()
            if len(desc) > 200:
                desc = desc[:197] + "..."
            return desc

    return f"{class_name} dynamical system"


def _build_system_registry() -> Dict[str, DystsSystemInfo]:
    """Build the complete registry of dysts systems.

    Returns:
        Dict mapping system_id to DystsSystemInfo.
    """
    registry = {}

    try:
        import dysts.flows as flows

        # Find the base class for flows
        base_class = None
        for name, obj in inspect.getmembers(flows):
            if inspect.isclass(obj) and name in ["DynamicalSystem", "Flow", "BaseDynamicalSystem"]:
                base_class = obj
                break

        # If we can't find base class, try to use any class that looks like a parent
        if base_class is None:
            # Fallback: just check for classes with common flow names
            pass

        # Enumerate all flow classes
        for name, obj in inspect.getmembers(flows):
            if not inspect.isclass(obj):
                continue

            # Skip private/internal classes
            if name.startswith("_"):
                continue

            # Skip base classes and abstract classes
            if name in ["DynamicalSystem", "Flow", "BaseDynamicalSystem", "ContinuousModel", "DiscreteModel"]:
                continue

            # Check if it's a concrete flow class
            if base_class and not (issubclass(obj, base_class) and obj is not base_class):
                continue

            # Try to instantiate the system
            try:
                model = obj()

                # Get dimension
                dimension = 3  # default
                for attr in ["_dimension", "dimension", "dim", "n"]:
                    if hasattr(model, attr):
                        dim_val = getattr(model, attr)
                        if isinstance(dim_val, int):
                            dimension = dim_val
                            break

                # Build system info
                system_id = f"dysts_{name.lower()}"
                parameters = _extract_parameters(model)
                equations = _extract_equations(model, name)
                description = _extract_description(model, name)

                info = DystsSystemInfo(
                    class_name=name,
                    system_id=system_id,
                    dimension=dimension,
                    parameters=parameters,
                    equations=equations,
                    description=description,
                )

                registry[system_id] = info

            except Exception as e:
                # Skip systems that can't be instantiated
                continue

    except ImportError:
        # dysts not installed, return empty registry
        pass

    return registry


def list_dysts_systems(subset: Optional[int] = None) -> List[DystsSystemInfo]:
    """List all available dysts continuous-time flow systems.

    Args:
        subset: If provided, return only the first N systems (sorted by class_name).
                Useful for testing or limiting computation.

    Returns:
        List of DystsSystemInfo objects describing available systems.
        Returns empty list if dysts is not installed.

    Example:
        >>> systems = list_dysts_systems(subset=5)
        >>> print(f"Found {len(systems)} systems")
        >>> for sys in systems:
        ...     print(f"  {sys.system_id}: dim={sys.dimension}")
    """
    global _DYSTS_SYSTEM_MAP

    # Build registry lazily
    if _DYSTS_SYSTEM_MAP is None:
        _DYSTS_SYSTEM_MAP = _build_system_registry()

    # Get all systems sorted by class name
    systems = sorted(_DYSTS_SYSTEM_MAP.values(), key=lambda s: s.class_name)

    # Apply subset limit if requested
    if subset is not None and subset > 0:
        systems = systems[:subset]

    return systems


def get_dysts_trajectory(
    class_name: str,
    n_points: int = 10000,
    **kwargs
) -> np.ndarray:
    """Generate a trajectory from a dysts system.

    Args:
        class_name: Name of the dysts flow class (e.g., "Lorenz", "Rossler").
        n_points: Number of trajectory points to generate.
        **kwargs: Additional keyword arguments passed to make_trajectory
                 (e.g., method, rtol, atol, pts_per_period).

    Returns:
        NumPy array of shape (n_points, dimension) containing the trajectory.
        Returns empty array if system not found or dysts not installed.

    Raises:
        ValueError: If class_name is not found in dysts.flows.
        ImportError: If dysts is not installed.

    Example:
        >>> traj = get_dysts_trajectory("Lorenz", n_points=5000)
        >>> print(traj.shape)  # (5000, 3)
        >>> print(f"Min: {traj.min():.2f}, Max: {traj.max():.2f}")
    """
    try:
        import dysts.flows as flows

        # Get the class by name
        if not hasattr(flows, class_name):
            raise ValueError(
                f"System '{class_name}' not found in dysts.flows. "
                f"Available systems: {list_dysts_systems()}"
            )

        system_class = getattr(flows, class_name)

        # Instantiate and generate trajectory
        model = system_class()
        trajectory = model.make_trajectory(n=n_points, **kwargs)

        return trajectory

    except ImportError as e:
        raise ImportError(
            "dysts package is required to generate trajectories. "
            "Install it with: pip install dysts"
        ) from e
    except Exception as e:
        # Return empty array on error with warning
        import warnings
        warnings.warn(
            f"Failed to generate trajectory for {class_name}: {e}",
            RuntimeWarning
        )
        return np.array([])


# Internal helper to initialize module-level DYSTS_SYSTEM_MAP
def _get_system_map() -> Dict[str, DystsSystemInfo]:
    """Get the lazy-loaded system registry.

    Returns:
        Dict mapping system_id strings (e.g., "dysts_lorenz") to DystsSystemInfo.
        Returns empty dict if dysts is not installed.

    Example:
        >>> from chaosbench.data.systems.dysts_registry import DYSTS_SYSTEM_MAP
        >>> lorenz = DYSTS_SYSTEM_MAP.get("dysts_lorenz")
        >>> if lorenz:
        ...     print(f"Lorenz dimension: {lorenz.dimension}")
    """
    global _DYSTS_SYSTEM_MAP
    if _DYSTS_SYSTEM_MAP is None:
        _DYSTS_SYSTEM_MAP = _build_system_registry()
    return _DYSTS_SYSTEM_MAP


# Module-level attribute access for lazy loading DYSTS_SYSTEM_MAP
def __getattr__(name):
    """Enable lazy loading of module-level DYSTS_SYSTEM_MAP constant.

    This allows: from chaosbench.data.systems.dysts_registry import DYSTS_SYSTEM_MAP
    """
    if name == "DYSTS_SYSTEM_MAP":
        return _get_system_map()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
