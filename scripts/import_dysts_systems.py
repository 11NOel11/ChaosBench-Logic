#!/usr/bin/env python3
"""Import systems from the dysts package and generate system JSON files.

This script imports dynamical systems from the dysts package, computes indicators,
infers truth assignments, and generates system JSON files compatible with the
ChaosBench-Logic dataset format.

Usage:
    python scripts/import_dysts_systems.py --output-dir systems/dysts
    python scripts/import_dysts_systems.py --subset 5 --force  # CI smoke test
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Import after path setup - import directly from modules to avoid circular dependencies
try:
    from chaosbench.data.indicators.zero_one_test import zero_one_test
    from chaosbench.data.indicators.permutation_entropy import permutation_entropy
    from chaosbench.logic.axioms import check_fol_violations
except ImportError as e:
    print(f"ERROR: Failed to import required ChaosBench modules: {e}")
    print("Make sure you're running from the project root and all dependencies are installed.")
    print("Install dependencies with: uv sync")
    sys.exit(1)


def convert_to_json_serializable(obj: Any) -> Any:
    """Convert numpy types and other non-serializable objects to JSON-compatible types.

    Args:
        obj: Object to convert.

    Returns:
        JSON-serializable version of the object.
    """
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        return obj


def check_dysts_available():
    """Check if dysts package is installed and raise clear error if not."""
    try:
        import dysts
        return dysts
    except ImportError:
        print("ERROR: dysts package not installed.")
        print("Install with: uv pip install 'dysts>=0.9,<1.0'")
        print("Or add to your environment: pip install 'dysts>=0.9,<1.0'")
        sys.exit(1)


def list_dysts_systems(subset: Optional[int] = None):
    """List available dysts systems sorted alphabetically.

    Args:
        subset: If provided, only return first N systems.

    Returns:
        List of (name, equation_cls) tuples.
    """
    dysts = check_dysts_available()

    # Get all available systems from dysts.flows module
    # System classes are the ones that start with uppercase
    import dysts.flows
    all_systems = []

    system_names = [
        name for name in dir(dysts.flows)
        if not name.startswith('_') and name[0].isupper()
    ]

    for name in sorted(system_names):
        try:
            # Get the class and instantiate it
            system_cls = getattr(dysts.flows, name)
            equation_cls = system_cls()
            all_systems.append((name, equation_cls))
        except Exception as e:
            print(f"  [WARNING] Could not load {name}: {e}")
            continue

    if subset is not None:
        all_systems = all_systems[:subset]

    return all_systems


def get_dysts_trajectory(equation_cls, n_points: int = 10000):
    """Generate a trajectory from a dysts system.

    Args:
        equation_cls: dysts DynSys instance.
        n_points: Number of points to generate.

    Returns:
        Trajectory array of shape (n_points, dimension).
    """
    # Use dysts built-in trajectory generation
    # Most dysts systems are continuous-time ODEs
    try:
        # Generate trajectory with reasonable time span
        trajectory = equation_cls.make_trajectory(
            n_points,
            resample=True,
            pts_per_period=100,
        )
        return trajectory
    except Exception as e:
        print(f"    [WARNING] Trajectory generation failed: {e}")
        return None


def compute_indicators(trajectory: np.ndarray, seed: int = 42) -> Dict[str, Optional[float]]:
    """Compute chaos indicators from trajectory.

    Args:
        trajectory: Array of shape (n_points, dimension).
        seed: Random seed for reproducible indicator computation.

    Returns:
        Dict with 'zero_one_K' and 'permutation_entropy' keys.
    """
    indicators = {
        "zero_one_K": None,
        "permutation_entropy": None,
    }

    if trajectory is None or len(trajectory) == 0:
        return indicators

    # Extract first component as 1D series
    series = trajectory[:, 0]

    # Compute zero-one test
    try:
        K = zero_one_test(series, seed=seed)
        indicators["zero_one_K"] = float(K)
    except Exception as e:
        print(f"    [WARNING] Zero-one test failed: {e}")

    # Compute permutation entropy
    try:
        pe = permutation_entropy(series, order=3, delay=1)
        indicators["permutation_entropy"] = float(pe)
    except Exception as e:
        print(f"    [WARNING] Permutation entropy failed: {e}")

    return indicators


def infer_truth_assignment(indicators: Dict[str, Optional[float]]) -> tuple[Dict[str, bool], bool]:
    """Infer truth assignment from indicators.

    Primary indicator: Permutation Entropy (threshold 0.40, 70% accuracy)
    Secondary indicator: Zero-One K (threshold 0.5, but unreliable - 63% accuracy)

    Per INDICATOR_THRESHOLDS.md, Zero-One K is poor for this dataset (63% accuracy),
    while Permutation Entropy is fair (70% accuracy). MEGNO is best but not computed
    for dysts imports.

    Args:
        indicators: Dict with 'zero_one_K' and 'permutation_entropy'.

    Returns:
        Tuple of (truth_assignment dict, uncertain flag).
    """
    zero_one_K = indicators.get("zero_one_K")
    perm_entropy = indicators.get("permutation_entropy")

    # Flag if all indicators failed
    uncertain = (zero_one_K is None and perm_entropy is None)

    # Use permutation entropy as primary indicator (better accuracy)
    # Threshold: 0.40 (empirically optimized, 70% accuracy)
    # Fall back to zero-one K if PE unavailable (threshold 0.5, but unreliable)
    if perm_entropy is not None:
        is_chaotic = perm_entropy > 0.40
    elif zero_one_K is not None:
        is_chaotic = zero_one_K > 0.5
    else:
        # Default to chaotic (most dysts systems are chaotic)
        is_chaotic = True

    if is_chaotic:
        truth_assignment = {
            "Chaotic": True,
            "Deterministic": True,
            "PosLyap": True,
            "Sensitive": True,
            "StrangeAttr": True,
            "PointUnpredictable": True,
            "StatPredictable": True,
            "QuasiPeriodic": False,
            "Random": False,
            "FixedPointAttr": False,
            "Periodic": False,
        }
    else:
        # Conservative defaults for non-chaotic dysts systems
        truth_assignment = {
            "Chaotic": False,
            "Deterministic": True,
            "PosLyap": False,
            "Sensitive": False,
            "StrangeAttr": False,
            "PointUnpredictable": False,
            "StatPredictable": True,
            "QuasiPeriodic": False,
            "Random": False,
            "FixedPointAttr": False,
            "Periodic": True,
        }

    return truth_assignment, uncertain


def validate_truth_assignment(truth_assignment: Dict[str, bool], system_id: str) -> bool:
    """Validate truth assignment for FOL consistency.

    Args:
        truth_assignment: Dict mapping predicate names to boolean values.
        system_id: System identifier for error reporting.

    Returns:
        True if valid, False if FOL violations detected.
    """
    # Convert to YES/NO format for FOL checker
    predictions = {
        pred: "YES" if val else "NO"
        for pred, val in truth_assignment.items()
    }

    violations = check_fol_violations(predictions)

    if violations:
        print(f"    [WARNING] FOL violations detected for {system_id}:")
        for v in violations:
            print(f"      - {v}")
        return False

    return True


def create_system_json(
    name: str,
    equation_cls,
    indicators: Dict[str, Optional[float]],
    truth_assignment: Dict[str, bool],
    uncertain: bool,
    dysts_version: str,
) -> Dict:
    """Create system JSON dict in ChaosBench-Logic format.

    Args:
        name: System name from dysts.
        equation_cls: dysts DynSys instance.
        indicators: Computed indicator values.
        truth_assignment: Inferred truth assignment.
        uncertain: Whether assignment is uncertain due to failed indicators.
        dysts_version: Version of dysts package.

    Returns:
        Dict representing the system JSON.
    """
    system_id = f"dysts_{name.lower().replace(' ', '_').replace('-', '_')}"
    category = "chaotic" if truth_assignment.get("Chaotic", False) else "non_chaotic"

    # Get dimension
    try:
        dim = equation_cls.embedding_dimension
    except AttributeError:
        dim = len(equation_cls.ic)

    # Get parameters
    try:
        raw_params = dict(equation_cls.params)
        # Convert numpy types to native Python types for JSON serialization
        params = convert_to_json_serializable(raw_params)
    except (AttributeError, TypeError):
        params = {}

    # Get equation string
    try:
        # Try to get LaTeX or string representation
        if hasattr(equation_cls, 'eq'):
            equations = str(equation_cls.eq)
        else:
            equations = f"{name} system (see dysts package for equations)"
    except Exception:
        equations = f"{name} system (see dysts package for equations)"

    # Build system dict
    system_data = {
        "system_id": system_id,
        "name": f"{name} (dysts)",
        "category": category,
        "equations": equations,
        "parameters": params,
        "dimension": dim,
        "description_simple": f"A dynamical system from the dysts package.",
        "description_complex": (
            f"A {'chaotic' if category == 'chaotic' else 'non-chaotic'} dynamical system "
            f"imported from the dysts package. This system exhibits "
            f"{'complex, sensitive dynamics characteristic of deterministic chaos.' if category == 'chaotic' else 'regular, predictable dynamics.'}"
        ),
        "truth_assignment": truth_assignment,
        "provenance": {
            "source": "dysts",
            "cite": "arXiv:2110.05266",
            "dysts_version": dysts_version,
            "import_timestamp": datetime.now(timezone.utc).isoformat(),
            "uncertain": uncertain,
        }
    }

    return system_data


def create_indicator_json(
    system_id: str,
    indicators: Dict[str, Optional[float]],
    seed: int,
) -> Dict:
    """Create indicator JSON dict.

    Args:
        system_id: System identifier.
        indicators: Computed indicator values.
        seed: Random seed used.

    Returns:
        Dict representing the indicator JSON.
    """
    indicator_data = {
        "system_id": system_id,
        "zero_one_K": indicators.get("zero_one_K"),
        "permutation_entropy": indicators.get("permutation_entropy"),
        "megno": None,  # Not computed for dysts systems
        "system_type": "ode",
        "seed": seed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return indicator_data


def load_dysts_metadata() -> Dict[str, Dict]:
    """Load built-in metadata from dysts package (chaotic_attractors.json).

    Returns:
        Dict mapping system name to metadata dict with Lyapunov exponents,
        parameters, description, etc.
    """
    import dysts
    data_path = os.path.join(os.path.dirname(dysts.__file__), "data", "chaotic_attractors.json")
    with open(data_path) as f:
        return json.load(f)


def import_system_from_metadata(
    name: str,
    equation_cls,
    metadata: Dict,
    output_dir: str,
    indicators_dir: str,
    force: bool,
    dysts_version: str,
) -> tuple[bool, bool, bool]:
    """Import a single system using dysts built-in metadata (no trajectory computation).

    Uses pre-computed Lyapunov exponents from the dysts package to classify systems.
    All 135 continuous flows in dysts have positive max Lyapunov exponents.

    Args:
        name: System name.
        equation_cls: dysts DynSys instance.
        metadata: Pre-loaded dysts metadata for this system.
        output_dir: Directory for system JSON files.
        indicators_dir: Directory for indicator JSON files.
        force: Overwrite existing files.
        dysts_version: Version of dysts package.

    Returns:
        Tuple of (success, is_chaotic, uncertain).
    """
    system_id = f"dysts_{name.lower().replace(' ', '_').replace('-', '_')}"
    system_path = os.path.join(output_dir, f"{system_id}.json")
    indicator_path = os.path.join(indicators_dir, f"{system_id}_indicators.json")

    if not force and os.path.exists(system_path):
        print(f"  [SKIP] {name} (already exists, use --force to overwrite)")
        return False, False, False

    print(f"  [IMPORT] {name}")

    max_lyap = metadata.get("maximum_lyapunov_estimated", 0)
    lyap_spectrum = metadata.get("lyapunov_spectrum_estimated", [])
    is_chaotic = max_lyap is not None and max_lyap > 0
    uncertain = max_lyap is None

    if is_chaotic:
        truth_assignment = {
            "Chaotic": True,
            "Deterministic": True,
            "PosLyap": True,
            "Sensitive": True,
            "StrangeAttr": True,
            "PointUnpredictable": True,
            "StatPredictable": True,
            "QuasiPeriodic": False,
            "Random": False,
            "FixedPointAttr": False,
            "Periodic": False,
        }
    else:
        truth_assignment = {
            "Chaotic": False,
            "Deterministic": True,
            "PosLyap": False,
            "Sensitive": False,
            "StrangeAttr": False,
            "PointUnpredictable": False,
            "StatPredictable": True,
            "QuasiPeriodic": False,
            "Random": False,
            "FixedPointAttr": False,
            "Periodic": True,
        }

    is_valid = validate_truth_assignment(truth_assignment, system_id)
    if not is_valid:
        print(f"    [ERROR] FOL validation failed")
        return False, False, False

    indicators = {
        "maximum_lyapunov_estimated": max_lyap,
        "lyapunov_spectrum_estimated": lyap_spectrum,
        "kaplan_yorke_dimension": metadata.get("kaplan_yorke_dimension"),
        "correlation_dimension": metadata.get("correlation_dimension"),
    }

    system_data = create_system_json(
        name,
        equation_cls,
        indicators,
        truth_assignment,
        uncertain,
        dysts_version,
    )

    if metadata.get("description"):
        system_data["description_simple"] = metadata["description"]

    indicator_data = {
        "system_id": system_id,
        "maximum_lyapunov_estimated": max_lyap,
        "lyapunov_spectrum": lyap_spectrum,
        "kaplan_yorke_dimension": metadata.get("kaplan_yorke_dimension"),
        "correlation_dimension": metadata.get("correlation_dimension"),
        "multiscale_entropy": metadata.get("multiscale_entropy"),
        "source": "dysts_metadata",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    with open(system_path, "w") as f:
        json.dump(system_data, f, indent=2)
    print(f"    Wrote {system_path}")

    with open(indicator_path, "w") as f:
        json.dump(convert_to_json_serializable(indicator_data), f, indent=2)
    print(f"    Wrote {indicator_path}")

    return True, is_chaotic, uncertain


def import_system(
    name: str,
    equation_cls,
    output_dir: str,
    indicators_dir: str,
    seed: int,
    force: bool,
    dysts_version: str,
) -> tuple[bool, bool, bool]:
    """Import a single system from dysts using trajectory computation (slow path).

    Args:
        name: System name.
        equation_cls: dysts DynSys instance.
        output_dir: Directory for system JSON files.
        indicators_dir: Directory for indicator JSON files.
        seed: Random seed.
        force: Overwrite existing files.
        dysts_version: Version of dysts package.

    Returns:
        Tuple of (success, is_chaotic, uncertain).
    """
    system_id = f"dysts_{name.lower().replace(' ', '_').replace('-', '_')}"
    system_path = os.path.join(output_dir, f"{system_id}.json")
    indicator_path = os.path.join(indicators_dir, f"{system_id}_indicators.json")

    if not force and os.path.exists(system_path):
        print(f"  [SKIP] {name} (already exists, use --force to overwrite)")
        return False, False, False

    print(f"  [IMPORT] {name}")

    print(f"    Generating trajectory...")
    trajectory = get_dysts_trajectory(equation_cls, n_points=10000)

    if trajectory is None:
        print(f"    [ERROR] Failed to generate trajectory")
        return False, False, False

    print(f"    Computing indicators...")
    indicators = compute_indicators(trajectory, seed=seed)

    print(f"    Inferring truth assignment...")
    truth_assignment, uncertain = infer_truth_assignment(indicators)

    if uncertain:
        print(f"    [WARNING] Indicators failed, using default chaotic assignment")

    print(f"    Validating FOL consistency...")
    is_valid = validate_truth_assignment(truth_assignment, system_id)

    if not is_valid:
        print(f"    [ERROR] FOL validation failed")
        return False, False, False

    system_data = create_system_json(
        name,
        equation_cls,
        indicators,
        truth_assignment,
        uncertain,
        dysts_version,
    )

    indicator_data = create_indicator_json(
        system_id,
        indicators,
        seed,
    )

    with open(system_path, "w") as f:
        json.dump(system_data, f, indent=2)
    print(f"    Wrote {system_path}")

    with open(indicator_path, "w") as f:
        json.dump(indicator_data, f, indent=2)
    print(f"    Wrote {indicator_path}")

    is_chaotic = truth_assignment.get("Chaotic", False)

    return True, is_chaotic, uncertain


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Import systems from dysts package and generate system JSON files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="systems/dysts",
        help="Directory to write system JSON files (default: systems/dysts)",
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Only import first N systems (sorted alphabetically), for CI smoke tests",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for trajectory generation and indicator computation (default: 42)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files",
    )
    parser.add_argument(
        "--slow",
        action="store_true",
        help="Use slow trajectory-based classification instead of dysts metadata",
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Check dysts availability
    dysts = check_dysts_available()
    try:
        dysts_version = dysts.__version__
    except AttributeError:
        # Some versions of dysts don't have __version__
        try:
            import importlib.metadata
            dysts_version = importlib.metadata.version('dysts')
        except Exception:
            dysts_version = "unknown"

    print("=" * 70)
    print("  ChaosBench-Logic: dysts System Importer")
    print("=" * 70)
    print(f"  Output directory:  {args.output_dir}")
    print(f"  Seed:              {args.seed}")
    print(f"  Force overwrite:   {args.force}")
    print(f"  Subset:            {args.subset if args.subset else 'all'}")
    print(f"  dysts version:     {dysts_version}")
    print(f"  Mode:              {'slow (trajectory)' if args.slow else 'fast (metadata)'}")
    print()

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    indicators_dir = os.path.join(args.output_dir, "indicators")
    os.makedirs(indicators_dir, exist_ok=True)

    # List systems
    print("Loading dysts systems...")
    systems = list_dysts_systems(subset=args.subset)
    print(f"Found {len(systems)} systems\n")

    # Load metadata for fast path
    dysts_metadata = {}
    if not args.slow:
        print("Loading dysts metadata (Lyapunov exponents, dimensions)...")
        dysts_metadata = load_dysts_metadata()
        print(f"Loaded metadata for {len(dysts_metadata)} systems\n")

    # Import systems
    stats = {
        "total": len(systems),
        "imported": 0,
        "skipped": 0,
        "failed": 0,
        "chaotic": 0,
        "non_chaotic": 0,
        "uncertain": 0,
    }

    for name, equation_cls in systems:
        if not args.slow and name in dysts_metadata:
            success, is_chaotic, uncertain = import_system_from_metadata(
                name,
                equation_cls,
                dysts_metadata[name],
                args.output_dir,
                indicators_dir,
                args.force,
                dysts_version,
            )
        else:
            success, is_chaotic, uncertain = import_system(
                name,
                equation_cls,
                args.output_dir,
                indicators_dir,
                args.seed,
                args.force,
                dysts_version,
            )

        if success:
            stats["imported"] += 1
            if is_chaotic:
                stats["chaotic"] += 1
            else:
                stats["non_chaotic"] += 1
            if uncertain:
                stats["uncertain"] += 1
        else:
            # Check if skipped or failed
            system_id = f"dysts_{name.lower().replace(' ', '_').replace('-', '_')}"
            system_path = os.path.join(args.output_dir, f"{system_id}.json")
            if os.path.exists(system_path):
                stats["skipped"] += 1
            else:
                stats["failed"] += 1

    # Print summary
    print()
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Total systems:           {stats['total']}")
    print(f"  Successfully imported:   {stats['imported']}")
    print(f"  Skipped (existing):      {stats['skipped']}")
    print(f"  Failed:                  {stats['failed']}")
    print()
    print(f"  Chaotic systems:         {stats['chaotic']}")
    print(f"  Non-chaotic systems:     {stats['non_chaotic']}")
    print(f"  Uncertain assignments:   {stats['uncertain']}")
    print()
    print(f"  System files:            {args.output_dir}/")
    print(f"  Indicator files:         {indicators_dir}/")
    print("=" * 70)

    if stats["failed"] > 0:
        print("\n[WARNING] Some systems failed to import. Check logs above.")
        sys.exit(1)

    print("\nDone!")


if __name__ == "__main__":
    main()
