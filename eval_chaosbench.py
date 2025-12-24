#!/usr/bin/env python3
"""
Evaluation script for ChaosBench-Logic.

- Loads JSONL batches of questions.
- Calls a model client (you plug in your API).
- Normalizes answers to YES/NO or similar labels.
- Computes metrics:
    * overall accuracy
    * accuracy by task_family
    * bias error rates
    * dialogue accuracy and contradiction rate
- Exports:
    * per_item_results.jsonl
    * summary.json
    * metrics_overview.csv
    * accuracy_by_task.csv
    * bias_errors.csv
    * task_accuracy_bar.png
    * bias_error_bar.png
"""

import argparse
import csv
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import matplotlib.pyplot as plt


############################
# 1. MODEL CLIENT INTERFACE
############################

@dataclass
class ModelConfig:
    name: str          # e.g. "gpt4", "claude3", "gemini", "llama3"
    mode: str          # "zeroshot", "cot", or "tool"
    temperature: float = 0.0
    max_tokens: int = 512


class ModelClient:
    """Base interface. Implement call() for each backend."""

    def __init__(self, config: ModelConfig):
        self.config = config

    def build_prompt(
        self,
        question: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build a prompt string based on mode and optional numeric/tool context."""
        context = context or {}
        mode = self.config.mode

        if mode == "zeroshot":
            return (
                "Answer the following question with either YES or NO.\n"
                "Provide your answer in this exact format:\n"
                "FINAL_ANSWER: YES\n"
                "or\n"
                "FINAL_ANSWER: NO\n\n"
                f"Question: {question}\n\n"
                "FINAL_ANSWER:"
            )

        if mode == "cot":
            return (
                "You are a careful mathematical assistant.\n"
                "Think step by step, then provide your final answer.\n\n"
                f"Question: {question}\n\n"
                "Instructions:\n"
                "1. Reason through the problem step by step\n"
                "2. On the last line, write your final answer in this exact format:\n"
                "   FINAL_ANSWER: YES\n"
                "   or\n"
                "   FINAL_ANSWER: NO\n\n"
                "Your response:"
            )

        if mode == "tool":
            numeric = context.get("numeric_facts", "")
            return (
                "You are a reasoning assistant with access to numeric facts "
                "about a dynamical system (e.g., Lyapunov exponents, attractor type).\n"
                "Use these facts to reason logically.\n\n"
                f"Facts:\n{numeric}\n\n"
                f"Question: {question}\n\n"
                "Instructions:\n"
                "1. Use the provided facts to reason step by step\n"
                "2. On the last line, write your final answer in this exact format:\n"
                "   FINAL_ANSWER: YES\n"
                "   or\n"
                "   FINAL_ANSWER: NO\n\n"
                "Your response:"
            )

        # Fallback
        return question

    def call(self, prompt: str) -> str:
        """Override this for each model type."""
        raise NotImplementedError("Implement call() in a subclass or factory.")


class DummyEchoModel(ModelClient):
    """
    Fallback model that just echo-answers 'YES'.
    Use only to test pipeline & metrics wiring.
    """

    def call(self, prompt: str) -> str:
        return "Final answer: YES"


def make_model_client(config: ModelConfig) -> ModelClient:
    """
    Factory to create model clients.
    Falls back to DummyEchoModel for testing if real clients fail.
    """
    # Use dummy model for testing
    if config.name.lower() in ("dummy", "test"):
        return DummyEchoModel(config)
    
    # Try to import and use real clients
    try:
        from clients import build_client
        return build_client(config)
    except (ImportError, ValueError) as e:
        print(f"Warning: Could not load model '{config.name}': {e}")
        print("Falling back to DummyEchoModel for testing")
        return DummyEchoModel(config)


############################
# 2. DATA LOADING
############################

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_batches(batch_paths: List[str]) -> List[Dict[str, Any]]:
    all_items: List[Dict[str, Any]] = []
    for bp in batch_paths:
        items = load_jsonl(bp)
        base = os.path.basename(bp)
        for item in items:
            item["_batch_file"] = base
        all_items.extend(items)
    return all_items


############################
# 3. ANSWER NORMALIZATION
############################

YES_SET = {"yes", "true", "y", "t"}
NO_SET = {"no", "false", "n", "f"}


def normalize_label(text: Optional[str]) -> Optional[str]:
    """
    Normalize a free-form model answer into 'YES' or 'NO'.

    ULTRA-ROBUST version that handles:
      - "FINAL_ANSWER: YES" or "FINAL ANSWER: YES" (structured format)
      - "Final answer: YES" or "final answer YES" (CoT format)
      - "Answer: YES" or "answer YES"
      - "The answer is YES"
      - "**Final Answer:** YES" (markdown)
      - "YES" anywhere in the response
      - "TRUE" / "FALSE" (converts to YES/NO)
      - Embedded in sentences: "so the answer is YES"

    Strategy: Try increasingly aggressive pattern matching.
    """
    if text is None:
        return None

    raw = text.strip()
    if not raw:
        return None

    import re

    # Clean up markdown and formatting
    text_cleaned = re.sub(r'[*_`]', '', raw)  # Remove markdown
    
    # Step 1: Look for "FINAL_ANSWER:" or "FINAL ANSWER:" (highest priority)
    final_answer_match = re.search(
        r'FINAL[_\s-]*ANSWER\s*[:=]\s*([^\n.,;]+)', 
        text_cleaned, 
        re.IGNORECASE
    )
    if final_answer_match:
        answer_part = final_answer_match.group(1).strip()
    else:
        # Step 2: Try multiple answer extraction patterns
        answer_patterns = [
            r'(?:final|ultimate|my)\s+answer\s*[:=]?\s*([^\n.,;]+)',
            r'(?:the\s+)?answer\s+is\s+([^\n.,;]+)',
            r'(?:i\s+)?answer\s*[:=]\s*([^\n.,;]+)',
            r'therefore\s*[,:=]?\s*([^\n.,;]+)',
            r'conclusion\s*[:=]\s*([^\n.,;]+)',
            r'so\s+(?:the\s+answer\s+is\s+)?([^\n.,;]+)',
        ]

        answer_part = None
        for pattern in answer_patterns:
            match = re.search(pattern, text_cleaned, re.IGNORECASE)
            if match:
                answer_part = match.group(1).strip()
                break

        # Step 3: If no pattern found, try last non-empty line
        if answer_part is None:
            lines = [l.strip() for l in text_cleaned.split('\n') if l.strip()]
            if lines:
                answer_part = lines[-1]

        # Step 4: Fallback to full text
        if not answer_part:
            answer_part = text_cleaned.strip()

    if not answer_part:
        return None

    # Step 5: Clean the answer part - remove punctuation and normalize
    cleaned = re.sub(r'[^\w\s]', ' ', answer_part)
    cleaned = ' '.join(cleaned.split()).lower()

    if not cleaned:
        # Step 6: Last resort - search for YES/NO/TRUE/FALSE anywhere in full text
        full_lower = raw.lower()
        # Look for standalone YES/NO/TRUE/FALSE
        if re.search(r'\byes\b', full_lower):
            return "YES"
        if re.search(r'\bno\b', full_lower):
            return "NO"
        if re.search(r'\btrue\b', full_lower):
            return "YES"
        if re.search(r'\bfalse\b', full_lower):
            return "NO"
        return None

    # Step 7: Look for YES/NO/TRUE/FALSE tokens
    tokens = cleaned.split()

    if not tokens:
        return None

    # Check first token (most common case)
    first_token = tokens[0]
    if first_token in YES_SET:
        return "YES"
    if first_token in NO_SET:
        return "NO"

    # Check if YES/TRUE/NO/FALSE appears as any standalone token
    # Iterate in REVERSE to find LAST occurrence (important for CoT)
    last_yes_token = None
    last_no_token = None
    for i, token in enumerate(tokens):
        if token in {"yes", "true"}:
            last_yes_token = i
        if token in {"no", "false"}:
            last_no_token = i

    # Return based on which appeared last
    if last_yes_token is not None and last_no_token is not None:
        if last_yes_token > last_no_token:
            return "YES"
        else:
            return "NO"
    elif last_yes_token is not None:
        return "YES"
    elif last_no_token is not None:
        return "NO"

    # Step 8: Last-resort fallback - find LAST occurrence of YES/NO in text
    # This handles CoT cases like "Let me think... initially seems yes, but actually no"
    # where the final answer is at the end without explicit FINAL_ANSWER: marker
    full_lower = raw.lower()
    last_yes_idx = max(
        full_lower.rfind(' yes'),
        full_lower.rfind(' yes.'),
        full_lower.rfind(' yes,'),
        full_lower.rfind('\nyes'),
        full_lower.rfind(' true'),
        full_lower.rfind(' true.'),
    )
    last_no_idx = max(
        full_lower.rfind(' no'),
        full_lower.rfind(' no.'),
        full_lower.rfind(' no,'),
        full_lower.rfind('\nno'),
        full_lower.rfind(' false'),
        full_lower.rfind(' false.'),
    )

    if last_yes_idx > last_no_idx and last_yes_idx >= 0:
        # YES appears after NO (or NO doesn't appear)
        return "YES"
    elif last_no_idx > last_yes_idx and last_no_idx >= 0:
        # NO appears after YES (or YES doesn't appear)
        return "NO"

    # Give up
    return None


############################
# 4. FOL VIOLATION CHECKING
############################

def get_fol_rules() -> Dict[str, Dict[str, List[str]]]:
    """
    Returns the first-order logic (FOL) axioms for chaotic dynamical systems.

    These rules define necessary and exclusionary relationships between predicates.
    Based on the formal ontology in the ChaosBench-Logic paper.

    Returns:
        Dict mapping predicate names to {'requires': [...], 'excludes': [...]}

    Examples:
        - If Chaotic=YES, then Deterministic, PosLyap, Sensitive must also be YES
        - If Chaotic=YES, then Random, Periodic, QuasiPeriodic must be NO
    """
    return {
        "Chaotic": {
            "requires": ["Deterministic", "PosLyap", "Sensitive", "PointUnpredictable", "StatPredictable"],
            "excludes": ["Random", "Periodic", "QuasiPeriodic", "FixedPointAttr"]
        },
        "Random": {
            "requires": [],
            "excludes": ["Deterministic", "Chaotic", "QuasiPeriodic", "Periodic"]
        },
        "QuasiPeriodic": {
            "requires": ["Deterministic"],
            "excludes": ["Chaotic", "Random", "Periodic", "FixedPointAttr"]
        },
        "Periodic": {
            "requires": ["Deterministic"],
            "excludes": ["Chaotic", "Random", "QuasiPeriodic", "StrangeAttr"]
        },
        "FixedPointAttr": {
            "requires": ["Deterministic"],
            "excludes": ["Chaotic", "Random", "QuasiPeriodic", "Periodic", "StrangeAttr"]
        },
        "Deterministic": {
            "requires": [],
            "excludes": ["Random"]
        },
    }


def load_system_ontology(systems_dir: str = "systems") -> Dict[str, Dict[str, bool]]:
    """
    Load ground truth ontology for all dynamical systems.

    Reads all JSON files from systems/ directory and extracts truth_assignment
    for each system's logical predicates.

    Args:
        systems_dir: Path to directory containing system JSON files

    Returns:
        Dict mapping system_id -> {predicate_name: bool_value}
        Example: {"lorenz63": {"Chaotic": true, "Deterministic": true, ...}, ...}

    Raises:
        FileNotFoundError: If systems_dir doesn't exist
        JSONDecodeError: If any JSON file is malformed
    """
    ontology: Dict[str, Dict[str, bool]] = {}

    if not os.path.exists(systems_dir):
        print(f"[WARNING] Systems directory not found: {systems_dir}")
        return ontology

    for filename in os.listdir(systems_dir):
        if not filename.endswith('.json'):
            continue

        filepath = os.path.join(systems_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                system_id = data.get("system_id")
                truth_assignment = data.get("truth_assignment", {})

                if system_id and truth_assignment:
                    ontology[system_id] = truth_assignment
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[WARNING] Failed to parse {filename}: {e}")
            continue

    return ontology


def extract_predicate_from_question(question: str) -> Optional[str]:
    """
    Extract the logical predicate being queried from a question.

    Uses keyword matching to identify which of the 11 FOL predicates
    a question is asking about.

    Args:
        question: The question text

    Returns:
        Predicate name (e.g., "Chaotic", "Deterministic") or None if unknown

    Examples:
        "Is the Lorenz-63 system chaotic?" → "Chaotic"
        "Does Lorenz-63 have a positive Lyapunov exponent?" → "PosLyap"
        "Is the system deterministic?" → "Deterministic"
    """
    if not question:
        return None

    q_lower = question.lower()

    # Keyword mappings (order matters - check specific before general)
    keyword_map = [
        (["chaotic", "chaos"], "Chaotic"),
        (["deterministic"], "Deterministic"),
        (["positive lyapunov", "poslyap", "largest lyapunov exponent"], "PosLyap"),
        (["sensitive dependence", "sensitivity to initial conditions", "sensitive"], "Sensitive"),
        (["strange attractor"], "StrangeAttr"),
        (["pointwise prediction", "point-wise prediction", "pointunpredictable", "long-term pointwise"], "PointUnpredictable"),
        (["statistically predictable", "statistical prediction", "statpredictable"], "StatPredictable"),
        (["quasi-periodic", "quasiperiodic"], "QuasiPeriodic"),
        (["random", "randomness", "stochastic"], "Random"),
        (["fixed point", "fixedpoint"], "FixedPointAttr"),
        (["periodic"], "Periodic"),
    ]

    for keywords, predicate in keyword_map:
        if any(kw in q_lower for kw in keywords):
            return predicate

    return None


def check_fol_violations(
    predictions: Dict[str, str],
    ground_truth: Optional[Dict[str, bool]] = None
) -> List[str]:
    """
    Check for first-order logic violations in a set of predictions.

    Compares model predictions against FOL axioms to identify logical inconsistencies.
    For example, if the model says Chaotic=YES but Deterministic=NO, this violates
    the axiom "Chaotic → Deterministic".

    Args:
        predictions: Model predictions as {predicate: "YES"|"NO"}
                    Example: {"Chaotic": "YES", "Deterministic": "NO", ...}
        ground_truth: Optional ground truth as {predicate: bool} (for debugging)

    Returns:
        List of violated implication strings
        Example: ["Chaotic → Deterministic", "Chaotic → PosLyap"]

    Notes:
        - Only checks predicates that appear in predictions
        - Treats missing predicates as "unknown" (no violation)
        - Single questions are treated as length-1 dialogues
    """
    violations: List[str] = []
    fol_rules = get_fol_rules()

    # Convert predictions to boolean for easier checking
    pred_bool: Dict[str, bool] = {}
    for pred_name, pred_value in predictions.items():
        if pred_value == "YES":
            pred_bool[pred_name] = True
        elif pred_value == "NO":
            pred_bool[pred_name] = False
        # else: unknown/unparsed, skip

    # Check each predicate's implications
    for predicate, is_true in pred_bool.items():
        if predicate not in fol_rules:
            continue  # No rules for this predicate

        rules = fol_rules[predicate]

        # If this predicate is TRUE, check requirements and exclusions
        if is_true:
            # Check "requires" implications
            for required_pred in rules.get("requires", []):
                if required_pred in pred_bool:
                    if not pred_bool[required_pred]:
                        # Violation: predicate is true but required predicate is false
                        violations.append(f"{predicate} → {required_pred}")

            # Check "excludes" implications
            for excluded_pred in rules.get("excludes", []):
                if excluded_pred in pred_bool:
                    if pred_bool[excluded_pred]:
                        # Violation: predicate is true but excluded predicate is also true
                        violations.append(f"{predicate} → ¬{excluded_pred}")

    return violations


############################
# 5. METRIC DATA STRUCTURE
############################

@dataclass
class EvalResult:
    item_id: str
    batch_file: str
    task_family: str
    bias_family: Optional[str]
    dialogue_id: Optional[str]
    turn_index: Optional[int]
    system_id: Optional[str]
    gold: Optional[str]
    pred_raw: Optional[str]
    pred_norm: Optional[str]
    correct: Optional[bool]
    error_type: Optional[str] = None  # Track error type for failed items
    question: Optional[str] = None  # Question text (for FOL predicate extraction)


############################
# 5. RATE LIMITING & RETRY CONFIGURATION
############################

# Provider-specific rate limiting policies
# UPDATED FOR PAID ACCOUNTS - More aggressive limits with sufficient funds
PROVIDER_POLICIES = {
    "openai": {"max_workers": 5, "delay": 0.1},       # 5 workers, 100ms delay (paid tier: 10k RPM, 2M TPM)
    "anthropic": {"max_workers": 4, "delay": 0.15},   # 4 workers, 150ms delay (paid tier: 4k RPM)
    "google": {"max_workers": 8, "delay": 0.05},      # 8 workers, 50ms delay (paid tier: high limits)
    "huggingface": {"max_workers": 3, "delay": 0.15}, # 3 workers, 150ms delay
    "default": {"max_workers": 2, "delay": 0.2},      # Safe default
}

def get_provider_policy(model_name: str) -> Dict[str, Any]:
    """Infer provider from model name and return rate limiting policy."""
    model_lower = model_name.lower()

    if "gpt" in model_lower or "openai" in model_lower:
        return PROVIDER_POLICIES["openai"]
    elif "claude" in model_lower or "anthropic" in model_lower:
        return PROVIDER_POLICIES["anthropic"]
    elif "gemini" in model_lower or "google" in model_lower:
        return PROVIDER_POLICIES["google"]
    elif any(keyword in model_lower for keyword in ["llama", "mixtral", "openhermes", "huggingface", "hf"]):
        return PROVIDER_POLICIES["huggingface"]
    else:
        return PROVIDER_POLICIES["default"]


def retry_with_backoff(func, max_retries=4, initial_delay=1.0):
    """
    Retry a function with exponential backoff.

    Catches HTTP 429 (rate limit) and 5xx errors.
    Backoff schedule: 1s → 2s → 4s → 8s

    Returns: (result, error_type, error_msg)
        - If successful: (result, None, None)
        - If all retries fail: (None, error_type, error_msg)

    Error types:
        - "RateLimitError" (429)
        - "AuthError" (401, 403)
        - "ServerError" (500-504)
        - "TimeoutError"
        - "InvalidAPIKeyError"
        - "OtherError"
    """
    last_error = None
    last_error_type = "OtherError"

    for attempt in range(max_retries + 1):
        try:
            result = func()
            return result, None, None
        except Exception as e:
            last_error = e
            error_str = str(e).lower()

            # Classify error type
            if "429" in error_str or "rate limit" in error_str or "rate_limit" in error_str:
                error_type = "RateLimitError"
                is_retryable = True
            elif "401" in error_str or "403" in error_str or "authentication" in error_str or "unauthorized" in error_str:
                error_type = "AuthError"
                is_retryable = False
            elif "api key" in error_str or "api_key" in error_str or "invalid x-api-key" in error_str:
                error_type = "InvalidAPIKeyError"
                is_retryable = False
            elif any(code in error_str for code in ["500", "502", "503", "504"]):
                error_type = "ServerError"
                is_retryable = True
            elif "timeout" in error_str:
                error_type = "TimeoutError"
                is_retryable = True
            else:
                error_type = "OtherError"
                is_retryable = False

            last_error_type = error_type

            if not is_retryable:
                # Non-retryable error, fail immediately
                return None, error_type, f"{error_type}: {str(e)[:200]}"

            if attempt < max_retries:
                delay = initial_delay * (2 ** attempt)
                print(f"  Retry attempt {attempt + 1}/{max_retries} after {delay}s ({error_type}: {str(e)[:100]})")
                time.sleep(delay)
            else:
                return None, error_type, f"All {max_retries} retries failed - {error_type}: {str(e)[:200]}"

    return None, last_error_type, f"Unexpected retry loop exit: {str(last_error)[:200]}"


############################
# 6. ROBUST PARALLEL EVALUATION
############################

def evaluate_single_item_robust(
    item: Dict[str, Any],
    client: ModelClient,
    numeric_fact_map: Dict[str, str],
    delay: float = 0.0,
) -> EvalResult:
    """
    Evaluate a single item with retry logic and robust field extraction.
    
    Args:
        item: Question item from dataset
        client: Model client
        numeric_fact_map: System ID -> numeric facts mapping
        delay: Delay before making API call (for rate limiting)
    
    Returns:
        EvalResult with correct=None if all retries failed
    """
    # Rate limiting delay
    if delay > 0:
        time.sleep(delay)
    
    # Extract fields robustly
    q = item.get("question", "")
    system_id = item.get("system_id")
    
    # Build context for tool mode
    ctx: Dict[str, Any] = {}
    if client.config.mode == "tool" and system_id in numeric_fact_map:
        ctx["numeric_facts"] = numeric_fact_map[system_id]
    
    # Robust gold label extraction
    gold_raw = (
        item.get("ground_truth")
        or item.get("gold_label")
        or item.get("gold")
        or item.get("answer")
        or item.get("label")
    )
    gold = normalize_label(gold_raw)
    
    # Robust task family extraction
    task_family = item.get("task_family") or item.get("type") or "unknown"
    
    # Robust bias family extraction
    bias_family = item.get("bias_family") or item.get("bias_type") or item.get("bias")
    
    # Robust turn index extraction
    turn_index = item.get("turn_index") or item.get("turn")
    
    # Try to get prediction with retry
    def call_model():
        prompt = client.build_prompt(q, context=ctx)
        return client.call(prompt)

    pred_text, error_type, error_msg = retry_with_backoff(call_model, max_retries=4, initial_delay=1.0)

    # Normalize prediction
    pred_norm = normalize_label(pred_text) if pred_text is not None else None
    
    # Log warning if normalization failed despite having a response
    if pred_text is not None and pred_norm is None:
        print(f"  ⚠️  Warning: normalize_label failed to extract YES/NO from: {pred_text[:100]}...")

    # Determine correctness
    correct: Optional[bool] = None
    if pred_text is None:
        # All retries failed - mark as None
        correct = None
    elif gold is not None and pred_norm is not None:
        correct = (pred_norm == gold)
    elif gold is not None and pred_norm is None:
        # Got response but couldn't parse it - treat as incorrect
        correct = False

    return EvalResult(
        item_id=item.get("id", ""),
        batch_file=item.get("_batch_file", ""),
        task_family=task_family,
        bias_family=bias_family,
        dialogue_id=item.get("dialogue_id"),
        turn_index=turn_index,
        system_id=system_id,
        gold=gold,
        pred_raw=pred_text if pred_text is not None else f"ERROR: {error_msg}" if error_msg else None,
        pred_norm=pred_norm,
        correct=correct,
        error_type=error_type,
        question=q,  # Store question text for FOL predicate extraction
    )


def evaluate_items_with_parallelism(
    items: List[Dict[str, Any]],
    client: ModelClient,
    numeric_fact_map: Optional[Dict[str, str]] = None,
    model_name: str = "unknown",
    mode: str = "zeroshot",
    max_workers: Optional[int] = None,
    checkpoint_file: Optional[str] = None,
    checkpoint_interval: int = 50,
    debug: bool = False,
    debug_samples: int = 10,
    output_dir: Optional[str] = None,
) -> List[EvalResult]:
    """
    Evaluate items with robust parallel execution, rate limiting, and retry logic.
    
    Args:
        items: List of question items
        client: Model client
        numeric_fact_map: System ID -> numeric facts mapping (for tool mode)
        model_name: Model name (for provider-specific rate limiting)
        mode: Evaluation mode (zeroshot/cot/tool)
        max_workers: Override max workers (None = use provider default)
        checkpoint_file: Path to checkpoint file for resume
        checkpoint_interval: Save checkpoint every N items
        debug: Enable debug mode (saves first N samples)
        debug_samples: Number of samples to save in debug mode
        output_dir: Output directory for debug samples
    
    Returns:
        List of EvalResult objects
    """
    numeric_fact_map = numeric_fact_map or {}
    
    # Get provider-specific rate limiting policy
    policy = get_provider_policy(model_name)
    workers = max_workers if max_workers is not None else policy["max_workers"]
    delay = policy["delay"]
    
    print(f"[CONFIG] Provider policy for '{model_name}': {workers} workers, {delay}s delay")
    
    # Load checkpoint if exists
    results: List[EvalResult] = []
    completed_ids = set()
    
    if checkpoint_file and os.path.exists(checkpoint_file):
        print(f"[RESUME] Loading checkpoint: {checkpoint_file}")
        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)
            completed_ids = set(checkpoint_data.get("completed_ids", []))
            for r in checkpoint_data.get("results", []):
                results.append(EvalResult(**r))
        print(f"[RESUME] Loaded {len(results)} previous results, skipping {len(completed_ids)} items")
    
    # Filter remaining items
    remaining_items = [item for item in items if item.get("id", "") not in completed_ids]
    
    if not remaining_items:
        print("[INFO] All items already completed!")
        return results
    
    total = len(items)
    print(f"[INFO] Processing {len(remaining_items)} items with {workers} parallel workers...")
    
    # Debug mode setup
    debug_results = []
    
    # Time tracking for ETA
    eval_start_time = time.time()
    items_processed = 0
    
    # Sequential mode (max_workers=1)
    if workers == 1:
        print("[INFO] Running in SEQUENTIAL mode (max_workers=1)")
        for idx, item in enumerate(remaining_items, 1):
            result = evaluate_single_item_robust(item, client, numeric_fact_map, delay=delay)
            results.append(result)
            items_processed += 1
            
            # Debug mode
            if debug and len(debug_results) < debug_samples:
                debug_results.append({
                    "item_id": result.item_id,
                    "question": item.get("question", ""),
                    "prompt": client.build_prompt(item.get("question", ""), {}),
                    "pred_raw": result.pred_raw,
                    "pred_norm": result.pred_norm,
                    "gold": result.gold,
                    "correct": result.correct,
                })
            
            # Progress with ETA
            completed = len(results)
            if completed % 10 == 0 or completed == total:
                elapsed = time.time() - eval_start_time
                items_per_sec = items_processed / elapsed if elapsed > 0 else 0
                remaining = len(remaining_items) - items_processed
                eta_seconds = remaining / items_per_sec if items_per_sec > 0 else 0
                eta_min = eta_seconds / 60
                
                pct = 100 * completed // total
                correct_str = "✓" if result.correct is True else "✗" if result.correct is False else "?"
                print(f"Progress: {completed}/{total} ({pct}%) - Last: {result.pred_norm} (gold: {result.gold}, {correct_str}) | Speed: {items_per_sec:.1f} items/s | ETA: {eta_min:.1f}m")
            
            # Checkpoint
            if checkpoint_file and completed % checkpoint_interval == 0:
                _save_checkpoint(checkpoint_file, results, completed_ids)
                print(f"[CHECKPOINT] Saved at {completed} items")
    
    # Parallel mode
    else:
        print(f"[INFO] Running in PARALLEL mode ({workers} workers)")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Submit all tasks
            future_to_item = {
                executor.submit(
                    evaluate_single_item_robust,
                    item,
                    client,
                    numeric_fact_map,
                    delay
                ): item
                for item in remaining_items
            }
            
            # Process completed tasks
            for future in as_completed(future_to_item):
                try:
                    result = future.result()
                    results.append(result)
                    items_processed += 1
                    item = future_to_item[future]
                    
                    # Debug mode
                    if debug and len(debug_results) < debug_samples:
                        debug_results.append({
                            "item_id": result.item_id,
                            "question": item.get("question", ""),
                            "prompt": client.build_prompt(item.get("question", ""), {}),
                            "pred_raw": result.pred_raw,
                            "pred_norm": result.pred_norm,
                            "gold": result.gold,
                            "correct": result.correct,
                        })
                    
                    # Progress with ETA
                    completed = len(results)
                    if completed % 10 == 0 or completed == total:
                        elapsed = time.time() - eval_start_time
                        items_per_sec = items_processed / elapsed if elapsed > 0 else 0
                        remaining = len(remaining_items) - items_processed
                        eta_seconds = remaining / items_per_sec if items_per_sec > 0 else 0
                        eta_min = eta_seconds / 60
                        
                        pct = 100 * completed // total
                        correct_str = "✓" if result.correct is True else "✗" if result.correct is False else "?"
                        print(f"Progress: {completed}/{total} ({pct}%) - Last: {result.pred_norm} (gold: {result.gold}, {correct_str}) | Speed: {items_per_sec:.1f} items/s | ETA: {eta_min:.1f}m")
                    
                    # Checkpoint
                    if checkpoint_file and completed % checkpoint_interval == 0:
                        _save_checkpoint(checkpoint_file, results, completed_ids)
                        print(f"[CHECKPOINT] Saved at {completed} items")
                
                except Exception as e:
                    print(f"[ERROR] Unexpected error processing item: {e}")
                    # Don't add failed items to results - they won't be counted
                    continue
    
    # Final checkpoint
    if checkpoint_file:
        _save_checkpoint(checkpoint_file, results, completed_ids)
        print(f"[CHECKPOINT] Final save at {len(results)} items")
    
    # Save debug samples
    if debug and debug_results and output_dir:
        debug_file = os.path.join(output_dir, "debug_samples.jsonl")
        with open(debug_file, 'w') as f:
            for sample in debug_results:
                f.write(json.dumps(sample) + '\n')
        print(f"[DEBUG] Saved {len(debug_results)} debug samples to {debug_file}")
    
    return results


def _save_checkpoint(checkpoint_file: str, results: List[EvalResult], completed_ids: set):
    """Save checkpoint with results and completed IDs."""
    checkpoint_data = {
        "completed_ids": list(set([r.item_id for r in results]) | completed_ids),
        "results": [asdict(r) for r in results],
        "timestamp": time.time(),
    }
    
    # Save to temp file first, then rename (atomic operation)
    temp_file = checkpoint_file + ".tmp"
    with open(temp_file, 'w') as f:
        json.dump(checkpoint_data, f)
    os.rename(temp_file, checkpoint_file)


############################
# 7. LEGACY EVALUATION (kept for compatibility)
############################

def evaluate_items(
    items: List[Dict[str, Any]],
    client: ModelClient,
    numeric_fact_map: Optional[Dict[str, str]] = None,
) -> List[EvalResult]:
    """Legacy evaluation function (sequential, no retry logic). Use evaluate_items_with_parallelism for production."""
    numeric_fact_map = numeric_fact_map or {}
    results: List[EvalResult] = []
    total = len(items)

    for idx, item in enumerate(items, 1):
        q = item["question"]
        system_id = item.get("system_id")
        ctx: Dict[str, Any] = {}

        if client.config.mode == "tool" and system_id in numeric_fact_map:
            ctx["numeric_facts"] = numeric_fact_map[system_id]

        prompt = client.build_prompt(q, context=ctx)
        pred_text = client.call(prompt)
        pred_norm = normalize_label(pred_text)
        gold_raw = (
            item.get("ground_truth")
            or item.get("gold_label")
            or item.get("answer")
            or item.get("label")
        )
        gold = normalize_label(gold_raw)

        correct: Optional[bool] = None
        if gold is not None and pred_norm is not None:
            correct = (pred_norm == gold)

        res = EvalResult(
            item_id=item.get("id", ""),
            batch_file=item.get("_batch_file", ""),
            task_family=item.get("type", "unknown"),
            bias_family=item.get("bias_family"),
            dialogue_id=item.get("dialogue_id"),
            turn_index=item.get("turn") or item.get("turn_index"),
            system_id=system_id,
            gold=gold,
            pred_raw=pred_text,
            pred_norm=pred_norm,
            correct=correct,
            error_type=None,  # No error tracking in legacy mode
            question=q,  # Store question text for FOL predicate extraction
        )
        results.append(res)

        # Progress indicator
        if idx % 10 == 0 or idx == total:
            correct_str = "✓" if correct is True else "✗" if correct is False else "?"
            print(f"Progress: {idx}/{total} ({100*idx//total}%) - Last: {pred_norm} (gold: {gold}, {correct_str})")

    return results


############################
# 8. METRICS AGGREGATION
############################

def compute_summary(results: List[EvalResult]) -> Dict[str, Any]:
    """
    Compute summary statistics with sanity checks and warnings.
    """
    summary: Dict[str, Any] = {}

    # Count different result types
    total_items = len(results)
    valid = [r for r in results if r.correct is not None]
    unanswered = [r for r in results if r.correct is None and r.gold is not None]
    no_gold = [r for r in results if r.gold is None]

    print(f"\n[SUMMARY] Result breakdown:")
    print(f"  Total items: {total_items}")
    print(f"  Valid (with correct/incorrect): {len(valid)}")
    print(f"  Unanswered (retries failed): {len(unanswered)}")
    print(f"  No gold label: {len(no_gold)}")

    # Error breakdown for unanswered items
    if unanswered:
        error_counts: Dict[str, int] = defaultdict(int)
        for r in unanswered:
            if r.error_type:
                error_counts[r.error_type] += 1
            else:
                error_counts["Unknown"] += 1

        print(f"\n[ERROR BREAKDOWN] Unanswered items by error type:")
        for error_type, count in sorted(error_counts.items(), key=lambda x: -x[1]):
            print(f"  {error_type}: {count}")
        summary["error_breakdown"] = dict(error_counts)

    # Overall accuracy
    if valid:
        num_correct = sum(1 for r in valid if r.correct is True)
        summary["overall_accuracy"] = num_correct / len(valid)
        
        # Sanity check: warn if accuracy is suspiciously low
        if summary["overall_accuracy"] == 0.0 and len(valid) > 50:
            print("\n⚠️  WARNING: Overall accuracy is 0.0% with >50 items!")
            print("  Possible issues:")
            print("    - Check normalization (see debug_samples.jsonl)")
            print("    - Many items may have failed due to rate limits")
            print("    - Model responses may not match expected format")
            print(f"    - {len(unanswered)} items had all retries fail\n")
    else:
        summary["overall_accuracy"] = None
        if total_items > 0:
            print("\n⚠️  WARNING: No valid items to compute accuracy!")

    # Accuracy by task_family
    by_task: Dict[str, List[EvalResult]] = defaultdict(list)
    for r in valid:
        by_task[r.task_family].append(r)

    task_acc: Dict[str, float] = {}
    for t, lst in by_task.items():
        n = len(lst)
        c = sum(1 for rr in lst if rr.correct is True)
        task_acc[t] = c / n if n > 0 else 0.0
    summary["task_accuracy"] = task_acc

    # Bias error rates (only items with bias_family)
    bias_items = [r for r in valid if r.bias_family is not None]
    
    if not bias_items and total_items > 0:
        print("\n⚠️  WARNING: No items contained a bias_family label; bias_error will be empty.")
    
    by_bias: Dict[str, List[EvalResult]] = defaultdict(list)
    for r in bias_items:
        assert r.bias_family is not None
        by_bias[r.bias_family].append(r)

    bias_err: Dict[str, float] = {}
    for b, lst in by_bias.items():
        n = len(lst)
        c = sum(1 for rr in lst if rr.correct is True)
        acc = c / n if n > 0 else 0.0
        bias_err[b] = 1.0 - acc
    summary["bias_error"] = bias_err

    # Dialogue metrics
    dialogues: Dict[str, List[EvalResult]] = defaultdict(list)
    for r in results:
        if r.dialogue_id is not None:
            dialogues[r.dialogue_id].append(r)

    dialogue_accs: List[float] = []
    contradiction_count = 0

    for did, turns in dialogues.items():
        turns_sorted = sorted(
            [t for t in turns if t.turn_index is not None],
            key=lambda x: x.turn_index if x.turn_index is not None else 0,
        )
        if not turns_sorted:
            continue

        # Dialogue is correct iff all turns correct and known
        all_known = all(t.correct is not None for t in turns_sorted)
        all_correct = all(t.correct is True for t in turns_sorted if t.correct is not None)
        if all_known and all_correct:
            dialogue_accs.append(1.0)
        else:
            dialogue_accs.append(0.0)

        # Naive contradiction: for any (system_id, task_family) in this dialogue,
        # model answered YES and NO at different turns.
        answers_by_key: Dict[tuple, set] = defaultdict(set)
        for t in turns_sorted:
            key = (t.system_id, t.task_family)
            if t.pred_norm is not None:
                answers_by_key[key].add(t.pred_norm)

        for ans_set in answers_by_key.values():
            if "YES" in ans_set and "NO" in ans_set:
                contradiction_count += 1
                break

    if dialogue_accs:
        summary["dialogue_accuracy"] = sum(dialogue_accs) / len(dialogue_accs)
    else:
        summary["dialogue_accuracy"] = None

    if dialogues:
        summary["contradiction_rate"] = contradiction_count / len(dialogues)
    else:
        summary["contradiction_rate"] = None

    # FOL Violation Metrics (NEW - per-dialogue violation counting)
    # Load system ontology for ground truth
    ontology = load_system_ontology(systems_dir="systems")

    # Collect all results (dialogues + single questions)
    # Treat single questions as length-1 dialogues
    all_dialogue_groups: Dict[str, List[EvalResult]] = {}

    # Multi-turn dialogues
    for did, turns in dialogues.items():
        all_dialogue_groups[did] = turns

    # Single questions (no dialogue_id) - create synthetic dialogue IDs
    single_questions = [r for r in results if r.dialogue_id is None]
    for r in single_questions:
        synthetic_id = f"single_{r.item_id}"
        all_dialogue_groups[synthetic_id] = [r]

    # Check FOL violations for each dialogue
    violation_counts: List[int] = []

    for dialogue_id, turns in all_dialogue_groups.items():
        # Group predictions by system_id
        predictions_by_system: Dict[str, Dict[str, str]] = defaultdict(dict)

        for turn in turns:
            if turn.system_id and turn.pred_norm and turn.question:
                # Extract which predicate this question asks about
                predicate = extract_predicate_from_question(turn.question)
                if predicate:
                    # Store this prediction for this system
                    predictions_by_system[turn.system_id][predicate] = turn.pred_norm

        # Check FOL violations for each system in this dialogue
        num_violations = 0
        for system_id, predictions in predictions_by_system.items():
            # Check FOL violations using the actual axioms
            violations = check_fol_violations(predictions)
            num_violations += len(violations)

        violation_counts.append(num_violations)

    # Compute violation metrics
    if violation_counts:
        summary["avg_violations_per_dialogue"] = sum(violation_counts) / len(violation_counts)

        # Breakdown by violation count
        violations_breakdown: Dict[str, int] = defaultdict(int)
        for count in violation_counts:
            if count == 0:
                violations_breakdown["0_violations"] += 1
            elif count == 1:
                violations_breakdown["1_violation"] += 1
            elif count == 2:
                violations_breakdown["2_violations"] += 1
            else:
                violations_breakdown["3+_violations"] += 1

        summary["violations_breakdown"] = dict(violations_breakdown)

        print(f"\n[FOL VIOLATIONS] Dialogue violation statistics:")
        print(f"  Avg violations per dialogue: {summary['avg_violations_per_dialogue']:.2f}")
        print(f"  Breakdown: {dict(violations_breakdown)}")
    else:
        summary["avg_violations_per_dialogue"] = None
        summary["violations_breakdown"] = {}

    return summary


############################
# 9. EXPORTS: JSONL, JSON, CSV, FIGURES
############################

def save_run_metadata(
    out_dir: str,
    model_name: str,
    mode: str,
    max_workers: int,
    results: List[EvalResult],
) -> None:
    """Save run metadata for debugging and analysis."""
    os.makedirs(out_dir, exist_ok=True)
    
    total = len(results)
    valid = len([r for r in results if r.correct is not None])
    unanswered = len([r for r in results if r.correct is None and r.gold is not None])
    no_gold = len([r for r in results if r.gold is None])
    
    metadata = {
        "model_name": model_name,
        "mode": mode,
        "max_workers": max_workers,
        "num_items_total": total,
        "num_items_evaluated": valid,
        "num_items_unanswered": unanswered,
        "num_items_no_gold": no_gold,
        "timestamp": datetime.now().isoformat(),
    }
    
    path = os.path.join(out_dir, "run_meta.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved run metadata to {path}")


def save_per_item_results(results: List[EvalResult], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "per_item_results.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(asdict(r)) + "\n")
    print(f"Saved per-item results to {path}")


def save_summary_json(summary: Dict[str, Any], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {path}")


def save_csvs(
    summary: Dict[str, Any],
    out_dir: str,
    model_name: str,
    mode: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # 1) Overview row: good for building Table 1 in the paper
    overview_path = os.path.join(out_dir, "metrics_overview.csv")
    header = [
        "model",
        "mode",
        "overall_accuracy",
        "dialogue_accuracy",
        "contradiction_rate",
    ]
    write_header = not os.path.exists(overview_path)
    with open(overview_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow([
            model_name,
            mode,
            summary.get("overall_accuracy"),
            summary.get("dialogue_accuracy"),
            summary.get("contradiction_rate"),
        ])
    print(f"Appended overview metrics to {overview_path}")

    # 2) Accuracy by task_family
    task_acc = summary.get("task_accuracy", {})
    task_path = os.path.join(out_dir, "accuracy_by_task.csv")
    with open(task_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["task_family", "accuracy"])
        for t, acc in sorted(task_acc.items()):
            writer.writerow([t, acc])
    print(f"Saved accuracy-by-task to {task_path}")

    # 3) Bias error rates
    bias_err = summary.get("bias_error", {})
    bias_path = os.path.join(out_dir, "bias_errors.csv")
    with open(bias_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["bias_family", "error_rate"])
        for b, err in sorted(bias_err.items()):
            writer.writerow([b, err])
    print(f"Saved bias error rates to {bias_path}")


def save_figures(summary: Dict[str, Any], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # Task accuracy bar plot
    task_acc = summary.get("task_accuracy", {})
    if task_acc:
        labels = list(task_acc.keys())
        values = [task_acc[k] for k in labels]

        plt.figure()
        plt.bar(labels, values)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Accuracy")
        plt.title("Accuracy by task family")
        plt.tight_layout()
        task_fig_path = os.path.join(out_dir, "task_accuracy_bar.png")
        plt.savefig(task_fig_path, dpi=300)
        plt.close()
        print(f"Saved task accuracy figure to {task_fig_path}")

    # Bias error bar plot
    bias_err = summary.get("bias_error", {})
    if bias_err:
        labels = list(bias_err.keys())
        values = [bias_err[k] for k in labels]

        plt.figure()
        plt.bar(labels, values)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Error rate")
        plt.title("Bias error rates")
        plt.tight_layout()
        bias_fig_path = os.path.join(out_dir, "bias_error_bar.png")
        plt.savefig(bias_fig_path, dpi=300)
        plt.close()
        print(f"Saved bias error figure to {bias_fig_path}")


############################
# 8. CLI
############################

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate ChaosBench-Logic on a given model.")
    ap.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name: gpt4, claude3, gemini, llama3, etc.",
    )
    ap.add_argument(
        "--mode",
        type=str,
        default="zeroshot",
        choices=["zeroshot", "cot", "tool"],
        help="Prompting mode.",
    )
    ap.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory with batch_*.jsonl files.",
    )
    ap.add_argument(
        "--batches",
        type=str,
        nargs="+",
        default=[
            "batch1_atomic_implication.jsonl",
            "batch2_multiHop_crossSystem.jsonl",
            "batch3_pde_chem_bio.jsonl",
            "batch4_maps_advanced.jsonl",
            "batch5_counterfactual_high_difficulty.jsonl",
            "batch6_deep_bias_probes.jsonl",
            "batch7_multiturn_advanced.jsonl",
        ],
        help="List of batch filenames to evaluate.",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="results",
        help="Base directory for saving results.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    config = ModelConfig(
        name=args.model,
        mode=args.mode,
    )
    client = make_model_client(config)

    batch_paths = [os.path.join(args.data_dir, b) for b in args.batches]
    items = load_batches(batch_paths)
    print(f"Loaded {len(items)} items from {len(batch_paths)} batches")

    # Optional: numeric facts per system for tool mode.
    numeric_fact_map: Dict[str, str] = {}
    # Example:
    # numeric_fact_map["lorenz63"] = "Largest Lyapunov exponent = 0.9\nAttractor: strange attractor"

    results = evaluate_items(items, client, numeric_fact_map=numeric_fact_map)
    summary = compute_summary(results)

    run_out_dir = os.path.join(args.out_dir, f"{args.model}_{args.mode}")
    os.makedirs(run_out_dir, exist_ok=True)

    save_per_item_results(results, run_out_dir)
    save_summary_json(summary, run_out_dir)
    save_csvs(summary, run_out_dir, args.model, args.mode)
    save_figures(summary, run_out_dir)

    print("=== Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
