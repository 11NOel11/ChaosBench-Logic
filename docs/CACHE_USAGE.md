# Response Cache Usage

The response cache stores model responses to avoid redundant API calls during evaluation runs. This is especially useful for:
- Re-running evaluations with the same model/mode
- Debugging specific test cases without making new API calls
- Reducing API costs and rate limit issues

## Basic Usage

### Command Line

Enable caching by providing the `--cache-dir` argument:

```bash
# Basic usage with cache
python run_benchmark.py --model gpt4 --mode zeroshot --cache-dir ./cache

# Subsequent runs will use cached responses
python run_benchmark.py --model gpt4 --mode zeroshot --cache-dir ./cache
```

### Python API

```python
from chaosbench.eval.cache import ResponseCache
from chaosbench.eval.runner import evaluate_items_with_parallelism
from chaosbench.models.prompt import ModelConfig, make_model_client

# Create cache
cache = ResponseCache("./cache")

# Use with evaluation
config = ModelConfig(name="gpt4", mode="zeroshot")
client = make_model_client(config)

results = evaluate_items_with_parallelism(
    items=items,
    client=client,
    model_name="gpt4",
    mode="zeroshot",
    cache=cache,
)

# Always close when done
cache.close()
```

### Context Manager

```python
from chaosbench.eval.cache import ResponseCache

# Automatically closes when done
with ResponseCache("./cache") as cache:
    cache.put("gpt4", "zeroshot", "item_001", "Question?", "Answer")
    response = cache.get("gpt4", "zeroshot", "item_001", "Question?")
```

## Cache Operations

### Get Statistics

```python
cache = ResponseCache("./cache")
stats = cache.stats()
print(f"Total entries: {stats['total_entries']}")
print(f"Unique models: {stats['models']}")
cache.close()
```

### Invalidate Cache Entries

```python
cache = ResponseCache("./cache")

# Invalidate all entries for a model+mode
deleted = cache.invalidate("gpt4", "zeroshot")
print(f"Deleted {deleted} entries")

# Invalidate specific item
deleted = cache.invalidate("gpt4", "zeroshot", "item_001")

cache.close()
```

## Cache Behavior

### Cache Key

Responses are cached by:
- `model`: Model name (e.g., "gpt4", "claude3")
- `mode`: Evaluation mode (e.g., "zeroshot", "cot")
- `item_id`: Question item identifier
- `question_sha256`: SHA-256 hash of the question text

Different questions with the same item_id are cached separately.

### Cache Hit Strategy

1. On cache hit: Returns cached response immediately (no API call)
2. On cache miss: Calls model API and caches the successful response
3. On error: Does not cache the error response

### Thread Safety

The cache uses SQLite with `check_same_thread=False` for thread-safe access during parallel evaluation.

## Storage

- Cache database: `{cache_dir}/response_cache.db`
- Format: SQLite database
- Schema: `responses(model, mode, item_id, question_sha256, response, timestamp)`

## Best Practices

1. Use a persistent cache directory (not `/tmp`) to preserve cache across system restarts
2. Separate cache directories for different evaluation campaigns
3. Regularly check cache statistics to monitor growth
4. Invalidate cache when updating prompts or system behavior
5. Back up cache database before major changes

## Example: Development Workflow

```bash
# First run - populates cache
python run_benchmark.py --model gpt4 --mode zeroshot --cache-dir ./gpt4_cache --max-items 100

# Debug run - uses cache for completed items
python run_benchmark.py --model gpt4 --mode zeroshot --cache-dir ./gpt4_cache --max-items 100

# Different mode - separate cache entries
python run_benchmark.py --model gpt4 --mode cot --cache-dir ./gpt4_cache --max-items 100
```

## Troubleshooting

### Cache not being used

- Verify `--cache-dir` is provided
- Check that model name and mode match exactly
- Ensure question text hasn't changed (uses SHA-256 hash)

### Cache growing too large

```python
# Check size
cache = ResponseCache("./cache")
stats = cache.stats()
print(f"Total entries: {stats['total_entries']}")

# Clear specific model
cache.invalidate("gpt4", "zeroshot")

# Or delete the database file manually
cache.close()
```

### Permission errors

- Ensure cache directory is writable
- Check that SQLite database file isn't locked by another process
