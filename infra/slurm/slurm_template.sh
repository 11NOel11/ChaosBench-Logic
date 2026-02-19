#!/bin/bash
#SBATCH --job-name={JOB_NAME}
#SBATCH --output={OUTPUT_DIR}/slurm_%j.out
#SBATCH --error={OUTPUT_DIR}/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1

# ChaosBench-Logic SLURM Batch Template
# Configurable via environment variables:
#   MODEL: Model name (gpt4, claude3, llama3, etc.)
#   MODE: Evaluation mode (zeroshot, cot, tool)
#   WORKERS: Number of parallel workers (default: 4)
#   SEED: Random seed (default: 42)
#   OUTPUT_DIR: Output directory for results
#   SHARD_INDEX: Shard index (0-based)
#   NUM_SHARDS: Total number of shards

# Load environment (uncomment and modify for your cluster)
# module load cuda/11.8
# module load python/3.10
# source /path/to/conda/etc/profile.d/conda.sh
# conda activate chaosbench

# Set default values if not provided
MODEL=${{MODEL:-gpt4}}
MODE=${{MODE:-zeroshot}}
WORKERS=${{WORKERS:-4}}
SEED=${{SEED:-42}}
OUTPUT_DIR=${{OUTPUT_DIR:-results}}
SHARD_INDEX=${{SHARD_INDEX:-0}}
NUM_SHARDS=${{NUM_SHARDS:-1}}

# Print configuration
echo "=========================================="
echo "ChaosBench-Logic Evaluation"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Model: $MODEL"
echo "Mode: $MODE"
echo "Workers: $WORKERS"
echo "Seed: $SEED"
echo "Output: $OUTPUT_DIR"
echo "Shard: $((SHARD_INDEX + 1))/$NUM_SHARDS"
echo "Start time: $(date)"
echo "=========================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Navigate to project root
cd $SLURM_SUBMIT_DIR || exit 1

# Run benchmark
python run_benchmark.py \
    --model "$MODEL" \
    --mode "$MODE" \
    --workers "$WORKERS" \
    --out-dir "$OUTPUT_DIR" \
    --num-shards "$NUM_SHARDS" \
    --shard-index "$SHARD_INDEX"

EXIT_CODE=$?

# Print completion info
echo "=========================================="
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE
