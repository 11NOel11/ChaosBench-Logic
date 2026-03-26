# Cluster Templates

This directory contains cluster templates for running ChaosBench-Logic at scale.

## SLURM Templates

Located in this directory:

- **slurm_template.sh**: Template for SLURM job submission
- Supports job arrays for parallel model evaluation
- Configurable memory, time limits, and worker counts

## Usage

### Submit a SLURM job

```bash
# Edit the template with your parameters
vim scripts/cluster/slurm_template.sh

# Submit the job
sbatch scripts/cluster/slurm_template.sh
```

### Run cluster evaluation

```bash
python scripts/run_cluster_eval.py \
    --models gpt4 claude3 \
    --modes zeroshot cot \
    --base-dir runs/cluster \
    --num-shards 10 \
    --submit
```

## Directory Structure

```
scripts/cluster/
├── slurm_template.sh
└── README.md
```
