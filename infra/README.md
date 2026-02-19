# Infrastructure Templates

This directory contains infrastructure templates for running ChaosBench-Logic at scale.

## SLURM Templates

Located in `slurm/`:

- **slurm_template.sh**: Template for SLURM job submission
- Supports job arrays for parallel model evaluation
- Configurable memory, time limits, and worker counts

## Usage

### Submit a SLURM job

```bash
# Edit the template with your parameters
vim infra/slurm/slurm_template.sh

# Submit the job
sbatch infra/slurm/slurm_template.sh
```

### Run cluster evaluation

```bash
python scripts/run_cluster_eval.py \
    --config configs/eval/gpt4_zeroshot.yaml \
    --slurm-template infra/slurm/slurm_template.sh \
    --num-shards 10
```

## Directory Structure

```
infra/
├── slurm/              # SLURM job templates
│   └── slurm_template.sh
└── README.md           # This file
```
