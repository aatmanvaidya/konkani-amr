#!/bin/bash
#SBATCH --job-name=AMR_Finetune_Konkani
#SBATCH --partition=gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aatman-vrundavan.vaidya@student.uni-tuebingen.de

module load devel/cuda/12.8
module load devel/python/3.13.3-llvm-19.1
echo "CUDA device(s):"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# ---------------------------------------------------------------------------
# 2. Project environment
# ---------------------------------------------------------------------------
echo "--- Setting up project environment ---"
PROJECT_ROOT=/home/tu/tu_tu/tu_zxord71/konkani-amr

if [ -d "$PROJECT_ROOT/.venv" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
else
    echo "Warning: .venv not found at $PROJECT_ROOT/.venv, trying local .venv"
    source .venv/bin/activate
fi

EXPERIMENT_DIR="$PROJECT_ROOT/experiments/mbart-large-cc25-ft-amr30-en/finetune"
cd "$EXPERIMENT_DIR" || { echo "ERROR: cannot cd to $EXPERIMENT_DIR"; exit 1; }
echo "Working dir : $(pwd)"

mkdir -p logs konkani_amr_finetuned

uv run finetune_konkani_amr.py \
    --data_csv ./data.csv \
    --output_dir ./konkani_amr_finetuned \
    --epochs 15 \
    --batch_size 4 \
    --grad_accum 4 \
    --fp16

EXIT_CODE=$?

# ---------------------------------------------------------------------------
# 5. Summary
# ---------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "Fine-tuning job finished"
echo "=========================================="
echo "Exit code  : $EXIT_CODE"
echo "End time   : $(date)"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Fine-tuning completed successfully."

else
    echo "✗ Fine-tuning failed (exit code $EXIT_CODE)."
    echo "Check error logs: logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"
fi

exit $EXIT_CODE