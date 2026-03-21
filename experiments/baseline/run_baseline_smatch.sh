#!/bin/bash
#SBATCH --job-name=AMR_Baseline_Smatch
#SBATCH --partition=gpu_a100_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=0:29:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=aatman-vrundavan.vaidya@student.uni-tuebingen.de

echo "=========================================="
echo "AMR Baseline SMATCH Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# 1. Load modules (adapt if your cluster uses different module names)
echo "Loading modules..."
module load devel/cuda/12.8
module load devel/python/3.13.3-llvm-19.1
echo "CUDA Home: $CUDA_HOME"
echo "Python: $(which python)"
echo ""

# 2. Project Setup
echo "Setting up project environment..."
PROJECT_ROOT=/home/tu/tu_tu/tu_zxord71/konkani-amr
if [ -d "$PROJECT_ROOT/.venv" ]; then
    source $PROJECT_ROOT/.venv/bin/activate
else
    echo "Warning: .venv not found at $PROJECT_ROOT/.venv, trying local .venv"
    source .venv/bin/activate
fi

# Navigate to the script directory
cd $PROJECT_ROOT/experiments/transfer_learning/finetune || exit 1
echo "Working directory: $(pwd)"

# Create logs directory if it doesn't exist
mkdir -p logs

# 4. Run baseline evaluation
python "experiments/baseline/calculate_baseline_smatch.py" \
    --model_name "BramVanroy/mbart-large-cc25-ft-amr30-en" \
    --data_csv "annotations/gemini/output_train/data.csv" \
    --text_column "sentence" \
    --amr_column "amr_penman" \
    --src_lang "en_XX" \
    --batch_size 4 \
    --num_beams 5 \
    --max_new_tokens 900 \
    --output_dir "experiments/baseline/results"

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "AMR baseline job completed"
echo "=========================================="
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Baseline evaluation completed successfully."
    echo "Results: experiments/baseline/results"
else
    echo "✗ Baseline evaluation failed with exit code $EXIT_CODE"
    echo "Check error logs: logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"
fi

exit $EXIT_CODE
