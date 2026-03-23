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

echo "=========================================="
echo "AMR Fine-tuning: Konkani (mBART)"
echo "=========================================="
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURM_NODELIST"
echo "GPUs        : $SLURM_GPUS_ON_NODE"
echo "Start time  : $(date)"
echo ""

# ---------------------------------------------------------------------------
# 1. Modules
# ---------------------------------------------------------------------------
echo "--- Loading modules ---"
module load devel/cuda/12.8
module load devel/python/3.13.3-llvm-19.1
echo "CUDA_HOME   : $CUDA_HOME"
echo "Python      : $(which python)"
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

EXPERIMENT_DIR="$PROJECT_ROOT/experiments/finetune"
cd "$EXPERIMENT_DIR" || { echo "ERROR: cannot cd to $EXPERIMENT_DIR"; exit 1; }
echo "Working dir : $(pwd)"

mkdir -p logs results checkpoints

# ---------------------------------------------------------------------------
# 3. Hyper-parameters (edit here — keeps the sbatch header clean)
# ---------------------------------------------------------------------------
MODEL_NAME="BramVanroy/mbart-large-cc25-ft-amr30-en"
DATA_CSV="./data.csv"
TEXT_COLUMN="sentence"
AMR_COLUMN="amr_penman"
SRC_LANG="hi_IN"          # Closest mBART code to Konkani (Devanagari script)

# Training
NUM_EPOCHS=10
LEARNING_RATE=5e-5
TRAIN_BATCH=4             # per GPU
EVAL_BATCH=4
GRAD_ACCUM=4              # effective batch = TRAIN_BATCH × GRAD_ACCUM = 16
WARMUP_RATIO=0.1
WEIGHT_DECAY=0.01
EARLY_STOPPING=3          # patience (epochs)

# Generation / evaluation
NUM_BEAMS=5
MAX_NEW_TOKENS=512
MAX_SOURCE_LEN=128
MAX_TARGET_LEN=512

# Paths
OUTPUT_DIR="./results"
CHECKPOINT_DIR="./checkpoints"

SEED=42

# ---------------------------------------------------------------------------
# 4. Run fine-tuning
# ---------------------------------------------------------------------------
echo ""
echo "--- Launching fine-tuning ---"
echo "Model       : $MODEL_NAME"
echo "Data        : $DATA_CSV"
echo "Epochs      : $NUM_EPOCHS"
echo "LR          : $LEARNING_RATE"
echo "Eff. batch  : $((TRAIN_BATCH * GRAD_ACCUM))"
echo ""

uv run finetune_konkani_amr.py \
    --model_name         "$MODEL_NAME" \
    --data_csv           "$DATA_CSV" \
    --text_column        "$TEXT_COLUMN" \
    --amr_column         "$AMR_COLUMN" \
    --src_lang           "$SRC_LANG" \
    --num_train_epochs   "$NUM_EPOCHS" \
    --learning_rate      "$LEARNING_RATE" \
    --per_device_train_batch_size "$TRAIN_BATCH" \
    --per_device_eval_batch_size  "$EVAL_BATCH" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --warmup_ratio       "$WARMUP_RATIO" \
    --weight_decay       "$WEIGHT_DECAY" \
    --max_source_length  "$MAX_SOURCE_LEN" \
    --max_target_length  "$MAX_TARGET_LEN" \
    --early_stopping_patience "$EARLY_STOPPING" \
    --num_beams          "$NUM_BEAMS" \
    --max_new_tokens     "$MAX_NEW_TOKENS" \
    --output_dir         "$OUTPUT_DIR" \
    --checkpoint_dir     "$CHECKPOINT_DIR" \
    --bf16 \
    --seed               "$SEED"

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
    echo ""
    echo "Outputs:"
    echo "  Metrics      : $EXPERIMENT_DIR/$OUTPUT_DIR/metrics.json"
    echo "  Predictions  : $EXPERIMENT_DIR/$OUTPUT_DIR/test_predictions.tsv"
    echo "  Best model   : $EXPERIMENT_DIR/$OUTPUT_DIR/best_model/"
    echo "  Train split  : $EXPERIMENT_DIR/$OUTPUT_DIR/train_split.csv"
    echo "  Test split   : $EXPERIMENT_DIR/$OUTPUT_DIR/test_split.csv"
    echo ""
    echo "--- SMATCH results (from metrics.json) ---"
    cat "$OUTPUT_DIR/metrics.json" 2>/dev/null || echo "(could not print metrics)"
else
    echo "✗ Fine-tuning failed (exit code $EXIT_CODE)."
    echo "Check error logs: logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"
fi

exit $EXIT_CODE