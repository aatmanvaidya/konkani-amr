# Konkani AMR Parsing

Abstract Meaning Representation (AMR) parsing for Konkani, a low-resource Indian language. We fine-tune [BramVanroy/mbart-large-cc25-ft-amr30-en](https://huggingface.co/BramVanroy/mbart-large-cc25-ft-amr30-en) — a multilingual mBART model already trained on English AMR, on a Konkani AMR dataset.

## Repository Structure

```
konkani-amr/
├── data/
│   ├── annotation/               # Annotation pipeline
│   │   ├── annotate.py           # Run Gemini-2.5-Pro annotation on sampled sentences
│   │   ├── cost.py               # Calculate Gemini API cost from token logs
│   │   ├── merge_outputs.py      # Merge per-batch JSON outputs → konkani_amr.csv
│   │   ├── inspect_outputs.py    # Print statistics for a raw annotation JSON
│   │   ├── annotation_token_log.jsonl
│   │   └── raw/                  # Per-batch Gemini annotation outputs (JSON)
│   └── konkani_amr.csv           # Final annotated dataset (1100 sentences)
│
├── scripts/
│   ├── sample_annotation_corpus.py   # Sample 500 sentences from ai4bharat/BPCC
│   └── build_pretraining_corpus.py   # Build ~180K Konkani MLM pretraining corpus
│
├── experiments/
│   ├── baseline/                 # Zero-shot baseline evaluation
│   │   ├── evaluate_smatch.py    # Compute smatch scores on baseline predictions
│   │   ├── smatch_scores.csv     # Baseline smatch results
│   │   └── utils/                # Smatch utilities (backoff, postprocessing)
│   │
│   ├── finetune/                 # Direct fine-tuning on annotated data
│   │   ├── train.py              # Fine-tuning script (CLI with argparse)
│   │   ├── train.ipynb           # Notebook version
│   │   └── run.sh                # SLURM job script
│   │
│   └── pretrain_finetune/        # Konkani MLM pretraining → AMR fine-tuning
│       ├── pretrain_mlm.ipynb    # Step 1: continued MLM pretraining on Konkani
│       └── finetune_amr.ipynb    # Step 2: AMR fine-tuning on pretrained model
│
├── pyproject.toml
├── .env.example
└── README.md
```

## Methodology

### 1. Dataset Creation

1100 Konkani sentences were annotated with AMR using **Gemini-2.5-Pro**:
- 500 sentences sampled from the [ai4bharat/BPCC](https://huggingface.co/datasets/ai4bharat/BPCC) Wikipedia subset
- 500 sentences sampled from the BPCC seed corpus
- 100 sentences from a Konkani newspaper

Each sentence was prompted with detailed AMR rules (PENMAN notation) and the model returned a JSON object containing the English translation and the AMR graph. All 1,100 AMR annotations were manually verified.

### 2. Baseline

We run the base `BramVanroy/mbart-large-cc25-ft-amr30-en` model zero-shot on our 1100 Konkani sentences to establish a baseline smatch score.

### 3. Fine-tuning

We fine-tune the mBART AMR model directly on our 1100 annotated examples using an 80/5/15 train/val/test split (20 epochs, batch size 16, lr 5e-5).

### 4. Pretraining + Fine-tuning

We first continue pre-training the mBART model on ~180K Konkani sentences using a masked language modelling (token-masking) objective to adapt it to the Konkani language, then fine-tune on AMR as in step 3.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install dependencies
git clone <repo-url>
cd konkani-amr
uv sync
```

### Environment variables

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

```
GEMINI_KEY=your_gemini_api_key
HF_TOKEN=your_huggingface_token
```

## Reproducing the Dataset

```bash
# Sample sentences for annotation (creates wiki_sample.csv and bpcc_sample.csv)
uv run scripts/sample_annotation_corpus.py

# Run Gemini annotation (set CSV_FILENAME in annotate.py as needed)
cd data/annotation
uv run annotate.py

# Merge all annotation outputs into final dataset
uv run merge_outputs.py
```

## Running Experiments

### Baseline evaluation

```bash
cd experiments/baseline
uv run evaluate_smatch.py
```

### Fine-tuning

```bash
# Via CLI
uv run experiments/finetune/train.py \
    --data_csv data/konkani_amr.csv \
    --output_dir experiments/finetune/outputs \
    --epochs 20 \
    --batch_size 4 \
    --grad_accum 4 \
    --fp16
```

### Pretraining + Fine-tuning

Run the notebooks in order:
1. `experiments/pretrain_finetune/pretrain_mlm.ipynb` — MLM pretraining on Konkani
2. `experiments/pretrain_finetune/finetune_amr.ipynb` — AMR fine-tuning

Build the pretraining corpus first (requires ~180K sentence download from HuggingFace):

```bash
uv run scripts/build_pretraining_corpus.py
```

## Dataset

The annotated dataset (`data/konkani_amr.csv`) has two columns:

| Column | Description |
|---|---|
| `sentence` | Konkani sentence (Devanagari script) |
| `amr_penman` | AMR graph in PENMAN notation |
