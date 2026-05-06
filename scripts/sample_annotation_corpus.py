"""
Sample Konkani sentences from ai4bharat/BPCC for AMR annotation.

Produces two CSV files (wiki_sample.csv, bpcc_sample.csv) each with
500 randomly-sampled Konkani sentences and unique IDs, ready to feed
into data/annotation/annotate.py.

Dataset: https://huggingface.co/datasets/ai4bharat/BPCC
  - wiki subset  : Wikipedia-sourced Konkani–English parallel sentences
  - bpcc subset  : BPCC seed corpus Konkani–English parallel sentences

Usage:
    uv run scripts/sample_annotation_corpus.py
"""

import uuid

import pandas as pd
from datasets import load_dataset

SEED = 42
SAMPLE_SIZE = 500
DATASET_ID = "ai4bharat/BPCC"

# Config names on HuggingFace for Konkani Devanagari (gom_Deva).
# Verify the exact names on https://huggingface.co/datasets/ai4bharat/BPCC
WIKI_CONFIG = "wiki_gom_Deva"
BPCC_CONFIG = "bpcc_seed_gom_Deva"

used_ids: set = set()


def generate_unique_uuid() -> str:
    while True:
        uid = str(uuid.uuid4())
        if uid not in used_ids:
            used_ids.add(uid)
            return uid


def sample_from_hf(config: str, output_csv: str) -> None:
    print(f"Loading {DATASET_ID} / {config} ...")
    ds = load_dataset(DATASET_ID, config, split="train", trust_remote_code=True)
    df = ds.to_pandas()

    # BPCC stores source in 'src' and target in 'tgt'
    tgt_col = "tgt"
    if tgt_col not in df.columns:
        raise ValueError(
            f"Column '{tgt_col}' not found in config '{config}'. "
            f"Available columns: {list(df.columns)}"
        )

    df = df.dropna(subset=[tgt_col])
    df[tgt_col] = df[tgt_col].str.strip()
    df = df[df[tgt_col] != ""].reset_index(drop=True)

    sampled = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=SEED)

    out_df = pd.DataFrame(
        {
            "id": [generate_unique_uuid() for _ in range(len(sampled))],
            "text": sampled[tgt_col].values,
        }
    )

    out_df.to_csv(output_csv, index=False)
    print(f"Saved {output_csv} ({len(out_df)} rows)")


sample_from_hf(WIKI_CONFIG, "wiki_sample.csv")
sample_from_hf(BPCC_CONFIG, "bpcc_sample.csv")
