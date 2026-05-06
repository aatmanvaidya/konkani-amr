"""
Build a deduplicated Konkani monolingual corpus for MLM pre-training.

Loads all available Konkani (gom_Deva) subsets from ai4bharat/BPCC,
extracts the target-language (Konkani) column, deduplicates, and saves
a CSV with UUID-tagged sentences to data/pretraining_corpus.csv.

Dataset: https://huggingface.co/datasets/ai4bharat/BPCC
  Expected output: ~180K+ unique Konkani sentences

Usage:
    uv run scripts/build_pretraining_corpus.py
"""

import uuid

import pandas as pd
from datasets import get_dataset_config_names, load_dataset

DATASET_ID = "ai4bharat/BPCC"
LANG_CODE = "gom_Deva"
OUTPUT_CSV = "data/pretraining_corpus.csv"

used_ids: set = set()


def generate_unique_uuid() -> str:
    while True:
        uid = str(uuid.uuid4())
        if uid not in used_ids:
            used_ids.add(uid)
            return uid


def load_all_konkani_sentences() -> list[str]:
    # Discover all configs that contain the Konkani language code
    all_configs = get_dataset_config_names(DATASET_ID, trust_remote_code=True)
    konkani_configs = [c for c in all_configs if LANG_CODE in c]

    if not konkani_configs:
        raise ValueError(
            f"No configs found containing '{LANG_CODE}' in {DATASET_ID}. "
            f"Available configs: {all_configs[:10]}..."
        )

    print(f"Found {len(konkani_configs)} Konkani configs: {konkani_configs}")

    all_texts: list[str] = []

    for config in konkani_configs:
        print(f"Loading {config} ...")
        try:
            ds = load_dataset(DATASET_ID, config, split="train", trust_remote_code=True)
            df = ds.to_pandas()

            tgt_col = "tgt"
            if tgt_col not in df.columns:
                print(f"  Skipping {config}: no '{tgt_col}' column. Columns: {list(df.columns)}")
                continue

            texts = df[tgt_col].dropna().str.strip()
            texts = texts[texts != ""].tolist()
            all_texts.extend(texts)
            print(f"  Collected {len(texts):,} sentences")
        except Exception as e:
            print(f"  Failed to load {config}: {e}")

    return all_texts


all_texts = load_all_konkani_sentences()
print(f"\nTotal sentences collected: {len(all_texts):,}")

unique_texts = list(set(all_texts))
print(f"Unique sentences after dedup: {len(unique_texts):,}")

final_df = pd.DataFrame(
    {
        "id": [generate_unique_uuid() for _ in range(len(unique_texts))],
        "text": unique_texts,
    }
)

final_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved pretraining corpus → {OUTPUT_CSV}")
