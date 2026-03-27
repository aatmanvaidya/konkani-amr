import os
import uuid

import pandas as pd

# Fixed seed for reproducibility
SEED = 42
SAMPLE_SIZE = 500

# File paths
base_path = "/home/aatman/Aatman/Study/Semantic Parsing/konkani-amr/bpcc_konkani"

wiki_path = os.path.join(base_path, "wiki/gom_Deva.tsv")
bpcc_path = os.path.join(base_path, "bpcc-seed-latest/gom_Deva.tsv")

# Global UUID set to guarantee uniqueness across BOTH files
used_ids = set()


def generate_unique_uuid():
    while True:
        uid = str(uuid.uuid4())
        if uid not in used_ids:
            used_ids.add(uid)
            return uid


def process_file(input_path, output_csv):
    # Load TSV
    df = pd.read_csv(input_path, sep="\t")

    # Ensure 'tgt' exists
    if "tgt" not in df.columns:
        raise ValueError(f"'tgt' column not found in {input_path}")

    # Drop NaNs
    df = df.dropna(subset=["tgt"])
    df["tgt"] = df["tgt"].str.strip()

    # Sample
    sampled = df.sample(n=SAMPLE_SIZE, random_state=SEED)

    # Create output dataframe
    out_df = pd.DataFrame(
        {
            "id": [generate_unique_uuid() for _ in range(len(sampled))],
            "text": sampled["tgt"].values,
        }
    )

    # Save CSV
    out_df.to_csv(output_csv, index=False)
    print(f"Saved {output_csv}")


# Run for both datasets
process_file(wiki_path, "wiki_sample.csv")
process_file(bpcc_path, "bpcc_latest_sample.csv")
