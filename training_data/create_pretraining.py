import os
import uuid
from glob import glob

import pandas as pd

base_path = "/home/aatman/Aatman/Study/Semantic Parsing/konkani-amr/bpcc_konkani"

# Find all TSV files recursively
tsv_files = glob(os.path.join(base_path, "**/*.tsv"), recursive=True)

print(f"Found {len(tsv_files)} TSV files")

all_texts = []
used_ids = set()


def generate_unique_uuid():
    while True:
        uid = str(uuid.uuid4())
        if uid not in used_ids:
            used_ids.add(uid)
            return uid


# 🔄 Load and merge
for file in tsv_files:
    print(f"Processing: {file}")

    df = pd.read_csv(file, sep="\t")

    if "tgt" not in df.columns:
        print(f"Skipping (no tgt column): {file}")
        continue

    # Clean text
    text = df["tgt"].dropna()
    text = text.str.strip()
    # text = text.str.replace(r"\s+", " ", regex=True)
    text = text[text != ""]

    all_texts.extend(text.tolist())

print(f"\nTotal sentences collected: {len(all_texts)}")

# 🧹 Optional: remove duplicates
unique_texts = list(set(all_texts))
print(f"Unique sentences: {len(unique_texts)}")

# 🆔 Create final dataframe
final_df = pd.DataFrame(
    {
        "id": [generate_unique_uuid() for _ in range(len(unique_texts))],
        "text": unique_texts,
    }
)

# 💾 Save
output_path = "merged_konkani.csv"
final_df.to_csv(output_path, index=False)

print(f"\nSaved merged dataset to: {output_path}")
