"""Merge raw per-batch annotation JSON files into the final dataset CSV."""
import csv
import json

files = [
    "raw/amr_outputs_100.json",
    "raw/amr_outputs_bpcc.json",
    "raw/amr_outputs_wiki.json",
]

merged = []

for file in files:
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
        merged.extend(data)

with open("../konkani_amr.csv", "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["sentence", "amr_penman"])
    for item in merged:
        sentence = item.get("sentence", "")
        amr = item.get("amr_penman", "")
        writer.writerow([sentence, amr])

print(f"Merged {len(merged)} entries → ../konkani_amr.csv")
