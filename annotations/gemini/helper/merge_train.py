import json
import csv

# Your JSON files
files = [r"../output_train/amr_outputs_100.json", r"../output_train/amr_outputs_bpcc_latest_sample.json", r"../output_train/amr_outputs_wiki_sample.json"]

merged = []

# Load and merge
for file in files:
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
        merged.extend(data)

# Write CSV
with open(r"../output_train/data.csv", "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    
    # header
    writer.writerow(["sentence", "amr_penman"])
    
    # rows
    for item in merged:
        sentence = item.get("sentence", "")
        amr = item.get("amr_penman", "")
        writer.writerow([sentence, amr])

print("CSV created: output.csv")