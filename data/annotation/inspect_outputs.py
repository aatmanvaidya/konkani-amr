"""Print statistics for a raw annotation JSON output file."""
import json
from collections import Counter

output_file = "raw/amr_outputs_100.json"

try:
    with open(output_file, "r", encoding="utf-8") as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Error: File '{output_file}' not found.")
    exit(1)
except json.JSONDecodeError:
    print(f"Error: Failed to parse JSON from '{output_file}'.")
    exit(1)

print("=" * 60)
print("AMR OUTPUT STATISTICS")
print("=" * 60)

total_entries = len(data)
print(f"\nTotal Entries: {total_entries}")

successful = sum(1 for entry in data if entry.get("amr_penman") is not None)
failed = total_entries - successful
success_rate = (successful / total_entries * 100) if total_entries > 0 else 0

print(f"Successful AMR generations: {successful} ({success_rate:.2f}%)")
print(f"Failed AMR generations: {failed} ({100 - success_rate:.2f}%)")

translations_present = sum(
    1 for entry in data if entry.get("english_translation") is not None
)
print(f"English translations present: {translations_present}")

models = Counter(entry.get("model", "unknown") for entry in data)
print("\nModels used:")
for model, count in models.items():
    print(f"   - {model}: {count} entries")

if data and data[0].get("timestamp_utc"):
    timestamps = [entry.get("timestamp_utc") for entry in data if entry.get("timestamp_utc")]
    if timestamps:
        print(f"\nTime range: {min(timestamps)} → {max(timestamps)}")

sentence_lengths = [len(entry.get("sentence", "")) for entry in data if entry.get("sentence")]
if sentence_lengths:
    print(f"\nSentence length (chars): avg={sum(sentence_lengths)/len(sentence_lengths):.1f}, min={min(sentence_lengths)}, max={max(sentence_lengths)}")

amr_lengths = [len(entry.get("amr_penman", "")) for entry in data if entry.get("amr_penman")]
if amr_lengths:
    print(f"AMR length (chars): avg={sum(amr_lengths)/len(amr_lengths):.1f}, min={min(amr_lengths)}, max={max(amr_lengths)}")

failed_entries = [entry for entry in data if entry.get("amr_penman") is None]
if failed_entries:
    print(f"\nFailed entries (showing up to 3):")
    for i, entry in enumerate(failed_entries[:3], 1):
        print(f"   {i}. ID: {entry.get('id', 'N/A')} | {entry.get('sentence', 'N/A')[:80]}...")

successful_entries = [entry for entry in data if entry.get("amr_penman") is not None]
if successful_entries:
    sample = successful_entries[0]
    print(f"\nSample successful entry:")
    print(f"   Sentence: {sample.get('sentence', 'N/A')[:100]}...")
    print(f"   Translation: {sample.get('english_translation', 'N/A')[:100]}...")
    print(f"   AMR: {sample.get('amr_penman', 'N/A')[:150]}...")

print("\n" + "=" * 60)
