import json
from collections import Counter

# Configuration
output_file = r"../output_train/amr_outputs_100.json"

# Load the JSON file
try:
    with open(output_file, "r", encoding="utf-8") as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Error: File '{output_file}' not found.")
    exit(1)
except json.JSONDecodeError:
    print(f"Error: Failed to parse JSON from '{output_file}'.")
    exit(1)

# Basic Statistics
print("=" * 60)
print("AMR OUTPUT STATISTICS")
print("=" * 60)

# Total entries
total_entries = len(data)
print(f"\n📊 Total Entries: {total_entries}")

# Success/Failure counts
successful = sum(1 for entry in data if entry.get("amr_penman") is not None)
failed = total_entries - successful
success_rate = (successful / total_entries * 100) if total_entries > 0 else 0

print(f"\n✅ Successful AMR generations: {successful} ({success_rate:.2f}%)")
print(f"❌ Failed AMR generations: {failed} ({100 - success_rate:.2f}%)")

# Translation statistics
translations_present = sum(
    1 for entry in data if entry.get("english_translation") is not None
)
print(f"\n🌐 English translations present: {translations_present}")

# Model information
models = Counter(entry.get("model", "unknown") for entry in data)
print("\n🤖 Models used:")
for model, count in models.items():
    print(f"   - {model}: {count} entries")

# Timestamp analysis
if data and data[0].get("timestamp_utc"):
    timestamps = [
        entry.get("timestamp_utc") for entry in data if entry.get("timestamp_utc")
    ]
    if timestamps:
        first_timestamp = min(timestamps)
        last_timestamp = max(timestamps)
        print("\n⏰ Time range:")
        print(f"   First entry: {first_timestamp}")
        print(f"   Last entry:  {last_timestamp}")

# Sentence length statistics
sentence_lengths = [
    len(entry.get("sentence", "")) for entry in data if entry.get("sentence")
]
if sentence_lengths:
    avg_length = sum(sentence_lengths) / len(sentence_lengths)
    min_length = min(sentence_lengths)
    max_length = max(sentence_lengths)
    print("\n📝 Sentence length (characters):")
    print(f"   Average: {avg_length:.2f}")
    print(f"   Min: {min_length}")
    print(f"   Max: {max_length}")

# AMR length statistics
amr_lengths = [
    len(entry.get("amr_penman", "")) for entry in data if entry.get("amr_penman")
]
if amr_lengths:
    avg_amr_length = sum(amr_lengths) / len(amr_lengths)
    min_amr_length = min(amr_lengths)
    max_amr_length = max(amr_lengths)
    print("\n🌳 AMR length (characters):")
    print(f"   Average: {avg_amr_length:.2f}")
    print(f"   Min: {min_amr_length}")
    print(f"   Max: {max_amr_length}")

# Sample failed entries (if any)
failed_entries = [entry for entry in data if entry.get("amr_penman") is None]
if failed_entries:
    print("\n⚠️  Sample failed entries (showing up to 3):")
    for i, entry in enumerate(failed_entries[:3], 1):
        print(f"\n   {i}. ID: {entry.get('id', 'N/A')}")
        print(f"      Sentence: {entry.get('sentence', 'N/A')[:80]}...")

# Sample successful entries
successful_entries = [entry for entry in data if entry.get("amr_penman") is not None]
if successful_entries:
    print("\n✨ Sample successful entry:")
    sample = successful_entries[0]
    print(f"\n   ID: {sample.get('id', 'N/A')}")
    print(f"   Sentence: {sample.get('sentence', 'N/A')[:100]}...")
    print(f"   Translation: {sample.get('english_translation', 'N/A')[:100]}...")
    print(f"   AMR: {sample.get('amr_penman', 'N/A')[:150]}...")

print("\n" + "=" * 60)
print("Analysis complete!")
print("=" * 60)
