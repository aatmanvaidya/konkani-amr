import json
import os
from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv
from google import genai
from tqdm import tqdm

load_dotenv()
client = genai.Client(api_key=os.getenv("KIRAN_GEMINI_KEY"))

MODEL = "gemini-2.5-pro"
CSV_FILENAME = "wiki_sample"
token_log_file = "gemini_token_log.jsonl"
error_log_file = "gemini_error_log.jsonl"
output_file = rf"output_train/amr_outputs_{CSV_FILENAME}.json"
input_csv_file = rf"/home/aatman/Aatman/Study/Semantic Parsing/konkani-amr/training_data/{CSV_FILENAME}.csv"

amr_rules = """
AMR RULES (PENMAN notation):

STRUCTURE:
- Rooted, directed acyclic graph: (variable / concept :role value ...)
- Each node gets a unique short variable (b, b2, g, etc.)
- Co-reference: define a node once as (x / concept), then reuse x bare

CONCEPTS:
- Use PropBank verb frames with sense numbers: go-02, want-01, believe-01
- Nouns, adjectives, and verbs all become concepts (no POS distinction)
- Named entities: (p / person :name (n / name :op1 "John")) with :wiki if notable

CORE ROLES (ARGx are predicate-specific per PropBank):
- :ARG0 = agent/doer, :ARG1 = patient/theme, :ARG2 = beneficiary/instrument/etc.
- Check PropBank frames for exact ARG meanings per predicate

NON-CORE ROLES:
:location, :time, :manner, :purpose, :cause, :condition, :concession,
:beneficiary, :instrument, :duration, :frequency, :degree,
:source, :destination, :path, :topic, :medium

MODIFIERS & SPECIAL:
- Negation: :polarity -
- Modality: (o / obligate-01 :ARG2 ...) or (p / permit-01 ...)
- Questions: :polarity ? or (a / amr-unknown) for wh-questions
- Imperatives: :mode imperative
- Possession: :poss
- Part-whole: :part-of
- Degree: :degree (m / more), :degree (s / superlative)

CONJUNCTIONS & LISTS:
- (a / and :op1 ... :op2 ...) — also or, contrast-01, etc.

QUANTITIES & ENTITIES:
- Numbers: (x / concept :quant 3)
- Dates: (d / date-entity :year 2024 :month 3 :day 5)
- Percentages: (p / percentage-entity :value 25)

KEY ABSTRACTIONS (ignore English surface form):
- Drop articles, tense, aspect, passive voice
- "The girl adjusted the machine" = "The girl made adjustments to the machine" → same AMR
- Inverse roles: use :ARG0-of, :ARG1-of for relative clauses and nominalizations
  e.g. (d / doctor :ARG0-of (h / heal-01)) = "the healing doctor"
"""

AMR_PROMPT_TEMPLATE = """
You are an expert linguist specializing in Abstract Meaning Representation (AMR).

{amr_rules}

Task: Given a Konkani sentence, do the following:
1. Provide an accurate English translation.
2. Produce the AMR in valid PENMAN notation.

Output format (STRICT JSON ONLY — no markdown, no explanation):
{{"english_translation": "...","amr_penman": "..."}}

Rules:
- Output ONLY valid JSON, no markdown fences (Do NOT use ```json)
- AMR must be valid PENMAN using the rules above

Sentence: "{sentence}"
"""


def clean_model_output(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def log_error(sentence_id: str, sentence: str, error_type: str, error_msg: str):
    """Log errors to a separate error log file"""
    error_entry = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "sentence_id": sentence_id,
        "sentence": sentence,
        "error_type": error_type,
        "error_message": error_msg,
    }
    with open(error_log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(error_entry, ensure_ascii=False) + "\n")


# Load existing output if it exists
if os.path.exists(output_file):
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
            if not isinstance(existing_data, list):
                existing_data = []
    except json.JSONDecodeError:
        existing_data = []
else:
    existing_data = []

# Create a set of already processed IDs for fast lookup
processed_ids = {entry.get("id") for entry in existing_data if entry.get("id")}
print(f"Found {len(processed_ids)} already processed entries")

# Read CSV file
df = pd.read_csv(input_csv_file)

# Count how many will be skipped
total_rows = len(df)
rows_to_skip = sum(1 for _, row in df.iterrows() if row["id"] in processed_ids)
rows_to_process = total_rows - rows_to_skip

print(f"Total entries in CSV: {total_rows}")
print(f"Already processed: {rows_to_skip}")
print(f"To be processed: {rows_to_process}")

# Process each row
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing Sentences"):
    sentence_id = row["id"]
    sentence = row["text"]

    # Skip if already processed
    if sentence_id in processed_ids:
        continue

    try:
        prompt = AMR_PROMPT_TEMPLATE.format(amr_rules=amr_rules, sentence=sentence)
        response = client.models.generate_content(
            model=MODEL,
            contents=[prompt],
            config={"temperature": 0},
        )

        # Check if response has text attribute
        if not hasattr(response, "text") or response.text is None:
            error_msg = f"API returned None response. Response object: {response}"
            print(f"\n❌ Error for sentence ID {sentence_id}")
            print(f"   Sentence: {sentence[:100]}...")
            print(f"   Error: {error_msg}")

            # Check for candidates and finish_reason
            if hasattr(response, "candidates") and response.candidates:
                finish_reason = (
                    response.candidates[0].finish_reason
                    if response.candidates
                    else "unknown"
                )
                print(f"   Finish reason: {finish_reason}")
                error_msg += f" | Finish reason: {finish_reason}"

            log_error(sentence_id, sentence, "API_RESPONSE_NONE", error_msg)

            # Save entry with None values
            output_entry = {
                "id": sentence_id,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "model": MODEL,
                "sentence": sentence,
                "english_translation": None,
                "amr_penman": None,
                "error": "API returned None response",
            }
            existing_data.append(output_entry)
            processed_ids.add(sentence_id)

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=2)

            continue

        response_text = response.text.strip()
        cleaned_text = clean_model_output(response_text)

        try:
            parsed = json.loads(cleaned_text)
            english_translation = parsed.get("english_translation")
            amr_penman = parsed.get("amr_penman")
        except json.JSONDecodeError as e:
            print(
                f"\n⚠️  JSON parse error for sentence ID {sentence_id}: {sentence[:100]}..."
            )
            print(f"   Error: {str(e)}")
            print(f"   Response text: {cleaned_text[:200]}...")
            log_error(
                sentence_id,
                sentence,
                "JSON_PARSE_ERROR",
                f"{str(e)} | Response: {cleaned_text[:500]}",
            )
            english_translation = None
            amr_penman = None

        output_entry = {
            "id": sentence_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model": MODEL,
            "sentence": sentence,
            "english_translation": english_translation,
            "amr_penman": amr_penman,
        }

        existing_data.append(output_entry)
        processed_ids.add(sentence_id)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)

        # Log token usage if available
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = response.usage_metadata
            timestamp_utc = datetime.now(timezone.utc).isoformat()

            token_log_entry = {
                "timestamp_utc": timestamp_utc,
                "file_name": input_csv_file,
                "model": MODEL,
                "sentence_id": sentence_id,
                "prompt_tokens": usage.prompt_token_count,
                "output_tokens": usage.candidates_token_count,
                "total_tokens": usage.total_token_count,
                "prompt_tokens_text": next(
                    (
                        d.token_count
                        for d in usage.prompt_tokens_details
                        if d.modality.name == "TEXT"
                    ),
                    None,
                ),
                "prompt_tokens_image": next(
                    (
                        d.token_count
                        for d in usage.prompt_tokens_details
                        if d.modality.name == "IMAGE"
                    ),
                    None,
                ),
            }

            with open(token_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(token_log_entry) + "\n")

    except Exception as e:
        # Catch any other unexpected errors
        error_msg = f"Unexpected error: {type(e).__name__}: {str(e)}"
        print(f"\n❌ Unexpected error for sentence ID {sentence_id}")
        print(f"   Sentence: {sentence[:100]}...")
        print(f"   Error: {error_msg}")
        log_error(sentence_id, sentence, "UNEXPECTED_ERROR", error_msg)

        # Save entry with None values and error info
        output_entry = {
            "id": sentence_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model": MODEL,
            "sentence": sentence,
            "english_translation": None,
            "amr_penman": None,
            "error": error_msg,
        }
        existing_data.append(output_entry)
        processed_ids.add(sentence_id)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)

print("\nProcessing complete")
print(f"Total entries in output file: {len(existing_data)}")
print(f"Check '{error_log_file}' for any errors that occurred")
