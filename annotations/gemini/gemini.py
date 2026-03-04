import json
import os
from datetime import datetime, timezone
from tqdm import tqdm

from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("KIRAN_GEMINI_KEY"))

MODEL = "gemini-2.5-pro"
token_log_file = "gemini_token_log.jsonl"
output_file = "amr_outputs_100.json"
input_txt_file = r"data/clean_konkani_sentences.txt"

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

with open(input_txt_file, "r", encoding="utf-8") as f:
    sentences = [line.strip() for line in f if line.strip()]

for sentence in tqdm(sentences, desc="Proc Sentences"):
    prompt = AMR_PROMPT_TEMPLATE.format(amr_rules=amr_rules, sentence=sentence)
    response = client.models.generate_content(
        model=MODEL,
        contents=[prompt],
        config={"temperature": 0},
    )
    response_text = response.text.strip()
    cleaned_text = clean_model_output(response_text)
    try:
        parsed = json.loads(cleaned_text)
        english_translation = parsed.get("english_translation")
        amr_penman = parsed.get("amr_penman")
    except json.JSONDecodeError:
        print("Failed to parse JSON for sentence:", sentence)
        english_translation = None
        amr_penman = None

    output_entry = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "sentence": sentence,
        "english_translation": english_translation,
        "amr_penman": amr_penman,
    }

    existing_data.append(output_entry)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=2)

    usage = response.usage_metadata
    timestamp_utc = datetime.now(timezone.utc).isoformat()

    token_log_entry = {
        "timestamp_utc": timestamp_utc,
        "file_name": input_txt_file,
        "model": MODEL,
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

print("Processing complete")
