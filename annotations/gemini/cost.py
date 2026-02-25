import json
from pathlib import Path

INPUT_PRICE_PER_1M = 1.25
OUTPUT_PRICE_PER_1M = 10.00

TOKEN_LOG_FILE = "gemini_token_log.jsonl"


def calculate_cost(jsonl_path: str):
    total_prompt_tokens = 0
    total_output_tokens = 0
    num_requests = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)

            total_prompt_tokens += record.get("prompt_tokens", 0)
            total_output_tokens += record.get("output_tokens", 0)
            num_requests += 1

    # ---- cost calculation ----
    input_cost = (total_prompt_tokens / 1_000_000) * INPUT_PRICE_PER_1M
    output_cost = (total_output_tokens / 1_000_000) * OUTPUT_PRICE_PER_1M
    total_cost = input_cost + output_cost

    # ---- pretty report ----
    print("\n📊 Gemini Cost Report")
    print("=" * 40)
    print(f"Requests processed : {num_requests}")
    print(f"Prompt tokens      : {total_prompt_tokens:,}")
    print(f"Output tokens      : {total_output_tokens:,}")
    print("-" * 40)
    print(f"Input cost         : ${input_cost:.6f}")
    print(f"Output cost        : ${output_cost:.6f}")
    print("=" * 40)
    print(f"💰 TOTAL COST       : ${total_cost:.6f}\n")

    return {
        "requests": num_requests,
        "prompt_tokens": total_prompt_tokens,
        "output_tokens": total_output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
    }


if __name__ == "__main__":
    if not Path(TOKEN_LOG_FILE).exists():
        print(f"❌ File not found: {TOKEN_LOG_FILE}")
    else:
        calculate_cost(TOKEN_LOG_FILE)
