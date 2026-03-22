#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd
import smatch
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, MBartForConditionalGeneration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple baseline: run mbart inference and print SMATCH.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="BramVanroy/mbart-large-cc25-ft-amr30-en",
        help="HF model name/path.",
    )
    parser.add_argument(
        "--data_csv",
        type=str,
        default="annotations/gemini/output_train/data.csv",
        help="CSV with sentence + gold AMR (Penman).",
    )
    parser.add_argument("--text_column", type=str, default="sentence", help="CSV column containing source text.")
    parser.add_argument("--amr_column", type=str, default="amr_penman", help="CSV column containing gold Penman AMRs.")
    parser.add_argument("--src_lang", type=str, default="en_XX", help="mbart source language code.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference.")
    parser.add_argument("--num_beams", type=int, default=5, help="Beam size for generation.")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max new tokens.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/baseline/results",
        help="Directory for predictions + metrics.",
    )
    parser.add_argument("--forced_bos_token_id", type=int, default=250181, help="AMR BOS token id from your notebook.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def setup_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_resources(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = MBartForConditionalGeneration.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    return model, tokenizer, device


def batch_translate(
    texts: List[str],
    model: MBartForConditionalGeneration,
    tokenizer: AutoTokenizer,
    src_lang: str,
    num_beams: int,
    max_new_tokens: int,
    forced_bos_token_id: int,
) -> List[str]:
    tokenizer.src_lang = src_lang
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=True,
            forced_bos_token_id=forced_bos_token_id,
        )
    return tokenizer.batch_decode(generated, skip_special_tokens=True)


def compute_smatch(golds: List[str], preds: List[str]):
    total_match = 0
    total_test = 0
    total_gold = 0
    unparsable = 0

    for gold, pred in zip(golds, preds):
        try:
            n_match, n_test, n_gold = smatch.get_amr_match(pred, gold)
            total_match += n_match
            total_test += n_test
            total_gold += n_gold
        except Exception:
            unparsable += 1

    precision, recall, f1 = smatch.compute_f(total_match, total_test, total_gold)
    return precision, recall, f1, unparsable


def main() -> None:
    args = parse_args()
    setup_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data_csv)
    if args.text_column not in df.columns or args.amr_column not in df.columns:
        raise ValueError(
            f"CSV must contain '{args.text_column}' and '{args.amr_column}'. Found: {list(df.columns)}"
        )
    df = df.dropna(subset=[args.text_column, args.amr_column]).reset_index(drop=True)

    texts = df[args.text_column].astype(str).tolist()
    refs = df[args.amr_column].astype(str).tolist()

    model, tokenizer, device = load_resources(args.model_name)
    print(f"Running on device: {device}")
    print(f"Dataset size: {len(texts)}")

    preds: List[str] = []
    for start in tqdm(range(0, len(texts), args.batch_size), desc="Inference", unit="batch"):
        batch_texts = texts[start : start + args.batch_size]
        batch_preds = batch_translate(
            texts=batch_texts,
            model=model,
            tokenizer=tokenizer,
            src_lang=args.src_lang,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            forced_bos_token_id=args.forced_bos_token_id,
        )
        preds.extend(batch_preds)

    preds_txt = out_dir / "predictions.txt"
    refs_txt = out_dir / "references.txt"
    preds_txt.write_text("\n\n".join(preds), encoding="utf-8")
    refs_txt.write_text("\n\n".join(refs), encoding="utf-8")

    precision, recall, f1, unparsable_count = compute_smatch(refs, preds)
    unparsable_pct = (100.0 * unparsable_count / len(preds)) if preds else 0.0

    metrics = {
        "Smatch Precision": float(precision),
        "Smatch Recall": float(recall),
        "Smatch Fscore": float(f1),
        "Smatch Unparsable": float(unparsable_pct),
        "Smatch Unparsable Count": int(unparsable_count),
        "Num Examples": int(len(df)),
    }

    predictions_tsv = out_dir / "predictions.tsv"
    pd.DataFrame(
        {
            "sentence": texts,
            "reference_penman": refs,
            "predicted_penman": preds,
        }
    ).to_csv(predictions_tsv, sep="\t", index=False, encoding="utf-8")

    metrics_json = out_dir / "metrics.json"
    metrics_json.write_text(
        json.dumps({"metrics": metrics}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nBaseline evaluation completed.")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nSaved metrics: {metrics_json}")
    print(f"Saved predictions: {predictions_tsv}")


if __name__ == "__main__":
    main()
