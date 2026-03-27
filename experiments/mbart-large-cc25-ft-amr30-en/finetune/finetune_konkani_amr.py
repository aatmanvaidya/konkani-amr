"""
Fine-tune BramVanroy/mbart-large-cc25-ft-amr30-en on Konkani AMR data.

Pipeline:
  1. Load data.csv  (columns: sentence, amr_penman)
  2. Split 80 / 5 / 15  → train / val / test
  3. Fine-tune with Hugging Face Trainer
  4. Run inference on the test set
  5. Save test predictions → test_predictions.json
  6. Compute smatch scores → smatch_scores_test.csv

Usage:
  python finetune_konkani_amr.py [--data_csv PATH] [--output_dir PATH] [--epochs N]
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from typing import List, Optional

import pandas as pd
import penman
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    DataCollatorForSeq2Seq,
    MBart50TokenizerFast,
    MBartForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.utils import logging as hf_logging

warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
MODEL_NAME = "BramVanroy/mbart-large-cc25-ft-amr30-en"
SRC_LANG = "hi_IN"  # Closest mBART language code for Konkani
TGT_LANG = "en_XX"
MAX_SRC_LEN = 128
MAX_TGT_LEN = 512
SEED = 42


# ─────────────────────────────────────────────
# AMR helpers  (identical to your inference script)
# ─────────────────────────────────────────────


def linearized_to_penman(linearized: str, graph_idx: int = 0) -> str:
    """Convert linearized MBart AMR output → Penman string."""
    text = linearized.replace("<AMR>", "").strip()
    pointer_map: dict = {}
    var_idx = 0

    def replace_pointer(match):
        nonlocal var_idx
        idx = match.group(1)
        if idx not in pointer_map:
            pointer_map[idx] = f"x{graph_idx}_{var_idx}"
            var_idx += 1
        return f"({pointer_map[idx]} / "

    text = re.sub(r"<pointer:(\d+)>", replace_pointer, text)
    text = text.replace("<rel>", " ").replace("</rel>", ")")
    text = re.sub(r"\s+", " ", text).strip()

    open_par = text.count("(")
    close_par = text.count(")")
    text += ")" * max(0, open_par - close_par)
    return text


def clean_pred_penman(text: str) -> str:
    """Clean up common artefacts in predicted Penman strings."""
    text = re.sub(
        r"<lit>\s*(.*?)\s*</lit>",
        lambda m: '"' + m.group(1).strip() + '"',
        text,
        flags=re.DOTALL,
    )
    text = re.sub(r"^thing\(", "(", text.strip())
    text = re.sub(r"\(x\d+_\d+\s*/\s*\)", "(amr-unknown)", text)
    text = re.sub(
        r"(:(?:ARG\d+|op\d+|mod|poss|quant|domain|time|location|manner|"
        r"cause|degree|purpose|condition|wiki|name|polarity|mode|li|value|snt\d+))"
        r"(x\d+_\d+)",
        r"\1 \2",
        text,
    )
    return text


def safe_encode(amr_str: str) -> Optional[str]:
    try:
        return penman.encode(penman.decode(amr_str.strip()))
    except Exception:
        return None


def smatch_score(gold_str: str, pred_str: str):
    """Return (P, R, F1) or None on failure."""
    gold_norm = safe_encode(gold_str)
    pred_norm = safe_encode(pred_str)
    if gold_norm is None or pred_norm is None:
        return None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".amr", delete=False, encoding="utf-8"
        ) as gf:
            gf.write(gold_norm + "\n\n")
            gold_path = gf.name
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".amr", delete=False, encoding="utf-8"
        ) as pf:
            pf.write(pred_norm + "\n\n")
            pred_path = pf.name

        result = subprocess.run(
            [sys.executable, "-m", "smatch", "--pr", "-f", pred_path, gold_path],
            capture_output=True,
            text=True,
            timeout=15,
        )
        os.unlink(gold_path)
        os.unlink(pred_path)

        p = r = f = None
        for line in result.stdout.strip().split("\n"):
            if "Precision" in line:
                p = float(line.split()[-1])
            elif "Recall" in line:
                r = float(line.split()[-1])
            elif "F-score" in line:
                f = float(line.split()[-1])
        if f is not None:
            return round(p, 4), round(r, 4), round(f, 4)
        return None
    except Exception:
        return None


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────


class KonkaniAMRDataset(Dataset):
    def __init__(
        self,
        sentences: List[str],
        amr_strings: List[str],
        tokenizer: MBart50TokenizerFast,
    ):
        self.sentences = sentences
        self.amr_strings = amr_strings
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        src = self.sentences[idx]
        tgt = self.amr_strings[idx]

        self.tokenizer.src_lang = SRC_LANG
        model_inputs = self.tokenizer(
            src,
            max_length=MAX_SRC_LEN,
            truncation=True,
            padding=False,
        )

        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                tgt,
                max_length=MAX_TGT_LEN,
                truncation=True,
                padding=False,
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


# ─────────────────────────────────────────────
# Data split
# ─────────────────────────────────────────────


def split_data(df: pd.DataFrame, train_frac=0.80, val_frac=0.05, seed=SEED):
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = df.iloc[:n_train]
    val = df.iloc[n_train : n_train + n_val]
    test = df.iloc[n_train + n_val :]
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


# ─────────────────────────────────────────────
# Inference on a list of sentences
# ─────────────────────────────────────────────


def run_inference(
    sentences: List[str],
    gold_amrs: List[str],
    model: MBartForConditionalGeneration,
    tokenizer: MBart50TokenizerFast,
    device: str,
    batch_size: int = 8,
) -> List[dict]:
    model.eval()
    results = []
    forced_bos = tokenizer.lang_code_to_id[TGT_LANG]

    for start in tqdm(range(0, len(sentences), batch_size), desc="Inference"):
        batch_sents = sentences[start : start + batch_size]
        batch_golds = gold_amrs[start : start + batch_size]

        tokenizer.src_lang = SRC_LANG
        inputs = tokenizer(
            batch_sents,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_SRC_LEN,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos,
                max_length=MAX_TGT_LEN,
                num_beams=4,
                early_stopping=True,
            )

        raw_outputs = tokenizer.batch_decode(generated, skip_special_tokens=True)

        for i, (sent, gold, raw) in enumerate(
            zip(batch_sents, batch_golds, raw_outputs)
        ):
            global_idx = start + i
            pred_penman = clean_pred_penman(
                linearized_to_penman(raw, graph_idx=global_idx)
            )
            results.append(
                {
                    "sentence": sent,
                    "gold_amr": gold.strip(),
                    "model_output_linearized": raw,
                    "model_output_penman": pred_penman,
                }
            )
    return results


# ─────────────────────────────────────────────
# Smatch evaluation
# ─────────────────────────────────────────────


def evaluate_smatch(predictions: List[dict], output_csv: str) -> None:
    results_out = []
    scored = skipped = 0

    for i, item in enumerate(tqdm(predictions, desc="Smatch")):
        score = smatch_score(item["gold_amr"], item["model_output_penman"])
        if score is None:
            skipped += 1
            results_out.append(
                {
                    "idx": i,
                    "sentence": item["sentence"],
                    "P": "",
                    "R": "",
                    "F1": "",
                    "status": "SKIP",
                }
            )
        else:
            p, r, f1 = score
            scored += 1
            results_out.append(
                {
                    "idx": i,
                    "sentence": item["sentence"],
                    "P": p,
                    "R": r,
                    "F1": f1,
                    "status": "OK",
                }
            )

    ok = [r for r in results_out if r["status"] == "OK"]
    f1_vals = [r["F1"] for r in ok]
    p_vals = [r["P"] for r in ok]
    r_vals = [r["R"] for r in ok]

    print(f"\n{'=' * 55}")
    print(f"Total test examples : {len(predictions)}")
    print(f"Scored (parsed)     : {scored}")
    print(f"Skipped             : {skipped}")
    if f1_vals:
        print(f"\n--- Scores on {scored} parseable pairs ---")
        print(f"Avg Precision   : {sum(p_vals) / len(p_vals):.4f}")
        print(f"Avg Recall      : {sum(r_vals) / len(r_vals):.4f}")
        print(f"Avg F1          : {sum(f1_vals) / len(f1_vals):.4f}")
        all_f1 = f1_vals + [0.0] * skipped
        print(f"\n--- Treating skipped as F1=0 (n={len(predictions)}) ---")
        print(f"Avg F1 (all)    : {sum(all_f1) / len(all_f1):.4f}")
        print("\nTop-10 F1:")
        for r in sorted(ok, key=lambda x: -x["F1"])[:10]:
            print(f"  [{r['idx']:04d}] F1={r['F1']:.3f}  {r['sentence'][:55]}")

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["idx", "sentence", "P", "R", "F1", "status"])
        w.writeheader()
        w.writerows(results_out)
    print(f"\nSaved smatch scores → {output_csv}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune mBART for Konkani AMR parsing")
    p.add_argument(
        "--data_csv",
        default="../baseline/data.csv",
        help="Path to data.csv with columns: sentence, amr_penman",
    )
    p.add_argument(
        "--output_dir",
        default="./konkani_amr_finetuned",
        help="Directory for model checkpoints and outputs",
    )
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument(
        "--batch_size", type=int, default=4, help="Per-device training batch size"
    )
    p.add_argument(
        "--grad_accum",
        type=int,
        default=4,
        help="Gradient accumulation steps (effective batch = batch_size * grad_accum)",
    )
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument(
        "--infer_batch", type=int, default=8, help="Batch size for inference"
    )
    p.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Enable fp16 training (requires CUDA)",
    )
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ── 1. Load & split data ──────────────────────────────────────────────
    print(f"\nLoading data from: {args.data_csv}")
    df = pd.read_csv(args.data_csv)
    assert "sentence" in df.columns and "amr_penman" in df.columns, (
        "CSV must have 'sentence' and 'amr_penman' columns."
    )
    df = df.dropna(subset=["sentence", "amr_penman"]).reset_index(drop=True)
    print(f"Total examples after dropping NaN: {len(df)}")

    train_df, val_df, test_df = split_data(df, seed=args.seed)
    print(f"Split  →  train: {len(train_df)}  val: {len(val_df)}  test: {len(test_df)}")

    # Save splits for reproducibility
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)
    print(f"Splits saved to {output_dir}/{{train,val,test}}.csv")

    # ── 2. Tokenizer & model ──────────────────────────────────────────────
    print(f"\nLoading tokenizer & model: {MODEL_NAME}")
    tokenizer = MBart50TokenizerFast.from_pretrained(MODEL_NAME)
    model = MBartForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.to(device)

    # ── 3. Datasets ───────────────────────────────────────────────────────
    train_dataset = KonkaniAMRDataset(
        train_df["sentence"].tolist(), train_df["amr_penman"].tolist(), tokenizer
    )
    val_dataset = KonkaniAMRDataset(
        val_df["sentence"].tolist(), val_df["amr_penman"].tolist(), tokenizer
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8
    )

    # ── 4. Training arguments ─────────────────────────────────────────────
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        predict_with_generate=False,  # loss-only eval during training
        fp16=args.fp16 and device == "cuda",
        logging_dir=str(output_dir / "logs"),
        logging_steps=50,
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
        # Needed for mBART – ensures labels have the correct lang token
        forced_bos_token_id=tokenizer.lang_code_to_id[TGT_LANG],
    )

    # ── 5. Train ──────────────────────────────────────────────────────────
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("\nStarting fine-tuning …")
    trainer.train()

    # Save the best model + tokenizer in a clean location
    best_model_dir = str(output_dir / "best_model")
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)
    print(f"Best model saved → {best_model_dir}")

    # ── 6. Reload best model for inference ───────────────────────────────
    print("\nLoading best model for test inference …")
    model = MBartForConditionalGeneration.from_pretrained(best_model_dir)
    model.to(device)
    tokenizer = MBart50TokenizerFast.from_pretrained(best_model_dir)

    # ── 7. Inference on test set ──────────────────────────────────────────
    test_sentences = test_df["sentence"].tolist()
    test_gold_amrs = test_df["amr_penman"].tolist()

    predictions = run_inference(
        test_sentences,
        test_gold_amrs,
        model,
        tokenizer,
        device,
        batch_size=args.infer_batch,
    )

    pred_json_path = str(output_dir / "test_predictions.json")
    with open(pred_json_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    print(f"\nTest predictions saved → {pred_json_path}")

    # ── 8. Smatch evaluation ──────────────────────────────────────────────
    smatch_csv_path = str(output_dir / "smatch_scores_test.csv")
    evaluate_smatch(predictions, smatch_csv_path)

    print("\nDone! 🎉")


if __name__ == "__main__":
    main()
