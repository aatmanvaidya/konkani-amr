#!/usr/bin/env python3
"""
Fine-tune mBART for Konkani AMR parsing and evaluate with SMATCH.

Usage:
    python finetune_konkani_amr.py [args]

Splits data 80/20 train/test, fine-tunes, then evaluates SMATCH on the held-out test set.
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import smatch
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    MBartForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune mBART for Konkani→AMR parsing and evaluate SMATCH."
    )
    # Data
    parser.add_argument("--data_csv", type=str, default="data.csv")
    parser.add_argument("--text_column", type=str, default="sentence")
    parser.add_argument("--amr_column", type=str, default="amr_penman")
    parser.add_argument("--test_size", type=float, default=0.2, help="Fraction held out for testing.")

    # Model
    parser.add_argument(
        "--model_name",
        type=str,
        default="BramVanroy/mbart-large-cc25-ft-amr30-en",
    )
    parser.add_argument("--src_lang", type=str, default="hi_IN",
                        help="mBART language code closest to Konkani (Devanagari).")
    parser.add_argument("--forced_bos_token_id", type=int, default=250181,
                        help="AMR BOS token id.")

    # Training hyper-params
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Effective batch = per_device × grad_accum.")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_source_length", type=int, default=128)
    parser.add_argument("--max_target_length", type=int, default=512)
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--fp16", action="store_true",
                        help="Enable mixed-precision training (A100 supports bf16).")
    parser.add_argument("--bf16", action="store_true",
                        help="Enable bfloat16 training (preferred on A100).")

    # Generation (for inference after training)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=512)

    # Paths
    parser.add_argument("--output_dir", type=str, default="./results_finetune")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                        help="Where HF Trainer saves model checkpoints.")

    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AMRDataset(Dataset):
    """Tokenises sentence→AMR pairs for Seq2Seq training."""

    def __init__(
        self,
        sentences: List[str],
        amrs: List[str],
        tokenizer: AutoTokenizer,
        src_lang: str,
        max_source_length: int,
        max_target_length: int,
    ):
        self.tokenizer = tokenizer
        self.src_lang = src_lang
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.sentences = sentences
        self.amrs = amrs

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> dict:
        self.tokenizer.src_lang = self.src_lang

        model_inputs = self.tokenizer(
            self.sentences[idx],
            max_length=self.max_source_length,
            truncation=True,
            padding=False,
        )

        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                self.amrs[idx],
                max_length=self.max_target_length,
                truncation=True,
                padding=False,
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_and_split(
    csv_path: str,
    text_col: str,
    amr_col: str,
    test_size: float,
    seed: int,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[text_col, amr_col]).reset_index(drop=True)

    n = len(df)
    if n < 5:
        raise ValueError(f"Dataset too small ({n} rows). Need at least 5 examples.")

    indices = list(range(n))
    random.shuffle(indices)

    n_test = max(1, int(n * test_size))
    n_train = n - n_test

    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    train_sentences = df[text_col].iloc[train_idx].astype(str).tolist()
    train_amrs = df[amr_col].iloc[train_idx].astype(str).tolist()
    test_sentences = df[text_col].iloc[test_idx].astype(str).tolist()
    test_amrs = df[amr_col].iloc[test_idx].astype(str).tolist()

    print(f"Dataset: {n} total → {n_train} train / {n_test} test")
    return train_sentences, train_amrs, test_sentences, test_amrs


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def batch_generate(
    texts: List[str],
    model: MBartForConditionalGeneration,
    tokenizer: AutoTokenizer,
    src_lang: str,
    num_beams: int,
    max_new_tokens: int,
    forced_bos_token_id: int,
    batch_size: int = 4,
) -> List[str]:
    tokenizer.src_lang = src_lang
    preds: List[str] = []

    for start in tqdm(range(0, len(texts), batch_size), desc="Generating", unit="batch"):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
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
        preds.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))

    return preds


# ---------------------------------------------------------------------------
# SMATCH
# ---------------------------------------------------------------------------

def compute_smatch(
    golds: List[str], preds: List[str]
) -> Tuple[float, float, float, int]:
    total_match = total_test = total_gold = 0
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    setup_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Load & split data ──────────────────────────────────────────────────
    train_sentences, train_amrs, test_sentences, test_amrs = load_and_split(
        csv_path=args.data_csv,
        text_col=args.text_column,
        amr_col=args.amr_column,
        test_size=args.test_size,
        seed=args.seed,
    )

    # Save the splits so results are reproducible
    pd.DataFrame({"sentence": train_sentences, "amr_penman": train_amrs}).to_csv(
        out_dir / "train_split.csv", index=False, encoding="utf-8"
    )
    pd.DataFrame({"sentence": test_sentences, "amr_penman": test_amrs}).to_csv(
        out_dir / "test_split.csv", index=False, encoding="utf-8"
    )
    print(f"Splits saved to {out_dir}/{{train,test}}_split.csv")

    # ── Load model & tokenizer ─────────────────────────────────────────────
    print(f"\nLoading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = MBartForConditionalGeneration.from_pretrained(args.model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Build datasets ─────────────────────────────────────────────────────
    common_ds_kwargs = dict(
        tokenizer=tokenizer,
        src_lang=args.src_lang,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
    )
    train_dataset = AMRDataset(train_sentences, train_amrs, **common_ds_kwargs)
    eval_dataset = AMRDataset(test_sentences, test_amrs, **common_ds_kwargs)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=tokenizer.pad_token_id,
        pad_to_multiple_of=8 if (args.fp16 or args.bf16) else None,
    )

    # ── Training arguments ─────────────────────────────────────────────────
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(ckpt_dir),
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        fp16=args.fp16,
        bf16=args.bf16,
        predict_with_generate=False,   # We do generation manually for SMATCH
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_dir=str(out_dir / "logs"),
        logging_steps=10,
        save_total_limit=2,
        report_to="none",              # Change to "wandb" if you want W&B logging
        seed=args.seed,
        dataloader_num_workers=4,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    # ── Fine-tune ──────────────────────────────────────────────────────────
    print("\n=== Starting fine-tuning ===")
    trainer.train()
    print("=== Fine-tuning complete ===\n")

    # Save the best model explicitly so we can reload it cleanly
    best_model_dir = out_dir / "best_model"
    trainer.save_model(str(best_model_dir))
    tokenizer.save_pretrained(str(best_model_dir))
    print(f"Best model saved to {best_model_dir}")

    # ── Evaluate on test split ─────────────────────────────────────────────
    print("\n=== Generating predictions on test set ===")
    model.eval()
    model.to(device)

    preds = batch_generate(
        texts=test_sentences,
        model=model,
        tokenizer=tokenizer,
        src_lang=args.src_lang,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
        forced_bos_token_id=args.forced_bos_token_id,
        batch_size=args.per_device_eval_batch_size,
    )

    # ── Compute SMATCH ─────────────────────────────────────────────────────
    precision, recall, f1, unparsable_count = compute_smatch(test_amrs, preds)
    unparsable_pct = (100.0 * unparsable_count / len(preds)) if preds else 0.0

    metrics = {
        "Smatch_Precision": float(precision),
        "Smatch_Recall": float(recall),
        "Smatch_F1": float(f1),
        "Smatch_Unparsable_Pct": float(unparsable_pct),
        "Smatch_Unparsable_Count": int(unparsable_count),
        "Num_Test_Examples": int(len(test_sentences)),
        "Num_Train_Examples": int(len(train_sentences)),
    }

    # ── Save outputs ───────────────────────────────────────────────────────
    pd.DataFrame(
        {
            "sentence": test_sentences,
            "reference_penman": test_amrs,
            "predicted_penman": preds,
        }
    ).to_csv(out_dir / "test_predictions.tsv", sep="\t", index=False, encoding="utf-8")

    (out_dir / "predictions.txt").write_text("\n\n".join(preds), encoding="utf-8")
    (out_dir / "references.txt").write_text("\n\n".join(test_amrs), encoding="utf-8")

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps({"metrics": metrics}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== Fine-tuning + Evaluation Complete ===")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nSaved metrics  : {metrics_path}")
    print(f"Saved preds    : {out_dir / 'test_predictions.tsv'}")
    print(f"Best model dir : {best_model_dir}")


if __name__ == "__main__":
    main()