#!/usr/bin/env python3
import argparse
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from typing import Dict, List

import pandas as pd
import penman
import torch
from smatchpp import eval_statistics, preprocess, solvers
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM

from amrlib.evaluate.smatch_enhanced import compute_scores as amrlib_compute_scores
from multi_amr.evaluate.backoff_smatch import BackOffSmatchpp
from multi_amr.tokenization import AMRTokenizerWrapper, TokenizerType


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AMR baseline and compute SMATCH metrics on a CSV dataset.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="BramVanroy/mbart-large-cc25-ft-amr30-en",
        help="HF model name/path for text-to-AMR parsing.",
    )
    parser.add_argument(
        "--data_csv",
        type=str,
        default="annotations/gemini/output_train/data.csv",
        help="CSV containing text + gold AMR Penman annotations.",
    )
    parser.add_argument("--text_column", type=str, default="sentence", help="CSV column containing source text.")
    parser.add_argument("--amr_column", type=str, default="amr_penman", help="CSV column containing gold Penman AMRs.")
    parser.add_argument("--src_lang", type=str, default="en_XX", help="Source language tag for mbart tokenizer.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference.")
    parser.add_argument("--num_beams", type=int, default=5, help="Beam size for generation.")
    parser.add_argument("--max_new_tokens", type=int, default=900, help="Max new tokens to generate.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/baseline/results",
        help="Directory where predictions and metrics are written.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()


def setup_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_resources(model_name: str):
    tok_wrapper = AMRTokenizerWrapper.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tok_wrapper.tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tok_wrapper.tokenizer))
    return model, tok_wrapper, device


def batch_translate(
    texts: List[str],
    model,
    tok_wrapper: AMRTokenizerWrapper,
    src_lang: str,
    num_beams: int,
    max_new_tokens: int,
) -> Dict[str, List]:
    if tok_wrapper.tokenizer_type not in (TokenizerType.MBART, TokenizerType.NLLB, TokenizerType.BART, TokenizerType.T5):
        raise NotImplementedError(f"Unsupported tokenizer type for this script: {tok_wrapper.tokenizer_type}")

    if tok_wrapper.tokenizer_type in (TokenizerType.MBART, TokenizerType.NLLB):
        tok_wrapper.tokenizer.src_lang = src_lang
        decoder_start_token_id = tok_wrapper.amr_token_id
    else:
        decoder_start_token_id = None

    encoded = tok_wrapper(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        return_attention_mask=True,
    ).to(model.device)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "num_beams": num_beams,
        "num_return_sequences": num_beams,
        "output_scores": True,
        "return_dict_in_generate": True,
    }
    if decoder_start_token_id is not None:
        gen_kwargs["decoder_start_token_id"] = decoder_start_token_id

    with torch.inference_mode():
        generated = model.generate(**encoded, **gen_kwargs)

    generated["sequences"] = generated["sequences"].cpu()
    generated["sequences_scores"] = generated["sequences_scores"].cpu()

    best = {"graph": [], "status": []}
    beam_size = num_beams
    for sample_idx in range(0, len(generated["sequences_scores"]), beam_size):
        sequences = generated["sequences"][sample_idx : sample_idx + beam_size]
        scores = generated["sequences_scores"][sample_idx : sample_idx + beam_size].tolist()
        outputs = tok_wrapper.batch_decode_amr_ids(sequences)
        statuses = outputs["status"]
        graphs = outputs["graph"]
        zipped = zip(statuses, scores, graphs)
        # prefer parse status first, then model confidence
        choice = sorted(zipped, key=lambda item: (item[0].value, -item[1]))[0]
        best["graph"].append(choice[2])
        best["status"].append(choice[0])
    return best


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

    model, tok_wrapper, device = load_resources(args.model_name)
    print(f"Running on device: {device}")
    print(f"Dataset size: {len(texts)}")

    pred_graphs = []
    statuses = []
    for start in tqdm(range(0, len(texts), args.batch_size), desc="Inference", unit="batch"):
        batch_texts = texts[start : start + args.batch_size]
        outputs = batch_translate(
            texts=batch_texts,
            model=model,
            tok_wrapper=tok_wrapper,
            src_lang=args.src_lang,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
        )
        pred_graphs.extend(outputs["graph"])
        statuses.extend(outputs["status"])

    preds = [penman.encode(graph) for graph in pred_graphs]
    status_names = [status.name.lower() for status in statuses]

    preds_txt = out_dir / "predictions.txt"
    refs_txt = out_dir / "references.txt"
    preds_txt.write_text("\n\n".join(preds), encoding="utf-8")
    refs_txt.write_text("\n\n".join(refs), encoding="utf-8")

    graph_standardizer = preprocess.AMRStandardizer(syntactic_standardization="dereify")
    ilp_solver = solvers.ILP()
    printer = eval_statistics.ResultPrinter(score_type="micro", do_bootstrap=True, output_format="json")
    smatch_metric = BackOffSmatchpp(alignmentsolver=ilp_solver, graph_standardizer=graph_standardizer, printer=printer)
    smatch_scores, _, backoff_idxs = smatch_metric.score_corpus(refs, preds)

    main_scores = smatch_scores["main"]
    unparsable_count = sum(1 for s in status_names if s == "backoff")
    unparsable_pct = (100.0 * unparsable_count / len(status_names)) if status_names else 0.0

    # Enhanced smatch breakdown (acts as a classification-style report)
    enhanced_buf = io.StringIO()
    with redirect_stdout(enhanced_buf):
        amrlib_compute_scores(str(preds_txt), str(refs_txt))
    smatch_classification_report = enhanced_buf.getvalue()

    metrics = {
        "Smatch Precision": float(main_scores["Precision"]),
        "Smatch Recall": float(main_scores["Recall"]),
        "Smatch Fscore": float(main_scores["F1"]),
        "Smatch Unparsable": float(unparsable_pct),
        "Smatch Unparsable Count": int(unparsable_count),
        "Smatch Backoff Align Count": int(len(backoff_idxs)),
        "Num Examples": int(len(df)),
    }

    predictions_tsv = out_dir / "predictions.tsv"
    pd.DataFrame(
        {
            "sentence": texts,
            "reference_penman": refs,
            "predicted_penman": preds,
            "status": status_names,
        }
    ).to_csv(predictions_tsv, sep="\t", index=False, encoding="utf-8")

    metrics_json = out_dir / "metrics.json"
    metrics_json.write_text(
        json.dumps(
            {
                "metrics": metrics,
                "smatch_classification_report": smatch_classification_report,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report_txt = out_dir / "smatch_classification_report.txt"
    report_txt.write_text(smatch_classification_report, encoding="utf-8")

    print("\nBaseline evaluation completed.")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nSaved metrics: {metrics_json}")
    print(f"Saved predictions: {predictions_tsv}")
    print(f"Saved classification report: {report_txt}")


if __name__ == "__main__":
    main()
