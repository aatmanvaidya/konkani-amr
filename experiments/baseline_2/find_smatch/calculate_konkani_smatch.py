#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any

import penman
from backoff_smatch import BackOffSmatchpp
from postprocessing_graph import (
    BACKOFF,
    ParsedStatus,
    connect_graph_if_not_connected,
    fix_and_make_graph,
)
from postprocessing_str import (
    postprocess_str_after_delinearization,
    tokenize_except_quotes_and_angles,
)
from smatchpp import eval_statistics, preprocess, solvers

# def _setup_imports() -> None:
#     repo_root = Path(__file__).resolve().parents[3]
#     src_dir = repo_root / "multilingual-text-to-amr" / "src"
#     if str(src_dir) not in sys.path:
#         sys.path.insert(0, str(src_dir))


# _setup_imports()


def linearized_to_penman(linearized: str) -> tuple[str, str]:
    try:
        postprocessed = postprocess_str_after_delinearization(linearized)
        nodes = tokenize_except_quotes_and_angles(postprocessed)
        graph = fix_and_make_graph(nodes, verbose=False)
        graph, status = connect_graph_if_not_connected(graph)
        return penman.encode(graph), status.name.lower()
    except Exception:
        return penman.encode(BACKOFF), ParsedStatus.BACKOFF.name.lower()


def load_predictions(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data


# def compute_smatch(refs: list[str], preds: list[str]) -> dict[str, float]:
#     graph_standardizer = preprocess.AMRStandardizer(syntactic_standardization="dereify")
#     ilp = solvers.ILP()
#     printer = eval_statistics.ResultPrinter(score_type="micro", do_bootstrap=True, output_format="json")
#     smatch_metric = BackOffSmatchpp(alignmentsolver=ilp, graph_standardizer=graph_standardizer, printer=printer)

#     score, _, backoff_idxs = smatch_metric.score_corpus(refs, preds)
#     main = score["main"]
#     return {
#         "smatch_precision": float(main["Precision"]),
#         "smatch_recall": float(main["Recall"]),
#         "smatch_fscore": float(main["F1"]),
#         "smatch_backoff_pairs": len(backoff_idxs),
#     }


def compute_smatch(refs: list[str], preds: list[str]) -> dict[str, float]:
    ilp = solvers.ILP()
    printer = eval_statistics.ResultPrinter(
        score_type="micro", do_bootstrap=True, output_format="json"
    )

    # smatchpp API changed across versions — AMRStandardizer moved or was renamed
    try:
        # Newer smatchpp versions
        from smatchpp import standardize

        graph_standardizer = standardize.AMRStandardizer(
            syntactic_standardization="dereify"
        )
    except (ImportError, AttributeError):
        try:
            # Older API
            graph_standardizer = preprocess.AMRStandardizer(
                syntactic_standardization="dereify"
            )
        except AttributeError:
            # Fall back to no standardizer
            graph_standardizer = None

    if graph_standardizer is not None:
        smatch_metric = BackOffSmatchpp(
            alignmentsolver=ilp,
            graph_standardizer=graph_standardizer,
            printer=printer,
        )
    else:
        smatch_metric = BackOffSmatchpp(alignmentsolver=ilp, printer=printer)

    score, _, backoff_idxs = smatch_metric.score_corpus(refs, preds)
    main = score["main"]
    return {
        "smatch_precision": float(main["Precision"]),
        "smatch_recall": float(main["Recall"]),
        "smatch_fscore": float(main["F1"]),
        "smatch_backoff_pairs": len(backoff_idxs),
    }


def build_results(
    rows: list[dict[str, Any]],
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    refs: list[str] = []
    preds: list[str] = []
    detailed_rows: list[dict[str, Any]] = []

    for idx, row in enumerate(rows):
        sentence = row.get("sentence", "")
        gold = row.get("gold_amr", "")
        linearized = row.get("model_output_linearized", "")

        if not gold or not isinstance(gold, str):
            continue
        if not linearized or not isinstance(linearized, str):
            continue

        pred_penman, parse_status = linearized_to_penman(linearized)

        refs.append(gold)
        preds.append(pred_penman)
        detailed_rows.append(
            {
                "idx": idx,
                "sentence": sentence,
                "gold_amr": gold,
                "model_output_linearized": linearized,
                "predicted_penman_processed": pred_penman,
                "parse_status": parse_status,
            }
        )

    return refs, preds, detailed_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Smatch for Konkani AMR predictions."
    )
    parser.add_argument(
        "--predictions-json",
        type=Path,
        default=Path(__file__).resolve().parent / "konkani_amr_predictions.json",
        help="Path to JSON with sentence/gold_amr/model_output_linearized fields.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(__file__).resolve().parent / "konkani_amr_smatch_results.json",
        help="Where to save aggregate Smatch results.",
    )
    parser.add_argument(
        "--output-detailed-json",
        type=Path,
        default=Path(__file__).resolve().parent / "konkani_amr_predictions_scored.json",
        help="Where to save per-example processed prediction details.",
    )
    args = parser.parse_args()

    rows = load_predictions(args.predictions_json)
    refs, preds, detailed_rows = build_results(rows)
    if not refs:
        raise ValueError(
            "No valid rows found. Ensure gold_amr and model_output_linearized exist."
        )

    metrics = compute_smatch(refs, preds)
    metrics["num_examples_scored"] = len(refs)
    metrics["num_examples_total"] = len(rows)

    args.output_json.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    args.output_detailed_json.write_text(
        json.dumps(detailed_rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nWrote metrics to: {args.output_json}")
    print(f"Wrote detailed rows to: {args.output_detailed_json}")


if __name__ == "__main__":
    main()
