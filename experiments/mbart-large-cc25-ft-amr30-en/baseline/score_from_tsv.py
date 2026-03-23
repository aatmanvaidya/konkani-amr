"""
score_from_tsv.py

Loads the baseline predictions TSV, converts the BramVanroy model's custom
XML pointer-token linearization into standard Penman AMR notation, and
computes SMATCH scores against the gold references.

The BramVanroy model outputs a pointer/relation linearization like:
    <AMR><AMR><rel><pointer:0> concept:role<rel><pointer:1> child</rel></rel>
This script parses that back into Penman: (a / concept :role (b / child))

Usage:
    python score_from_tsv.py --tsv results/predictions.tsv
    python score_from_tsv.py --tsv results/predictions.tsv --output_dir results/
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import smatch

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


def tokenize(text: str) -> list:
    """Split linearized string into XML tags and plain word tokens."""
    result = []
    for part in re.split(r"(<[^>]*>)", text):
        if part.startswith("<") and part.endswith(">"):
            result.append(part.strip())
        else:
            result.extend(part.split())
    return [t for t in result if t]


# ---------------------------------------------------------------------------
# Variable name generator
# ---------------------------------------------------------------------------


def _fresh_var(concept: str, used: set) -> str:
    base = concept[0].lower() if concept and concept[0].isalpha() else "x"
    cand = base
    n = 2
    while cand in used:
        cand = f"{base}{n}"
        n += 1
    used.add(cand)
    return cand


# ---------------------------------------------------------------------------
# Regex for splitting fused value:role tokens
# e.g. "239000000:unit" → value="239000000", pending_role=":unit"
# e.g. "-:name"         → value="-",          pending_role=":name"
# ---------------------------------------------------------------------------

_ROLE_SUFFIX = re.compile(r"^(.+?)(:[a-zA-Z][a-zA-Z0-9_-]*)$")


# ---------------------------------------------------------------------------
# Linearization → Penman parser
# ---------------------------------------------------------------------------


class LinearizedAMRParser:
    """
    Converts BramVanroy's pointer-token AMR linearization to Penman notation.

    Key format observations (verified against real model output):
      - Starts with <AMR><AMR>
      - Nodes: <rel><pointer:N> concept ...children... </rel>
      - Concept token may carry first role: "concept:role" e.g. "change-01:ARG0"
      - Child roles are bare tokens: ":ARG1", ":wiki", ":quant", etc.
      - Values are one of:
          <rel><pointer:N> ... </rel>   nested node
          <lit> words </lit>            quoted string literal
          <pointer:N>                   back-reference (re-entrancy)
          plain token                   number, "-", bare word
      - IMPORTANT: plain value tokens sometimes carry a fused role suffix,
        e.g. "239000000:unit" or "-:name". The parser splits these and
        injects the role suffix back as the next child role via _pending_role.
    """

    def parse(self, text: str):
        try:
            self.toks = tokenize(text)
            self.pos = 0
            self._pending_role = None  # role suffix split off from a fused token
            # First pass: map pointer IDs to concept names
            self.p2c = {}
            self._scan_concepts()
            # Assign short Penman variables
            used = set()
            self.p2v = {pid: _fresh_var(c, used) for pid, c in sorted(self.p2c.items())}
            # Second pass: recursive parse
            self.pos = 0
            self._pending_role = None
            self._skip(["<AMR>"])
            result = self._parse_rel()
            if result and result.startswith("(") and " / " in result:
                return result
        except Exception:
            pass
        return None

    # -- Cursor helpers --

    def _cur(self):
        """Return current token, honouring any pending injected role."""
        if self._pending_role is not None:
            return self._pending_role
        return self.toks[self.pos] if self.pos < len(self.toks) else None

    def _adv(self):
        """Advance and return current token, consuming pending role first."""
        if self._pending_role is not None:
            t = self._pending_role
            self._pending_role = None
            return t
        t = self.toks[self.pos] if self.pos < len(self.toks) else None
        self.pos += 1
        return t

    def _skip(self, tags):
        while self._cur() in tags:
            self._adv()

    # -- First pass: scan pointer→concept mappings --

    def _scan_concepts(self):
        for i, tok in enumerate(self.toks):
            m = re.fullmatch(r"<pointer:(\d+)>", tok)
            if not m:
                continue
            pid = int(m.group(1))
            j = i + 1
            while j < len(self.toks) and self.toks[j] in (
                "<AMR>",
                "<rel>",
                "</rel>",
                "<lit>",
                "</lit>",
            ):
                j += 1
            if j < len(self.toks) and not self.toks[j].startswith("<"):
                raw = self.toks[j]
                concept = raw.split(":")[0]
                if (
                    concept
                    and not concept.startswith(":")
                    and concept not in ("-", "+")
                ):
                    self.p2c[pid] = concept

    # -- Second pass: recursive descent --

    def _parse_rel(self):
        """Parse a <rel><pointer:N> ... </rel> block into a Penman node."""
        self._skip(["<AMR>"])
        if self._cur() != "<rel>":
            return None
        self._adv()
        self._skip(["<AMR>"])

        tok = self._cur()
        m = re.fullmatch(r"<pointer:(\d+)>", tok or "")
        if not m:
            return None
        self._adv()
        pid = int(m.group(1))
        concept = self.p2c.get(pid)
        var = self.p2v.get(pid, f"x{pid}")
        if not concept:
            return None

        children = []

        # Consume concept word token (may carry ":firstRole" suffix)
        tok = self._cur()
        if tok and not tok.startswith("<"):
            self._adv()
            if ":" in tok:
                role = tok[tok.index(":") :]
                child = self._parse_value()
                if child is not None:
                    children.append(f"{role} {child}")

        # Consume remaining children until </rel>
        while True:
            self._skip(["<AMR>"])
            tok = self._cur()
            if tok is None or tok == "</rel>":
                break
            if tok and tok.startswith(":"):
                role = self._adv()
                child = self._parse_value()
                if child is not None:
                    children.append(f"{role} {child}")
            elif re.fullmatch(r"<pointer:\d+>", tok or ""):
                self._adv()  # stray back-reference without role
            else:
                self._adv()

        if self._cur() == "</rel>":
            self._adv()

        if children:
            return f"({var} / {concept} {' '.join(children)})"
        return f"({var} / {concept})"

    def _parse_value(self):
        """Parse the value of a role."""
        self._skip(["<AMR>"])
        tok = self._cur()
        if tok is None:
            return None
        if tok == "<rel>":
            return self._parse_rel()
        if tok == "<lit>":
            return self._parse_lit()
        m = re.fullmatch(r"<pointer:(\d+)>", tok)
        if m:
            self._adv()
            pid = int(m.group(1))
            nxt = self._cur()
            if nxt and not nxt.startswith("<") and not nxt.startswith(":"):
                concept = self.p2c.get(pid, nxt.split(":")[0])
                var = self.p2v.get(pid, f"x{pid}")
                self._adv()
                return f"({var} / {concept})"
            return self.p2v.get(pid, f"x{pid}")
        if tok and tok.startswith("<") and tok.endswith(">"):
            return None  # unknown tag
        # Plain token — split off any fused ":role" suffix
        self._adv()
        m2 = _ROLE_SUFFIX.match(tok)
        if m2:
            # E.g. "239000000:unit" → return "239000000", inject ":unit" as next role
            # E.g. "-:name"         → return "-",          inject ":name" as next role
            self._pending_role = m2.group(2)
            return m2.group(1)
        return tok

    def _parse_lit(self):
        """Parse <lit> words </lit> into a quoted string."""
        if self._cur() != "<lit>":
            return None
        self._adv()
        parts = []
        while self._cur() and self._cur() != "</lit>":
            parts.append(self._adv())
        if self._cur() == "</lit>":
            self._adv()
        return f'"{" ".join(parts)}"'


# ---------------------------------------------------------------------------
# Public conversion entry point
# ---------------------------------------------------------------------------


def linearized_to_penman(text: str):
    return LinearizedAMRParser().parse(text)


# ---------------------------------------------------------------------------
# SMATCH scoring
# ---------------------------------------------------------------------------


def compute_smatch(golds: list, preds: list):
    total_match = total_test = total_gold = 0
    unparsable = 0
    for gold, pred in zip(golds, preds):
        if pred is None:
            unparsable += 1
            continue
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
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert XML-linearized AMR predictions to Penman and compute SMATCH."
    )
    parser.add_argument(
        "--tsv",
        type=str,
        required=True,
        help="Path to predictions TSV (sentence / reference_penman / predicted_penman).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for converted TSV + metrics JSON. Defaults to TSV's directory.",
    )
    parser.add_argument("--sentence_col", type=str, default="sentence")
    parser.add_argument("--reference_col", type=str, default="reference_penman")
    parser.add_argument("--prediction_col", type=str, default="predicted_penman")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tsv_path = Path(args.tsv)
    if not tsv_path.exists():
        raise FileNotFoundError(f"TSV not found: {tsv_path}")

    out_dir = Path(args.output_dir) if args.output_dir else tsv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {tsv_path}")
    df = pd.read_csv(tsv_path, sep="\t")
    required = {args.sentence_col, args.reference_col, args.prediction_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}. Found: {list(df.columns)}")

    df = df.dropna(subset=[args.reference_col, args.prediction_col]).reset_index(
        drop=True
    )
    n = len(df)
    print(f"Rows: {n}")

    refs = df[args.reference_col].astype(str).tolist()
    raw_preds = df[args.prediction_col].astype(str).tolist()

    print("Converting linearized predictions to Penman...")
    converted = [linearized_to_penman(p) for p in raw_preds]
    conv_ok = sum(1 for c in converted if c is not None)
    conv_rate = 100.0 * conv_ok / n if n else 0.0
    print(f"  Converted: {conv_ok}/{n} ({conv_rate:.1f}%)")

    print("Computing SMATCH...")
    precision, recall, f1, unparsable = compute_smatch(refs, converted)
    unparsable_pct = 100.0 * unparsable / n if n else 0.0

    metrics = {
        "Smatch Precision": round(float(precision), 4),
        "Smatch Recall": round(float(recall), 4),
        "Smatch Fscore": round(float(f1), 4),
        "Smatch Unparsable": round(unparsable_pct, 2),
        "Smatch Unparsable Count": int(unparsable),
        "Conversion Success Rate": round(conv_rate, 2),
        "Num Examples": int(n),
    }

    # Save converted predictions
    out_df = pd.DataFrame(
        {
            args.sentence_col: df[args.sentence_col].tolist(),
            args.reference_col: refs,
            "predicted_linearized": raw_preds,
            "predicted_penman": [c if c else "" for c in converted],
        }
    )
    out_tsv = out_dir / "predictions_converted.tsv"
    out_df.to_csv(out_tsv, sep="\t", index=False, encoding="utf-8")

    metrics_json = out_dir / "metrics_converted.json"
    metrics_json.write_text(
        json.dumps({"metrics": metrics}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 50)
    print("Results")
    print("=" * 50)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nConverted predictions : {out_tsv}")
    print(f"Metrics JSON          : {metrics_json}")

    print("\n--- Sample conversions (first 3) ---")
    shown = 0
    for i, (raw, conv) in enumerate(zip(raw_preds, converted)):
        if conv is not None and shown < 3:
            print(f"\n[{i}] RAW:  {raw[:100]}...")
            print(f"[{i}] CONV: {conv[:200]}")
            shown += 1
    if shown == 0:
        print("No successful conversions — check the prediction format.")


if __name__ == "__main__":
    main()
