import csv
import json
import os
import re
import subprocess
import sys
import tempfile

import penman

with open("../find_smatch/konkani_amr_predictions.json", encoding="utf-8") as f:
    data = json.load(f)


def clean_pred_penman(text):
    text = re.sub(
        r"<lit>\s*(.*?)\s*</lit>",
        lambda m: '"' + m.group(1).strip() + '"',
        text,
        flags=re.DOTALL,
    )
    text = re.sub(r"^thing\(", "(", text.strip())
    text = re.sub(r"\(x\d+_\d+\s*/\s*\)", "(amr-unknown)", text)
    text = re.sub(
        r"(:(?:ARG\d+|op\d+|mod|poss|quant|domain|time|location|manner|cause|degree|purpose|condition|wiki|name|polarity|mode|li|value|snt\d+))(x\d+_\d+)",
        r"\1 \2",
        text,
    )
    return text


def safe_encode(amr_str):
    try:
        return penman.encode(penman.decode(amr_str.strip()))
    except Exception:
        return None


def smatch_score(gold_str, pred_str):
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
            # ["python3", "-m", "smatch", "--pr", "-f", pred_path, gold_path],
            [sys.executable, "-m", "smatch", "--pr", "-f", pred_path, gold_path],
            capture_output=True,
            text=True,
            timeout=15,
        )
        os.unlink(gold_path)
        os.unlink(pred_path)
        lines = result.stdout.strip().split("\n")
        p = r = f = None
        for line in lines:
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


results_out = []
skipped = 0
scored = 0

for i, item in enumerate(data):
    if i % 100 == 0:
        print(f"  Processing {i}/{len(data)}...", flush=True)
    pred_cleaned = clean_pred_penman(item["model_output_penman"])
    score = smatch_score(item["gold_amr"], pred_cleaned)
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

ok_results = [r for r in results_out if r["status"] == "OK"]
f1_vals = [r["F1"] for r in ok_results]
p_vals = [r["P"] for r in ok_results]
r_vals = [r["R"] for r in ok_results]

print(f"\n{'=' * 55}")
print(f"Total examples  : {len(data)}")
print(f"Scored (parsed) : {scored}")
print(f"Skipped         : {skipped}")
print(f"\n--- Scores on {scored} parseable pairs ---")
print(f"Avg Precision   : {sum(p_vals) / len(p_vals):.4f}")
print(f"Avg Recall      : {sum(r_vals) / len(r_vals):.4f}")
print(f"Avg F1          : {sum(f1_vals) / len(f1_vals):.4f}")
print(f"\n--- Treating skipped as F1=0 (n={len(data)}) ---")
all_f1 = f1_vals + [0.0] * skipped
print(f"Avg F1 (all)    : {sum(all_f1) / len(all_f1):.4f}")
print("\nTop-10 F1:")
for r in sorted(ok_results, key=lambda x: -x["F1"])[:10]:
    print(f"  [{r['idx']:04d}] F1={r['F1']:.3f}  {r['sentence'][:55]}")

with open(r"smatch_scores.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["idx", "sentence", "P", "R", "F1", "status"])
    w.writeheader()
    w.writerows(results_out)
print("\nSaved → smatch_scores.csv")

"""
=======================================================
Total examples  : 1100
Scored (parsed) : 642
Skipped         : 458

--- Scores on 642 parseable pairs ---
Avg Precision   : 0.1744
Avg Recall      : 0.2822
Avg F1          : 0.2069

--- Treating skipped as F1=0 (n=1100) ---
Avg F1 (all)    : 0.1207

Top-10 F1:
  [0369] F1=0.620  बिटॉनीचो जन्म  1934 वर्सा किआराव्हाल्य, आंकोन्या जालो.
  [0216] F1=0.600  पुनर्रचणूक 31 ऑक्टोबर 2019 सावन लागू जाली.
  [0202] F1=0.560  बेंगळुरू, मुंबय, अहमदाबाद आनी दिल्ली निदर्शनां जालीं.
  [0856] F1=0.560  पूण वर्साक 7,200 अमेरिकी डॉलर खर्च केल्यार तुमची बचत 14
  [0749] F1=0.490  1967 वर्सा भैरो सिंग शेखावताच्या फुडारपणाखाला जन संघान 
  [0761] F1=0.470  सरगुरच्यान उत्तरेक सुमार 12 किलोमीटर पयस एच.डी. कोटे वस
  [0115] F1=0.420  ती किंताना रु राज्याची राजधानी.
  [0250] F1=0.420  रुबेन दारिओ, अर्नस्टो कार्डिनॅल आनी जायकॉण्डो बॅली हे क
  [0355] F1=0.420  20 फेब्रुवारी 2008 दिसा नॅशनल हॉकी लीगांत 6 खेळ जाल्ले.
  [0453] F1=0.420  सिरियांत आनी इराकांत सुमार 500 विदेशी  झुजारी बेल्जियन 

Saved → smatch_scores.csv
"""
