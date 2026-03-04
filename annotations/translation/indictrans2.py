import csv
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

src_lang, tgt_lang = "gom_Deva", "eng_Latn"
model_name = "ai4bharat/indictrans2-indic-en-1B"

input_txt_file = "clean_konkani_sentences.txt"
output_csv_file = "konkani_to_english.csv"
batch_size = 16

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    attn_implementation="flash_attention_2" if DEVICE == "cuda" else None,
).to(DEVICE)

ip = IndicProcessor(inference=True)

with open(input_txt_file, "r", encoding="utf-8") as f:
    sentences = [line.strip() for line in f if line.strip()]

with open(output_csv_file, "w", encoding="utf-8", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["konkani_text", "english_translation"])

    for i in range(0, len(sentences), batch_size):
        batch_sentences = sentences[i:i + batch_size]

        processed_batch = ip.preprocess_batch(
            batch_sentences,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
        )

        inputs = tokenizer(
            processed_batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(DEVICE)

        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=256,
                num_beams=5,
                num_return_sequences=1,
            )

        generated_tokens = tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        translations = ip.postprocess_batch(generated_tokens, lang=tgt_lang)

        for src, tgt in zip(batch_sentences, translations):
            writer.writerow([src, tgt])

print("Translation complete")