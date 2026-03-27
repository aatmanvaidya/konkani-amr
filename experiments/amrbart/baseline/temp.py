import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Model name
model_name = "xfbai/AMRBART-large-finetuned-AMR3.0-AMRParsing-v2"

# Load tokenizer + model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# Test sentence
sentence = "The boy wants to go to school."

# Tokenize
inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512).to(
    device
)

# Generate AMR
with torch.no_grad():
    outputs = model.generate(**inputs, max_length=512, num_beams=5)

# Decode
amr = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Input Sentence:")
print(sentence)
print("\nPredicted AMR:")
print(amr)
