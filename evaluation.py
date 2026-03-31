!pip install transformers accelerate torch

!pip install huggingface_hub
from huggingface_hub import login

login()

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "google/gemma-2-2b"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to(device)
model.eval()

#probability calculation
def option_loglike(context, option):
    text = context + " " + option
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    with torch.no_grad():
        out = model(input_ids, labels=input_ids)
        loss = out.loss #mean negative loglikelihood over all tokens

    return -loss.item() #item for a float only => loss -> eval score

#evaluation using the dataset
def evaluate(jsonl_path):
    correct = 0
    total = 0
    with open(jsonl_path, "r", encoding="utf-8") as file:
        for line in file:
            ex = json.loads(line) #example #.loads => json line -> python dict
            ctx = ex["context"]
            options = ex["endings"]
            gold = ex["label"]

            scores = [option_loglike(ctx, opt) for opt in options] #probability for every option
            pred = max(range(len(scores)), key=lambda i: scores[i])

            total += 1
            if pred == gold:
                correct += 1

    return correct/total

accuracy_fa_edited = evaluate("hellaswag_val_500_fa_edited.jsonl") #change files for evaluation
print("Accuracy for FA edited:", accuracy_fa_edited)
