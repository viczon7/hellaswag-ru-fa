from datasets import load_dataset
import json

dataset_val = load_dataset("Rowan/hellaswag", split="validation")

dataset_500 = dataset_val.shuffle(seed=42).select(range(500))

out_path = "hellaswag_val_500_en.jsonl"
with open(out_path, "w", encoding="utf-8") as f:
     for ex in dataset_500:
         f.write(json.dumps({
             "context": ex["ctx"],
             "endings": ex["endings"],
             "label": int(ex["label"]),
             "activity_label": ex["activity_label"],
             "source_id": ex["source_id"],
         }, ensure_ascii=False) + "\n")

print("wrote", len(dataset_500), "examples to", out_path)
