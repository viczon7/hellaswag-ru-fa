import json

replacements = {
    "[заголовок]": "[header]",
    "[название]": "[title]",
    "[шаг]": "[step]",
    "[подшаги]": "[substeps]",
    "[подшагивания]": "[substeps]"
}

def normalize_markers(text: str) -> str:
    for src, tgt in replacements.items():
        text = text.replace(src, tgt)
    return text

in_path = "hellaswag_val_500_ru.jsonl"
out_path = "hellaswag_val_500_ru_fixed.jsonl"

with open(in_path, "r", encoding="utf-8") as fin, \
     open(out_path, "w", encoding="utf-8") as fout:
    for line in fin:
        ex = json.loads(line)

        ex["context"] = normalize_markers(ex["context"])
        ex["endings"] = [normalize_markers(e) for e in ex["endings"]]

        fout.write(json.dumps(ex, ensure_ascii=False) + "\n")