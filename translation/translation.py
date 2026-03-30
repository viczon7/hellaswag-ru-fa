!pip install deepl

import os
os.environ["DEEPL_API_KEY"] = "#API_key"

import os, json, deepl

auth_key = os.environ["DEEPL_API_KEY"]
translator = deepl.Translator(auth_key)

def translate_text(txt, target_lang):
    if not txt:
        return txt
    result = translator.translate_text(txt, target_lang=target_lang)
    return result.text

def translate_file(input_path, output_path, target_lang):
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            ex = json.loads(line)
            ctx_tr = translate_text(ex["context"], target_lang)
            ends_tr = [translate_text(e, target_lang) for e in ex["endings"]]
            ex_tr = {
                "context": ctx_tr,
                "endings": ends_tr,
                "label": ex["label"],
                "activity_label": ex["activity_label"],
                "source_id": ex["source_id"],
            }
            fout.write(json.dumps(ex_tr, ensure_ascii=False) + "\n")

translate_file("/content/hellaswag_val_500_en.jsonl",
               "/content/hellaswag_val_500_ru.jsonl", "RU") #FA for Persian
