# Cross-lingual Commonsense Evaluation on HellaSwag

This repository contains Python code and datasets used in the term paper **'Cross-lingual Commonsense Evaluation: Translating HellaSwag into Russian and Persian'**. It provides scripts for translating a 500-item subset of HellaSwag, post-processing the machine translations, and evaluating Gemma‑2‑2B on the English, Russian, and Persian versions.

## Contents

- `translation.py` – uses the DeepL API to translate `hellaswag_val_500_en.jsonl` into target languages (RU / FA) while preserving the JSONL structure.
- `post-processing.py` – normalizes structural markers (e.g. `[header]`, `[title]`, `[step]`) in the translated files and fixes minor formatting issues.
- `evaluation.py` – computes log-likelihood–based multiple-choice scores for each option and reports accuracy for each JSONL file.
- `samples_hellaswag.py` – takes random 500 items from the HellaSwag dataset.
- `hellaswag_val_500_en.jsonl` – original 500-item English validation subset
- `hellaswag_val_500_ru.jsonl` / `hellaswag_val_500_fa.jsonl` – raw DeepL translations.
- `hellaswag_val_500_ru_fixed.jsonl` – Russian file after marker normalization.
- `hellaswag_val_500_ru_edited.jsonl` / `hellaswag_val_500_fa_edited.jsonl` – manually post-edited versions.
- `human post-editing RU.pdf` / `human post-editing FA.pdf` – short documentation of mistakes in Russian and Persian machine translation.

## Requirements

- Python 3.10+
- `transformers`, `torch`
- `deepl` (official Python client)
- A valid DeepL API key (for running `translation.py`)

## Usage

1. **Translate the English subset**

   ```bash
   export DEEPL_API_KEY=your_key_here
   python translation.py

2. **Normalize structural markers and post-edit translations**

   ```bash
   python post-processing.py

3. **Evaluate Gemma‑2‑2B on a given language file**

   The script prints accuracy for each JSONL file (English, initial Russian and Persian, edited Russian and Persian).

   ```bash
   python evaluation.py
   
## Citation

References to the underlying benchmark and model:

- Zellers et al. (2019), HellaSwag: Can a Machine Really Finish Your Sentence?
- Gemma Team (2024), Gemma 2: Improving Open Language Models at a Practical Size.
