import json
import os
import ast
import re

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ==================== INDICBART CONFIGURATION ====================
LANG_CODES = {
    'kn': '<2kn>',
    'hi': '<2hi>',
    'en': '<2en>'
}

# ==================== INFERENCE CONFIGURATION ====================
model_name      = "ai4bharat/IndicBART"
LANG            = 'hi'                   # change to 'kn' for Kannada
CHECKPOINT_PATH = "/raid/home/rizwank/Normalization/model_building/Indic_bart/indic_bart_hi/transformers_model"
INPUT_FILE      = "/raid/home/rizwank/Normalization/data_generation/Post_processed_Data/HINDI/train_hindi.txt"
OUTPUT_FILE     = "/raid/home/rizwank/Normalization/model_building/Indic_bart/indic_bart_hi/inference_out.jsonl"
DEVICE_STR      = "cuda:6"
BATCH_SIZE      = 32
MAX_INPUT_LEN   = 128
MAX_NEW_TOKENS  = 128
NUM_BEAMS       = 4


# ==================== CLEANING FUNCTIONS ====================
def clean_tagged_text(text):
    if text is None:
        return ""
    text = str(text).strip()
    text = re.sub(r"^\s*\d+\.\s+", "", text)
    text = re.sub(r"<[A-Z_]+>", "", text)
    text = re.sub(r"<([^>]+)>", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ==================== PARSING ====================
def parse_line(line):
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(line)
        except Exception:
            return None


# ==================== DATASET ====================
class IndicBARTInferenceDataset(Dataset):
    """
    Reads the same JSONL format as training.
    Returns both the tokenized input and the original record dict
    so all original keys are preserved in the output file.
    """

    INPUT_KEYS = ["tagged_output", "input", "source", "unnormalized"]

    def __init__(self, filepath, tokenizer, lang='hi', max_length=128):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.lang_token = LANG_CODES[lang]
        self.records    = []   # original dicts
        self.inputs     = []   # cleaned input strings

        lang_token_id = tokenizer.convert_tokens_to_ids(self.lang_token)
        if lang_token_id == tokenizer.unk_token_id:
            raise ValueError(
                f"Language token '{self.lang_token}' not found in tokenizer vocabulary."
            )
        self.forced_bos_token_id = lang_token_id

        print(f"Loading data from : {filepath}")
        print(f"Language          : {lang} | Token: {self.lang_token} | ID: {lang_token_id}")

        line_count = skipped_count = 0
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line_count += 1
                obj = parse_line(line)
                if obj is None:
                    skipped_count += 1
                    continue

                input_text = self._get_value(obj, self.INPUT_KEYS)
                if input_text is None:
                    skipped_count += 1
                    continue

                input_text = clean_tagged_text(input_text)
                if not input_text:
                    skipped_count += 1
                    continue

                self.records.append(obj)
                self.inputs.append(input_text)

        print(f"Total lines: {line_count} | Kept: {len(self.records)} | Skipped: {skipped_count}")

    @staticmethod
    def _get_value(obj, keys):
        for k in keys:
            if k in obj:
                return obj[k]
        return None

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        input_with_lang = f"{self.inputs[idx]} {self.lang_token}"
        enc = self.tokenizer(
            input_with_lang,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "record_idx":     idx,
        }


# ==================== COLLATE FUNCTION ====================
def make_collate_fn(pad_token_id):
    def collate_fn(batch):
        input_ids_list      = [item["input_ids"]      for item in batch]
        attention_mask_list = [item["attention_mask"]  for item in batch]
        record_idxs         = [item["record_idx"]      for item in batch]

        padded_inputs = torch.nn.utils.rnn.pad_sequence(
            input_ids_list, batch_first=True, padding_value=pad_token_id
        )
        padded_masks = torch.nn.utils.rnn.pad_sequence(
            attention_mask_list, batch_first=True, padding_value=0
        )
        return {
            "input_ids":      padded_inputs,
            "attention_mask": padded_masks,
            "record_idxs":    record_idxs,
        }
    return collate_fn


# ==================== INFERENCE ====================
def main():
    print("=" * 60)
    print(f"IndicBART Inference  |  Language: {LANG}  |  Model: {CHECKPOINT_PATH}")
    print(f"Device: {DEVICE_STR}  |  Batch: {BATCH_SIZE}  |  Beams: {NUM_BEAMS}")
    print("=" * 60)

    # Load tokenizer & model from the saved HuggingFace checkpoint
    print("\nLoading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(
        CHECKPOINT_PATH, use_fast=False, keep_accents=True
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(CHECKPOINT_PATH)

    device = torch.device(DEVICE_STR if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"Model loaded | Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Dataset & dataloader
    dataset = IndicBARTInferenceDataset(
        INPUT_FILE, tokenizer, lang=LANG, max_length=MAX_INPUT_LEN
    )
    collate_fn  = make_collate_fn(pad_token_id=tokenizer.pad_token_id)
    dataloader  = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        collate_fn=collate_fn,
        shuffle=False,          # keep original order
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )

    forced_bos_token_id = dataset.forced_bos_token_id

    # Collect outputs indexed by record position
    outputs_map = {}   # record_idx -> generated string

    print("\nRunning inference...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Batches"):
            input_ids      = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            record_idxs    = batch["record_idxs"]

            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                # Force target language token at decoder start — same as training
                forced_bos_token_id=forced_bos_token_id,
                max_new_tokens=MAX_NEW_TOKENS,
                num_beams=NUM_BEAMS,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )

            decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            for idx, text in zip(record_idxs, decoded):
                outputs_map[idx] = text.strip()

    # Write output — original record + new key indicbart_op
    os.makedirs(os.path.dirname(OUTPUT_FILE) or ".", exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        for i, rec in enumerate(dataset.records):
            out_rec = dict(rec)
            out_rec["indicbart_op"] = outputs_map.get(i, "")
            out_f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

    print(f"\nDone! Output written to : {OUTPUT_FILE}")
    print(f"Total records processed : {len(dataset.records)}")

    # ── Spot-check ────────────────────────────────────────────────────────────
    print("\nSpot-check (first 5 outputs):")
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        for _ in range(5):
            line = f.readline()
            if not line:
                break
            obj = json.loads(line)
            src = clean_tagged_text(obj.get("tagged_output") or obj.get("input") or "")
            out = obj.get("indicbart_op", "")
            print(f"  IN : {src[:80]}")
            print(f"  OUT: {out[:80]}")
            print()

    print("=" * 60)
    print("Inference finished successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()