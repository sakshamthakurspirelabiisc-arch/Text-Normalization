import json
import os
import ast
import re
import string

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_scheduler

# ==================== INDICBART CONFIGURATION ====================
model_name = "ai4bharat/IndicBART"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, keep_accents=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Language token IDs for forcing decoder start token
LANG_CODES = {
    'kn': '<2kn>',
    'hi': '<2hi>',
    'en': '<2en>'
}


# ==================== CLEANING FUNCTIONS ====================
def clean_tagged_text(text):
    """
    Clean input/output text:
    - Remove XML-style tags like <DIGIT>, <CURR> etc.
    - Strip numbered list prefixes (e.g., "1. ")
    - Do NOT strip digits or punctuation from the actual content —
      normalization input may contain "100 km" which must be preserved.
    """
    if text is None:
        return ""
    text = str(text).strip()

    # Remove numbered prefixes at the very start (e.g., "1. ", "2. ")
    text = re.sub(r"^\s*\d+\.\s+", "", text)

    # Remove self-closing or all-caps tags like <DIGIT>, <CURR>, <NUM>
    text = re.sub(r"<[A-Z_]+>", "", text)

    # Unwrap any remaining angle-bracket tags, keeping inner content
    text = re.sub(r"<([^>]+)>", r"\1", text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ==================== PARSING ====================
def parse_line(line):
    """Try JSON first, then Python literal, return None on failure."""
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
class IndicBARTNormalizationDataset(Dataset):
    """
    Dataset for text normalization using IndicBART.

    Expected JSONL format per line:
        {"translated_tagged": "<input sentence>", "normalized_output": "<target>"}

    The input can be Hindi or Kannada (with code-mixing).
    For Kannada, text is optionally transliterated to Devanagari.

    IndicBART input format:  "<sentence> <2hi>"  (language token appended)
    IndicBART target format: "<sentence>"         (tokenizer handles lang internally)
    """

    # Keys to look for in the JSON — adapt if your data uses different field names
    INPUT_KEYS  = ["tagged_output", "input", "source", "unnormalized"]
    OUTPUT_KEYS = ["normalized_output", "output", "target", "normalized"]

    def __init__(self, filepath, tokenizer, lang='hi', max_length=128):
        self.tokenizer   = tokenizer
        self.max_length  = max_length
        self.lang        = lang
        self.lang_token  = LANG_CODES[lang]
        self.data        = []
       
        # Get the integer ID of the target language token for decoder forcing
        lang_token_id = tokenizer.convert_tokens_to_ids(self.lang_token)
        if lang_token_id == tokenizer.unk_token_id:
            raise ValueError(
                f"Language token '{self.lang_token}' not found in tokenizer vocabulary. "
                f"Make sure you are using the IndicBART tokenizer."
            )
        self.forced_bos_token_id = lang_token_id

        print(f"Loading data from: {filepath}")
        print(f"Language: {lang} | Token: {self.lang_token} | Token ID: {lang_token_id}")

        line_count = skipped_count = 0

        with open(filepath, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line_count += 1
                obj = parse_line(line)
                
                if obj is None:
                    skipped_count += 1
                    continue

                # Flexible key lookup
                input_text = self._get_value(obj, self.INPUT_KEYS)
                label_text = self._get_value(obj, self.OUTPUT_KEYS)

                if input_text is None or label_text is None:
                    skipped_count += 1
                    continue

                input_text = clean_tagged_text(input_text)
                label_text = clean_tagged_text(label_text)

                if not input_text or not label_text:
                    skipped_count += 1
                    continue

                # IndicBART supports Kannada natively via <2kn> — no script conversion needed.
                # IndicBART input format: sentence + space + <2xx>
                input_with_lang = f"{input_text} {self.lang_token}"

                # Tokenize input
                enc = tokenizer(
                    input_with_lang,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )

                # Tokenize target — use text_target so tokenizer sets up labels correctly
                dec = tokenizer(
                    text_target=label_text,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )

                self.data.append({
                    "input_ids":      enc["input_ids"].squeeze(0),       # (src_len,)
                    "attention_mask": enc["attention_mask"].squeeze(0),  # (src_len,)
                    "labels":         dec["input_ids"].squeeze(0),       # (tgt_len,)
                })

                if (idx + 1) % 10_000 == 0:
                    print(f"  Processed {idx + 1} lines | kept {len(self.data)} ...")

        print(f"Total lines: {line_count} | Kept: {len(self.data)} | Skipped: {skipped_count}")

    @staticmethod
    def _get_value(obj, keys):
        for k in keys:
            if k in obj:
                return obj[k]
        return None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]   # dict with input_ids, attention_mask, labels


# ==================== COLLATE FUNCTION ====================
def make_collate_fn(pad_token_id):
    """
    Returns a collate function that pads a batch of variable-length sequences.
    Labels are padded with -100 so they are ignored in the cross-entropy loss.
    """
    def collate_fn(batch):
        input_ids_list      = [item["input_ids"]      for item in batch]
        attention_mask_list = [item["attention_mask"]  for item in batch]
        labels_list         = [item["labels"]          for item in batch]

        # Pad inputs to the longest sequence in the batch
        padded_inputs = torch.nn.utils.rnn.pad_sequence(
            input_ids_list,
            batch_first=True,
            padding_value=pad_token_id
        )
        padded_masks = torch.nn.utils.rnn.pad_sequence(
            attention_mask_list,
            batch_first=True,
            padding_value=0
        )
        # Pad labels with -100 (ignored by loss)
        padded_labels = torch.nn.utils.rnn.pad_sequence(
            labels_list,
            batch_first=True,
            padding_value=-100
        )

        return {
            "input_ids":      padded_inputs,   # (B, src_len)
            "attention_mask": padded_masks,    # (B, src_len)
            "labels":         padded_labels,   # (B, tgt_len)
        }

    return collate_fn


# ==================== TRAINING ====================
def main():
    # ── Configuration ────────────────────────────────────────────────────────
    lang           = 'hi'          # 'hi' = Hindi, 'kn' = Kannada
    train_file     = "/raid/home/rizwank/Normalization/data_generation/Post_processed_Data/HINDI/train_hindi.txt"
    checkpoint_dir = "/raid/home/rizwank/Normalization/model_building/Indic_bart/indic_bart_hi"
    device_str     = "cuda:6" if torch.cuda.is_available() else "cpu"

    batch_size    = 16
    num_epochs    = 4
    learning_rate = 3e-5
    save_steps    = 10_000
    max_length    = 128
    max_ckpts     = 3       # Rolling window of checkpoints to keep
    # ─────────────────────────────────────────────────────────────────────────

    print("=" * 60)
    print(f"Training IndicBART  |  Language: {lang}  |  Model: {model_name}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {device_str}  |  Batch: {batch_size}  |  LR: {learning_rate}")
    print("=" * 60)

    # Dataset
    print("\nLoading dataset...")
    train_dataset = IndicBARTNormalizationDataset(
        train_file, tokenizer, lang=lang, max_length=max_length
    )
    collate_fn = make_collate_fn(pad_token_id=tokenizer.pad_token_id)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )
    print(f"Examples: {len(train_dataset):,}  |  Batches/epoch: {len(train_dataloader):,}")

    # Device & model
    device = torch.device(device_str)
    model.to(device)

    # Optimizer & scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = len(train_dataloader) * num_epochs
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps
    )

    # Mixed precision
    use_amp = torch.cuda.is_available()
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Decoder start token (language token) for IndicBART
    forced_bos_token_id = train_dataset.forced_bos_token_id

    # Checkpoint dir & config
    os.makedirs(checkpoint_dir, exist_ok=True)
    config = {
        'lang': lang, 'model_name': model_name,
        'batch_size': batch_size, 'learning_rate': learning_rate,
        'num_epochs': num_epochs, 'max_length': max_length,
        'train_file': train_file, 'total_examples': len(train_dataset),
        'total_steps': num_training_steps,
        'forced_bos_token_id': forced_bos_token_id
    }
    with open(os.path.join(checkpoint_dir, 'training_config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\nConfig saved → {checkpoint_dir}/training_config.json")

    def save_checkpoint(path, epoch, global_step, avg_loss):
        torch.save({
            'epoch': epoch,
            'step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict() if use_amp else None,
            'loss': avg_loss,
            'config': config
        }, path)

    # ── Training loop ─────────────────────────────────────────────────────────
    print("\nStarting training...\n")
    global_step       = 0
    checkpoint_history = []

    for epoch in range(num_epochs):
        print(f"\n── Epoch {epoch + 1}/{num_epochs} ──")
        model.train()

        total_loss  = 0.0
        num_batches = 0
        loop = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}", leave=True)

        for batch in loop:
            global_step += 1

            input_ids      = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels         = batch["labels"].to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    # Force the decoder to start with the target language token.
                    # This is critical for IndicBART — without it, the model may
                    # decode in the wrong language.
                    decoder_input_ids=torch.full(
                        (input_ids.size(0), 1),
                        fill_value=forced_bos_token_id,
                        dtype=torch.long,
                        device=device
                    )
                )
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()

            total_loss  += loss.item()
            num_batches += 1
            avg_loss     = total_loss / num_batches

            loop.set_postfix({
                "loss":     f"{loss.item():.4f}",
                "avg_loss": f"{avg_loss:.4f}",
                "lr":       f"{lr_scheduler.get_last_lr()[0]:.2e}",
                "step":     global_step
            })

            # ── Step checkpoint ───────────────────────────────────────────
            if global_step % save_steps == 0:
                ckpt_path = os.path.join(
                    checkpoint_dir, f"checkpoint-step{global_step}.pt"
                )
               # save_checkpoint(ckpt_path, epoch, global_step, avg_loss)
                print(f"\n  Saved step checkpoint: {ckpt_path}")

                checkpoint_history.append(ckpt_path)
                if len(checkpoint_history) > max_ckpts:
                    old = checkpoint_history.pop(0)
                    if os.path.exists(old):
                        os.remove(old)
                        print(f"  Removed old checkpoint: {old}")

        # ── Epoch checkpoint ──────────────────────────────────────────────
        epoch_ckpt = os.path.join(checkpoint_dir, f"checkpoint-epoch{epoch + 1}.pt")
        save_checkpoint(epoch_ckpt, epoch, global_step, avg_loss)
        print(f"\n  Saved epoch checkpoint: {epoch_ckpt}")
        print(f"  Epoch {epoch + 1} avg loss: {avg_loss:.4f}")

    # ── Final save ────────────────────────────────────────────────────────────
    print("\nTraining complete. Saving final model...")

    final_pt = os.path.join(checkpoint_dir, "final_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'final_loss': avg_loss
    }, final_pt)
    print(f"  PyTorch checkpoint → {final_pt}")

    hf_path = os.path.join(checkpoint_dir, "transformers_model")
    model.save_pretrained(hf_path)
    tokenizer.save_pretrained(hf_path)
    print(f"  HuggingFace format  → {hf_path}")

    # Save generation config so inference uses the correct language token
    gen_config_path = os.path.join(hf_path, "generation_config.json")
    with open(gen_config_path, "w") as f:
        json.dump({"forced_bos_token_id": forced_bos_token_id}, f, indent=2)
    print(f"  Generation config   → {gen_config_path}")

    with open(os.path.join(checkpoint_dir, 'training_completed.txt'), 'w') as f:
        f.write(f"Completed at step {global_step}\nFinal loss: {avg_loss:.4f}\n")

    print("\n" + "=" * 60)
    print("Training finished successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()