"""
MT5-Large + LoRA fine-tuning for Text Normalization
-------------------------------------------------
Input  : tagged_output  (unnormalized, possibly code-mixed Kannada/Hindi/English)
Output : normalized_output (normalized text)

Requirements:
    pip install transformers peft torch tqdm sentencepiece
"""

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
from transformers import (
    AutoTokenizer,
    MT5ForConditionalGeneration,
    get_scheduler,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
)


# ==================== CONFIGURATION ====================
MODEL_NAME     = "google/mt5-large"        # 1.2B parameters
TRAIN_FILE     = "/raid/home/rizwank/kannada_train.txt"
CHECKPOINT_DIR = "/raid/home/rizwank/Normalization/model_building/mt5_large_kan"

LANG           = "kn"           # 'kn' Kannada, 'hi' Hindi
DEVICE_STR     = "cuda:0"
BATCH_SIZE     = 8          # Reduce to 4 if OOM even with LoRA
NUM_EPOCHS     = 3
LEARNING_RATE  = 3e-4           # LoRA trains best with higher LR than full fine-tune
SAVE_STEPS     = 10_000
MAX_LENGTH     = 128
MAX_CKPTS      = 2              # Rolling window — keeps disk usage low

# ── LoRA hyperparameters ──────────────────────────────────────────────────────
# MT5/T5 attention projections are named q, v (key names inside each layer).
# LoRA injects trainable low-rank matrices into these projections only,
# leaving the rest of the ~3.7B parameters frozen.
# LoRA settings for exactly ~50M trainable parameters on MT5-Large:
# r=128, [q,k,v,o] × 48 layers × 2 × 1024 × 128 = 50,331,648 (~50.3M params, ~8.6% of model)
# Adding FFN modules (wi_0, wi_1) would push to 113M — so we keep attention-only for precision
LORA_R          = 128           # Rank 128 — required on MT5-Large (d_model=1024) to reach 50M
LORA_ALPHA      = 256           # Scaling factor = 2 * r  (standard rule of thumb)
LORA_DROPOUT    = 0.1
LORA_TARGET_MODULES = ["q", "k", "v", "o"]  # 4 attn projections × r=128 = exactly 50.3M params
# ─────────────────────────────────────────────────────────────────────────────


# ==================== CLEANING ====================
def clean_tagged_text(text):
    """
    Remove XML-style tags from tagged_output while preserving actual content.
    Digits and punctuation inside the text are kept — normalization input
    like '9845012345' or '₹500' must reach the model intact.
    """
    if text is None:
        return ""
    text = str(text).strip()

    # Remove list prefixes like "1. " at the very start
    text = re.sub(r"^\s*\d+\.\s+", "", text)

    # Strip all-caps tags: <DIGIT>, <CURR>, <DATE> etc.
    text = re.sub(r"<[A-Z_]+>", "", text)

    # Unwrap any remaining tags, keeping inner text
    text = re.sub(r"<([^>]+)>", r"\1", text)

    # Collapse multiple spaces
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
class NormalizationDataset(Dataset):
    """
    JSONL dataset where each line has:
        "tagged_output"    → unnormalized input (with XML tags stripped at load time)
        "normalized_output" → normalized reference

    MT5 does not use language tokens — the model is purely multilingual
    and handles Kannada, Hindi, English, and code-mixed text natively.
    A task prefix is prepended to the input so MT5 knows what to do,
    following the original T5/mT5 training convention.
    """

    INPUT_KEYS  = ["translated_tagged",    "input",  "source", "unnormalized"]
    OUTPUT_KEYS = ["normalized_output", "output", "target", "normalized"]

    TASK_PREFIX = "normalize: "   # MT5 task prefix — helps the model identify the task

    def __init__(self, filepath, tokenizer, max_length=128):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.data       = []

        print(f"Loading data from: {filepath}")
        line_count = skipped_count = 0

        with open(filepath, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line_count += 1
                obj = parse_line(line)

                if obj is None:
                    skipped_count += 1
                    continue

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

                # Prepend task prefix
                input_text = self.TASK_PREFIX + input_text

                enc = tokenizer(
                    input_text,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                dec = tokenizer(
                    text_target=label_text,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )

                self.data.append({
                    "input_ids":      enc["input_ids"].squeeze(0),
                    "attention_mask": enc["attention_mask"].squeeze(0),
                    "labels":         dec["input_ids"].squeeze(0),
                })

                if (idx + 1) % 10_000 == 0:
                    print(f"  Processed {idx + 1} lines | kept {len(self.data)} ...")

        print(f"Total: {line_count} | Kept: {len(self.data)} | Skipped: {skipped_count}")

    @staticmethod
    def _get_value(obj, keys):
        for k in keys:
            if k in obj:
                return obj[k]
        return None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ==================== COLLATE ====================
def make_collate_fn(pad_token_id):
    def collate_fn(batch):
        input_ids_list      = [item["input_ids"]      for item in batch]
        attention_mask_list = [item["attention_mask"]  for item in batch]
        labels_list         = [item["labels"]          for item in batch]

        padded_inputs = torch.nn.utils.rnn.pad_sequence(
            input_ids_list, batch_first=True, padding_value=pad_token_id
        )
        padded_masks = torch.nn.utils.rnn.pad_sequence(
            attention_mask_list, batch_first=True, padding_value=0
        )
        # -100 is ignored by cross-entropy loss
        padded_labels = torch.nn.utils.rnn.pad_sequence(
            labels_list, batch_first=True, padding_value=-100
        )

        return {
            "input_ids":      padded_inputs,
            "attention_mask": padded_masks,
            "labels":         padded_labels,
        }
    return collate_fn


# ==================== LORA SETUP ====================
def apply_lora(model):
    """
    Wrap the MT5 model with LoRA adapters.

    Attention projections (q,k,v,o) only — FFN modules excluded to stay at ~50M
    receive trainable LoRA weights. Everything else stays frozen.

    Trainable parameter count with these settings:
        r=128, alpha=256, targets=[q,k,v,o]
        48 layers × 4 modules × 2 × 1024 × 128 = 50,331,648
        MT5-Large (~582M total) → ~50.3M trainable params (~8.6%)
    """
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,    # Seq2Seq — correct for MT5
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",                         # Don't train bias terms
    )

    model = get_peft_model(model, lora_config)

    # Print trainable vs frozen parameter counts
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"\nLoRA applied:")
    print(f"  Trainable params : {trainable:,}  ({100 * trainable / total:.2f}%)")
    print(f"  Frozen params    : {total - trainable:,}")
    print(f"  Total params     : {total:,}")

    return model


# ==================== TRAINING ====================
def main():
    print("=" * 60)
    print(f"MT5-Large + LoRA  |  Task: Text Normalization  |  Lang: {LANG}")
    print(f"Model       : {MODEL_NAME}")
    print(f"LoRA rank   : {LORA_R}  |  alpha: {LORA_ALPHA}  |  targets: {LORA_TARGET_MODULES}")
    print(f"Device      : {DEVICE_STR}  |  Batch: {BATCH_SIZE}  |  LR: {LEARNING_RATE}")
    print("=" * 60)

    # ── Tokenizer & base model ────────────────────────────────────────────────
    print(f"\nLoading tokenizer and base model: {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = MT5ForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,   # Full fp32 — avoids NaN issues
    )

    # ── Apply LoRA ────────────────────────────────────────────────────────────
    model = apply_lora(base_model)

    # ── Dataset & dataloader ──────────────────────────────────────────────────
    print("\nLoading dataset...")
    train_dataset = NormalizationDataset(TRAIN_FILE, tokenizer, max_length=MAX_LENGTH)
    collate_fn    = make_collate_fn(pad_token_id=tokenizer.pad_token_id)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    print(f"Examples: {len(train_dataset):,}  |  Batches/epoch: {len(train_dataloader):,}")

    # ── Device ───────────────────────────────────────────────────────────────
    device = torch.device(DEVICE_STR)
    model.to(device)
    model.train()

    # ── Optimizer — only optimizes LoRA params (frozen params have no grad) ──
    # Using higher LR than full fine-tuning since LoRA adapters are small
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )

    num_training_steps = len(train_dataloader) * NUM_EPOCHS
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=int(0.05 * num_training_steps),  # 5% warmup
        num_training_steps=num_training_steps,
    )

    # ── Checkpoint dir & config ───────────────────────────────────────────────
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    config = {
        "model_name":    MODEL_NAME,
        "lang":          LANG,
        "lora_r":        LORA_R,
        "lora_alpha":    LORA_ALPHA,
        "lora_dropout":  LORA_DROPOUT,
        "lora_targets":  LORA_TARGET_MODULES,
        "batch_size":    BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "num_epochs":    NUM_EPOCHS,
        "max_length":    MAX_LENGTH,
        "train_file":    TRAIN_FILE,
        "total_examples": len(train_dataset),
        "total_steps":   num_training_steps,
    }
    with open(os.path.join(CHECKPOINT_DIR, "training_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nConfig saved → {CHECKPOINT_DIR}/training_config.json")

    def save_lora_checkpoint(step_or_tag, epoch, global_step, avg_loss):
        """
        Save only the LoRA adapter weights (tiny — ~hundreds of MB vs ~2.3GB for full MT5-Large).
        The base model stays unchanged and is loaded separately at inference.
        """
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"lora-{step_or_tag}")
        model.save_pretrained(ckpt_path)       # saves adapter_config.json + adapter_model.bin
        tokenizer.save_pretrained(ckpt_path)   # saves tokenizer files for convenience
        # Also save training state for resuming
        torch.save({
            "epoch":      epoch,
            "step":       global_step,
            "loss":       avg_loss,
            "optimizer":  optimizer.state_dict(),
            "scheduler":  lr_scheduler.state_dict(),
            "config":     config,
        }, os.path.join(ckpt_path, "training_state.pt"))
        return ckpt_path

    # ── Training loop ─────────────────────────────────────────────────────────
    print("\nStarting training...\n")
    global_step        = 0
    checkpoint_history = []
    avg_loss           = 0.0

    for epoch in range(NUM_EPOCHS):
        print(f"\n── Epoch {epoch + 1}/{NUM_EPOCHS} ──")
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

            # Full fp32 forward — no AMP to avoid NaN
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

            # NaN guard
            if not torch.isfinite(loss):
                print(f"\n  WARNING: Non-finite loss at step {global_step}, skipping batch.")
                optimizer.zero_grad()
                continue

            loss.backward()
            # Clip only trainable (LoRA) gradients
            clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=1.0
            )
            optimizer.step()
            lr_scheduler.step()

            total_loss  += loss.item()
            num_batches += 1
            avg_loss     = total_loss / num_batches

            loop.set_postfix({
                "loss":     f"{loss.item():.4f}",
                "avg_loss": f"{avg_loss:.4f}",
                "lr":       f"{lr_scheduler.get_last_lr()[0]:.2e}",
                "step":     global_step,
            })

            # ── Step checkpoint ───────────────────────────────────────────
            if global_step % SAVE_STEPS == 0:
                ckpt_path = save_lora_checkpoint(
                    f"step{global_step}", epoch, global_step, avg_loss
                )
                print(f"\n  Saved LoRA checkpoint: {ckpt_path}")

                checkpoint_history.append(ckpt_path)
                if len(checkpoint_history) > MAX_CKPTS:
                    import shutil
                    old = checkpoint_history.pop(0)
                    if os.path.exists(old):
                        shutil.rmtree(old)
                        print(f"  Removed old checkpoint: {old}")

        # ── Epoch checkpoint ──────────────────────────────────────────────
        ckpt_path = save_lora_checkpoint(
            f"epoch{epoch + 1}", epoch, global_step, avg_loss
        )
        print(f"\n  Saved epoch checkpoint: {ckpt_path}")
        print(f"  Epoch {epoch + 1} avg loss: {avg_loss:.4f}")

    # ── Final save ────────────────────────────────────────────────────────────
    print("\nTraining complete. Saving final LoRA adapter...")

    final_path = os.path.join(CHECKPOINT_DIR, "lora-final")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"  LoRA adapter saved → {final_path}")

    with open(os.path.join(CHECKPOINT_DIR, "training_completed.txt"), "w") as f:
        f.write(f"Completed at step {global_step}\nFinal avg loss: {avg_loss:.4f}\n")

    print("\n" + "=" * 60)
    print("Training finished!")
    print(f"Load for inference with:")
    print(f"  base  = MT5ForConditionalGeneration.from_pretrained('google/mt5-large')")
    print(f"  model = PeftModel.from_pretrained(base, '{final_path}')")
    print("=" * 60)


if __name__ == "__main__":
    main()