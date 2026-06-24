import os
import re
import json
import torch
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import string
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup
)
import ast
from peft import PeftModel


# ==========================
# PATHS & HYPERPARAMETERS
# ==========================
TRAIN_FILE = "/raid/home/rizwank/kannada_train.txt"

TEACHER_LORA_WEIGHTS = "/raid/home/rizwank/Normalization/model_building/weights_1b_gemma_kannada/epoch_2"

OUTPUT_DIR = "/raid/home/rizwank/Normalization/model_building/distilled_gemma_270m_kan"

MAX_LEN = 512
EPOCHS = 2
LR = 2e-4
BATCH_SIZE = 4
GRAD_ACC = 8

TEMPERATURE = 2.0  # Reduced from 4.0 (typical range: 1-3)
ALPHA = 0.5        # Balanced weight (was 0.6)

device = "cuda:1"


# ==========================
# TOKENIZER
# ==========================
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")


# ==========================
# DATA PREP
# ==========================
#INSTRUCTION = (
 #   "You are a text normalization system. "
  #  "Convert the unnormalised sentence into a natural spoken Hindi sentence. "
   # "Normalize all numbers into spoken Hindi words. "
   # "Keep the meaning unchanged."
#)

INSTRUCTION = (
    "You are a text normalization system. "
    "Convert the unnormalised sentence into a natural spoken Kannada sentence. "
    "Normalize all numbers into spoken kannada words. "
    "Keep the meaning unchanged."
)

# =======================
# DATASET
# =======================

def clean_tagged_text(text):
    if text is None:
        return ""
    text = str(text)  # ensure it's a string
    
    # Remove numbered prefixes at start
    text = re.sub(r"^(?:\s*[0-9]\.\s*)+", "", text)

    # Remove punctuation and digits outside tags
    def remove_punct_and_digits_outside_tags(t):
        result = []
        in_tag = False
        for char in t:
            if char == '<':
                in_tag = True
                result.append(char)
            elif char == '>':
                in_tag = False
                result.append(char)
            elif not in_tag and (char in string.punctuation or char.isdigit()):
                continue
            else:
                result.append(char)
        return ''.join(result)

    text = remove_punct_and_digits_outside_tags(text)

    # Remove tags like <DATE>, <CARDINAL>
    text = re.sub(r"<[A-Z]+>", "", text)

    # Remove angle brackets but keep content inside
    text = re.sub(r"<([^>]+)>", r"\1", text)

    return text.strip()

def remove_tags(text: str) -> str:
    """
    Removes semantic tags like <TIME>, <DATE> etc.
    Keeps inner text and removes angle brackets.
    """
    text = re.sub(r"<\s*[A-Z_]+\s*>", "", text)  # Remove semantic tags
    text = text.replace("<", "").replace(">", "")  # Remove remaining brackets
    text = re.sub(r"\s+", " ", text).strip()      # Normalize spaces
    return text


def parse_line(line):
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(line)
        except Exception:
            return None

class PromptDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.samples = []
        self.max_len = max_len

        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if not line.strip():
                    continue

                obj = parse_line(line)
                if obj is None:
                    continue

                # Required fields
                if "translated_tagged" not in obj or "normalized_output" not in obj:
                    continue

                # ---- RAW DATA ----
                raw_input = obj["translated_tagged"]
                raw_output = obj["normalized_output"]

                # ---- CLEAN DATA ----
                input_text = clean_tagged_text(raw_input)
                output_text = clean_tagged_text(raw_output)

                if not input_text.strip() or not output_text.strip():
                    continue

                # ---- Same instruction prompt ----
                prompt_text = (
                    "### Instruction:\n"
                    f"{INSTRUCTION}\n\n"
                    "### Input:\n"
                    f"{input_text}\n\n"
                )

                # ---- Tokenize prompt ----
                tokenized_prompt = tokenizer(
                    prompt_text,
                    truncation=True,
                    max_length=max_len,
                    add_special_tokens=True,
                )

                # ---- Tokenize output ----
                tokenized_output = tokenizer(
                    output_text,
                    truncation=True,
                    max_length=max_len,
                    add_special_tokens=True,
                )

                input_ids = tokenized_prompt["input_ids"] + tokenized_output["input_ids"]
                attention_mask = [1] * len(input_ids)

                # ---- Mask prompt tokens ----
                labels = [-100] * len(tokenized_prompt["input_ids"]) + tokenized_output["input_ids"]

                self.samples.append({
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                })

        print(f" Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]



dataset=PromptDataset(TRAIN_FILE,tokenizer)
def collate_fn(batch):
    input_ids = pad_sequence(
        [b["input_ids"] for b in batch],
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )
    labels = pad_sequence(
        [b["labels"] for b in batch],
        batch_first=True,
        padding_value=-100,
    )
    attention_mask = pad_sequence(
        [b["attention_mask"] for b in batch],
        batch_first=True,
        padding_value=0,
    )
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }



dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)


# ==========================
# LOAD TEACHER (1B + LoRA)
# ==========================
print("Loading teacher model (Gemma-3-1B + LoRA)…")

teacher_base = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-it")
teacher = PeftModel.from_pretrained(teacher_base, TEACHER_LORA_WEIGHTS)

teacher = teacher.to(device)
teacher.eval()

for p in teacher.parameters():
    p.requires_grad = False


# ==========================
# LOAD STUDENT (270M)
# ==========================
print("Loading student model (Gemma-3-270M)…")

student = AutoModelForCausalLM.from_pretrained("google/gemma-3-270m-it")
student = student.to(device)
student.train()

optimizer = torch.optim.AdamW(student.parameters(), lr=LR)

total_steps = len(dataloader) * EPOCHS // GRAD_ACC
scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)


# ==========================
# TRAINING (DISTILLATION)
# ==========================
print(f"\nStarting distillation training...")
print(f"Temperature: {TEMPERATURE}, Alpha: {ALPHA}")
print(f"Total steps: {total_steps}\n")

for epoch in range(EPOCHS):
    loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    optimizer.zero_grad()
    
    epoch_ce_loss = 0.0
    epoch_kd_loss = 0.0
    epoch_total_loss = 0.0
    num_batches = 0

    for step, batch in enumerate(loop):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # ---- Teacher forward (no grad) ----
        with torch.no_grad():
            teacher_logits = teacher(
                input_ids=input_ids,
                attention_mask=attention_mask
            ).logits.detach()

        # ---- Student forward ----
        student_out = student(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        student_logits = student_out.logits

        # Cross-entropy loss (real labels)
        ce_loss = student_out.loss

        # ---- Distillation loss (KL divergence with masking) ----
        T = TEMPERATURE
        
        # Create mask for valid positions (where labels != -100)
        valid_mask = (labels != -100)
        
        # Flatten tensors
        batch_size, seq_len, vocab_size = student_logits.shape
        
        # Reshape to (batch_size * seq_len, vocab_size)
        teacher_logits_flat = teacher_logits.view(-1, vocab_size)
        student_logits_flat = student_logits.view(-1, vocab_size)
        valid_mask_flat = valid_mask.view(-1)
        
        # Select only valid positions
        if valid_mask_flat.sum() > 0:
            teacher_logits_valid = teacher_logits_flat[valid_mask_flat]
            student_logits_valid = student_logits_flat[valid_mask_flat]
            
            # Compute softmax with temperature
            teacher_probs = torch.nn.functional.softmax(teacher_logits_valid / T, dim=-1)
            student_log_probs = torch.nn.functional.log_softmax(student_logits_valid / T, dim=-1)
            
            # KL divergence loss
            kd_loss = torch.nn.functional.kl_div(
                student_log_probs,
                teacher_probs,
                reduction='batchmean'
            ) * (T * T)
        else:
            kd_loss = torch.tensor(0.0, device=device)

        # ---- Combine losses ----
        loss = (1 - ALPHA) * ce_loss + ALPHA * kd_loss
        loss = loss / GRAD_ACC

        loss.backward()

        # Track losses for logging
        epoch_ce_loss += ce_loss.item()
        epoch_kd_loss += kd_loss.item()
        epoch_total_loss += (loss.item() * GRAD_ACC)
        num_batches += 1

        # Gradient accumulation step
        if (step + 1) % GRAD_ACC == 0 or (step + 1) == len(dataloader):
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Update progress bar
        loop.set_postfix({
            'loss': f"{loss.item() * GRAD_ACC:.4f}",
            'ce': f"{ce_loss.item():.4f}",
            'kd': f"{kd_loss.item():.4f}"
        })

    # Epoch summary
    avg_ce = epoch_ce_loss / num_batches
    avg_kd = epoch_kd_loss / num_batches
    avg_total = epoch_total_loss / num_batches
    
    print(f"\nEpoch {epoch+1} Summary:")
    print(f"  Avg Total Loss: {avg_total:.4f}")
    print(f"  Avg CE Loss: {avg_ce:.4f}")
    print(f"  Avg KD Loss: {avg_kd:.4f}")

    # Save checkpoint each epoch
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, f"epoch_{epoch+1}")
    student.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    print(f" Saved student checkpoint at {save_path}\n")

print(" Knowledge distillation finished successfully!")