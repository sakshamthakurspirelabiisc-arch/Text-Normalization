import torch
from torch.utils.data import DataLoader,Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import json
import re
import os
import string
import ast
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-it")

TRAIN_FILE = "/raid/home/rizwank/Normalization/data_generation/Post_processed_Data/HINDI/train_24feb.txt"
OUTPUT_DIR = "/raid/home/rizwank/Normalization/model_building/weights_lora_hi_24_feb.txt"
MAX_LEN = 512
EPOCHS = 2
LR = 2e-4
BATCH_SIZE = 8
GRAD_ACC = 8


INSTRUCTION = (
    "You are a text normalization system. "
    "Convert the unnormalised sentence into a natural spoken Hindi sentence. "
    "Normalize all numbers into spoken hindi words. "
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

    ##text = remove_punct_and_digits_outside_tags(text)

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
                if "tagged_output" not in obj or "normalized_output" not in obj:
                    continue

                # ---- RAW DATA ----
                raw_input = obj["tagged_output"]
                
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


lora_config = LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)
model.train()

# =======================
# DATASET & DATALOADER
# =======================
#dataset = PromptDataset("train.txt", tokenizer, max_len=MAX_LEN)
def collate_fn(batch):
    input_ids = pad_sequence([b["input_ids"] for b in batch], batch_first=True, padding_value=tokenizer.pad_token_id)
    labels = pad_sequence([b["labels"] for b in batch], batch_first=True, padding_value=-100)
    attention_mask = pad_sequence([b["attention_mask"] for b in batch], batch_first=True, padding_value=0)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask
    }
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,collate_fn=collate_fn)
device="cuda:4"
# =======================
# OPTIMIZER + SCHEDULER
# =======================
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
total_steps = len(dataloader) * EPOCHS // GRAD_ACC
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=total_steps
)
model=model.to(device)

# =======================
# TRAINING LOOP
# =======================
model.train()
for epoch in range(EPOCHS):
    loop = tqdm(dataloader, leave=True)
    optimizer.zero_grad()

    for step, batch in enumerate(loop):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss / GRAD_ACC
        loss.backward()

        if (step + 1) % GRAD_ACC == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        loop.set_description(f"Epoch {epoch+1}")
        loop.set_postfix(loss=loss.item() * GRAD_ACC)

    # =======================
    # SAVE AFTER EACH EPOCH
    # =======================
    epoch_dir = os.path.join(OUTPUT_DIR, f"epoch_{epoch+1}")
    os.makedirs(epoch_dir, exist_ok=True)

    model.save_pretrained(epoch_dir)
    tokenizer.save_pretrained(epoch_dir)

    print(f" Saved LoRA weights for epoch {epoch+1} at {epoch_dir}")

# SAVE LoRA WEIGHTS
# =======================


print("LoRA fine-tuning complete.")
