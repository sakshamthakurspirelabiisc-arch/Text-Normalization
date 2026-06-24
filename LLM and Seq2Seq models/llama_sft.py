import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import json
import re
import os
import string
import ast

# =======================
# CONFIG
# =======================
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
TRAIN_FILE = "/raid/home/rizwank/kannada_train.txt"
OUTPUT_DIR = "/raid/home/rizwank/Normalization/model_building/lora_llama_kan_updated"
EXISTING_LORA_PATH = "/raid/home/rizwank/Normalization/model_building/weights_lora_llama_kan/epoch_1"  # Load existing LoRA
MAX_LEN = 512
EPOCHS = 1  # Just 1 more epoch
LR = 2e-5  # Lower learning rate for continued training (2e-5 instead of 2e-4)
BATCH_SIZE = 8
GRAD_ACC = 8
DEVICE = "cuda:5"

# Instruction
INSTRUCTION = (
     "You are a text normalization system. "
     "Convert the unnormalized sentence into a natural spoken Kannada sentence. "
     "Normalize all numbers into spoken Kannada words. "
     "Keep the meaning unchanged."
 )

# =======================
# LOAD TOKENIZER & BASE MODEL
# =======================
print("=" * 60)
print("🦙 LOADING TOKENIZER AND BASE MODEL")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
)

# =======================
# LOAD EXISTING LORA WEIGHTS
# =======================
print(f" Loading existing LoRA from: {EXISTING_LORA_PATH}")
print("=" * 60)

# Load the existing LoRA weights
model = PeftModel.from_pretrained(base_model, EXISTING_LORA_PATH)
model = model.to(DEVICE)

# ✅ CRITICAL FIX: Enable training mode and LoRA layers
model.train()  # Set to training mode

# Disable inference mode for all adapters
for adapter_name in model.peft_config:
    model.peft_config[adapter_name].inference_mode = False

# Ensure LoRA parameters are trainable
for name, param in model.named_parameters():
    if 'lora' in name.lower():
        param.requires_grad = True

print("✅ Existing LoRA loaded and configured for training!")
model.print_trainable_parameters()

# Debug: Check LoRA layer status
print("\n🔍 Verifying LoRA layer status:")
lora_params = 0
total_params = 0
for name, param in model.named_parameters():
    total_params += param.numel()
    if 'lora' in name.lower():
        lora_params += param.numel()
        print(f"  ✓ {name}: requires_grad={param.requires_grad}, shape={param.shape}")

print(f"\n📊 Statistics:")
print(f"  Total LoRA parameters: {lora_params:,}")
print(f"  Total model parameters: {total_params:,}")
if total_params > 0:
    print(f"  Trainable percentage: {100 * lora_params / total_params:.2f}%")

if lora_params == 0:
    print("\ WARNING: No LoRA parameters found! Check if the LoRA path is correct.")
    exit(1)

# =======================
# HELPERS
# =======================

def clean_tagged_text(text):
    if text is None:
        return ""
    text = str(text)
    text = re.sub(r"^(?:\s*[0-9]\.\s*)+", "", text)
    text = re.sub(r"<[A-Z_]+>", "", text)
    text = re.sub(r"<([^>]+)>", r"\1", text)
    return text.strip()

def parse_line(line):
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(line)
        except Exception:
            return None

# =======================
# DATASET
# =======================

class PromptDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.samples = []
        self.max_len = max_len

        print("\n📖 Loading training data...")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                obj = parse_line(line)
                if obj is None:
                    continue

                if "translated_tagged" not in obj or "normalized_output" not in obj:
                    continue

                input_text = clean_tagged_text(obj["translated_tagged"])
                output_text = clean_tagged_text(obj["normalized_output"])

                if not input_text.strip() or not output_text.strip():
                    continue

                # LLaMA chat-style prompt
                prompt_text = (
                    f"<|begin_of_text|>"
                    f"<|start_header_id|>system<|end_header_id|>\n{INSTRUCTION}<|eot_id|>"
                    f"<|start_header_id|>user<|end_header_id|>\n{input_text}<|eot_id|>"
                    f"<|start_header_id|>assistant<|end_header_id|>\n"
                )

                tokenized_prompt = tokenizer(
                    prompt_text,
                    truncation=True,
                    max_length=max_len,
                    add_special_tokens=False,
                )

                tokenized_output = tokenizer(
                    output_text + "<|eot_id|>",
                    truncation=True,
                    max_length=max_len,
                    add_special_tokens=False,
                )

                input_ids = tokenized_prompt["input_ids"] + tokenized_output["input_ids"]
                attention_mask = [1] * len(input_ids)
                labels = [-100] * len(tokenized_prompt["input_ids"]) + tokenized_output["input_ids"]

                # Skip if combined length exceeds max_len
                if len(input_ids) > max_len:
                    continue

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

# =======================
# DATALOADER
# =======================

dataset = PromptDataset(TRAIN_FILE, tokenizer, max_len=MAX_LEN)

def collate_fn(batch):
    input_ids = pad_sequence([b["input_ids"] for b in batch], batch_first=True,
                              padding_value=tokenizer.pad_token_id)
    labels = pad_sequence([b["labels"] for b in batch], batch_first=True,
                           padding_value=-100)
    attention_mask = pad_sequence([b["attention_mask"] for b in batch], batch_first=True,
                                   padding_value=0)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# =======================
# OPTIMIZER + SCHEDULER (with lower LR for continued training)
# =======================
print("\n" + "=" * 60)
print("⚙️ SETTING UP OPTIMIZER (lower LR for fine-tuning)")
print("=" * 60)

# Only optimize trainable parameters (LoRA layers)
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=0.01)

total_steps = len(dataloader) * EPOCHS // GRAD_ACC
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.03 * total_steps),
    num_training_steps=total_steps
)

# =======================
# TRAINING LOOP (1 additional epoch)
# =======================
print("\n" + "=" * 60)
print(f"CONTINUING TRAINING FOR {EPOCHS} MORE EPOCH")
print(f"   Starting from existing LoRA: {EXISTING_LORA_PATH}")
print("=" * 60)

model.train()
start_epoch = 2  # Since we already have epoch_1, this will be epoch_2

for epoch in range(start_epoch, start_epoch + EPOCHS):
    loop = tqdm(dataloader, leave=True)
    optimizer.zero_grad()
    total_loss = 0
    step_count = 0

    for step, batch in enumerate(loop):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss / GRAD_ACC
        loss.backward()

        if (step + 1) % GRAD_ACC == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * GRAD_ACC
        step_count += 1
        avg_loss = total_loss / step_count

        loop.set_description(f"Epoch {epoch}")
        loop.set_postfix(loss=loss.item() * GRAD_ACC, avg_loss=avg_loss)

    # Save the updated model after this epoch
    epoch_dir = os.path.join(OUTPUT_DIR, f"epoch_{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)
    model.save_pretrained(epoch_dir)
    tokenizer.save_pretrained(epoch_dir)
    print(f"\n💾 Saved LoRA weights for epoch {epoch} at {epoch_dir}")
    print(f"   Average loss for epoch {epoch}: {avg_loss:.4f}")

print("\n" + "=" * 60)
print(" LoRA fine-tuning complete!")
print(f" Updated weights saved to: {OUTPUT_DIR}/epoch_2")
print("=" * 60)

# =======================
# OPTIONAL: Quick validation test
# =======================
print("\n🔍 Running quick validation test...")
model.eval()

test_cases = [
    "9/22",
    "47/112",
    "48/66",
    "4/4",
    "3/30"
]

for test_input in test_cases:
    test_prompt = (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n{INSTRUCTION}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n{test_input}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
    )

    inputs = tokenizer(test_prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the assistant's response
    if "assistant" in generated:
        response = generated.split("assistant")[-1].strip()
    else:
        response = generated.strip()
    
    print(f"Input: {test_input:10} → Output: {response}")

print("\n" + "=" * 60)
print(" Validation complete!")
print("=" * 60)