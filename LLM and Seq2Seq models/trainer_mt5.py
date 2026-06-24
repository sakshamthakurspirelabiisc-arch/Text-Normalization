import json
import os
from transformers import MT5Tokenizer, MT5ForConditionalGeneration
from torch.nn.utils import clip_grad_norm_
import torch
from torch.utils.data import Dataset ,DataLoader
import json
import re
from torch.optim import AdamW
from transformers import get_scheduler
import re
from Levenshtein import distance as levenshtein_distance
from tqdm import tqdm

from torch.nn.utils import clip_grad_norm_


import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

nltk.download('punkt_tab')
model_name = "google/mt5-small"  # or mt5-base, mt5-large if resources permit
tokenizer = MT5Tokenizer.from_pretrained(model_name)
model = MT5ForConditionalGeneration.from_pretrained(model_name)
import re
import string





def clean_tagged_text_1(text):
    # 1. Remove prefix like 1. or 0. 0. etc. at start (with spaces allowed)
    text = re.sub(r"^(?:\s*[0-9]\.\s*)+", "", text)

    # 2. Remove punctuation and digits not inside <...>
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
                continue  # skip punctuation or digit outside tags
            else:
                result.append(char)
        return ''.join(result)

    text = remove_punct_and_digits_outside_tags(text)

    # 3. Remove tags like <DATE>, <CARDINAL>
    text = re.sub(r"<[A-Z]+>", "", text)

    # 4. Remove angle brackets but keep content inside
    text = re.sub(r"<([^>]+)>", r"\1", text)

    return text.strip()
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

import json
import ast

def parse_line(line):
    try:
        # Try JSON first (double quotes)
        return json.loads(line)
    except json.JSONDecodeError:
        # If that fails, fallback to ast.literal_eval (single quotes)
        return ast.literal_eval(line)


class KannadaNormalizationDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_length=128):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(filepath, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                obj = parse_line(line)
                if obj is None:
                    continue  # skip unparsable lines

                # Skip if required keys missing
                if "translated_tagged" not in obj or "normalized_output" not in obj:
                    print(f"Skipping line {idx} due to missing keys")
                    continue

                input_text = clean_tagged_text(obj["translated_tagged"])
                label_text = clean_tagged_text(obj["normalized_output"])

                if not input_text or not label_text:
                    continue  # skip empty cleaned lines

                # Tokenize input
                inputs = tokenizer(
                    input_text,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )["input_ids"]

                # Tokenize target
                targets = tokenizer(
                    text_target=label_text,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )["input_ids"]

                self.data.append({
                    "input_ids": inputs,
                    "labels": targets
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]["input_ids"], self.data[idx]["labels"]



def collate_function(batch):
    # Unpack batch of 2D tensors: (1, seq_len)
    inputs, targets = zip(*batch)  # Each: Tensor of shape (1, seq_len)

    # Add 0 at the beginning along dim=1 (sequence length)
    #inputs = [torch.cat([torch.tensor([[0]]), inp], dim=1) for inp in inputs]    # Shape: (1, seq_len+1)
    #targets = [torch.cat([torch.tensor([[0]]), tgt], dim=1) for tgt in targets]  # Shape: (1, seq_len+1)

    # Determine max sequence lengths
    max_input_len = max(x.size(1) for x in inputs)
    max_target_len = max(y.size(1) for y in targets)

    # Pad inputs with 0 (pad token)
    padded_inputs = [
        torch.cat([x, torch.zeros(1, max_input_len - x.size(1), dtype=torch.long)], dim=1)
        for x in inputs
    ]

    # Pad targets with -100 (to be ignored in loss)
    padded_targets = [
        torch.cat([y, torch.full((1, max_target_len - y.size(1)), -100, dtype=torch.long)], dim=1)
        for y in targets
    ]

    # Stack into batch tensors
    batch_inputs = torch.cat(padded_inputs, dim=0)  # Shape: (batch_size, seq_len)
    batch_targets = torch.cat(padded_targets, dim=0)

    return batch_inputs,batch_targets
train_dataset=KannadaNormalizationDataset("/home/sakshamt/SPIRE_TN/train_kannada.txt",tokenizer)
#val_dataset=KannadaNormalizationDataset("/home/sakshamt/SPIRE_TN/valid.txt",tokenizer)
device = "cuda:2"

train_dataloader=DataLoader(train_dataset,batch_size=16,collate_fn=collate_function)
#val_dataloader=DataLoader(val_dataset,batch_size=16,collate_fn=collate_function)
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from transformers import AdamW, get_scheduler


model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 3
num_training_steps = len(train_dataloader) * num_epochs

lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

use_amp = torch.cuda.is_available()
scaler = torch.cuda.amp.GradScaler() if use_amp else None

#  Checkpoint directory and state
checkpoint_dir = "./checkpoints_kan"
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_history = []  # keep last 3 checkpoint filenames

# Evaluation function
def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    count = 0
    with torch.no_grad():
        for input_ids, labels in tqdm(dataloader, desc="Validating", leave=False):
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            outputs = model(input_ids=input_ids, labels=labels)
            total_loss += outputs.loss.item()
            count += 1
    avg_loss = total_loss / count
    print(f"🧪 Validation loss: {round(avg_loss, 4)}")
    model.train()

#  Training loop
step = 72000
for epoch in range(num_epochs):
    loop = tqdm(train_dataloader, desc=f"Epoch {epoch+1}", leave=True)
    total_loss = 0.0
    num_batches = 0

    for input_ids, labels in loop:
        step += 1
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        #with torch.cuda.amp.autocast(enabled=use_amp):
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss

        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        lr_scheduler.step()

        total_loss += loss.item()
        num_batches += 1
        avg_loss = total_loss / num_batches

        loop.set_postfix({
            "curr_loss": round(loss.item(), 4),
            "avg_loss": round(avg_loss, 4),
            "lr": lr_scheduler.get_last_lr()[0]
        })

        #  Evaluate every 2000 steps
       # if step % 8000 == 0:
        #    evaluate(model, val_dataloader)

        # Save checkpoint every 4000 steps
        if step % 10000 == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"checkpoint-step{step}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"💾 Saved checkpoint: {ckpt_path}")
            checkpoint_history.append(ckpt_path)

            #  Delete older checkpoints if > 3
            if len(checkpoint_history) > 1:
                old_ckpt = checkpoint_history.pop(0)
                if os.path.exists(old_ckpt):
                    os.remove(old_ckpt)
                    print(f" Deleted old checkpoint: {old_ckpt}")
