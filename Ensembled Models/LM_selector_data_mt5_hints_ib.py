import torch
from torch.utils.data import DataLoader,Dataset
from torch.nn.utils.rnn import pad_sequence 
import json
import random
from transformers import MT5Tokenizer, MT5ForConditionalGeneration,MT5TokenizerFast
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import MT5EncoderModel, MT5Tokenizer
from TB_RB import normalize_sent 
import ast
import os
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    DataCollatorForTokenClassification, Trainer, TrainingArguments,
    TrainerCallback
)
import evaluate
import numpy as np
from safetensors.torch import load_file
from transformers import AutoTokenizer
from indic_numtowords import num2words
import re
import json
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel, BertModel, AutoConfig, TrainingArguments, Trainer
import json
import torch
from typing import List, Dict, Optional
# rb:0,mt5:1,hints_IB 2

class_to_id = {
    "O": 0,
    "B-DATE": 1, "I-DATE": 2,
    "B-CARDINAL": 3, "I-CARDINAL": 4,
    "B-FRACTION": 5, "I-FRACTION": 6,
    "B-MONEY": 7, "I-MONEY": 8,
    "B-TELEPHONE": 9, "I-TELEPHONE": 10,
    "B-MEASURE": 11, "I-MEASURE": 12,
    "B-TIME": 13, "I-TIME": 14,
    "B-DECIMAL": 15, "I-DECIMAL": 16,
}

tokenizer = AutoTokenizer.from_pretrained("ai4bharat/IndicBERTv2-MLM-Sam-TLM")
label_list = list(class_to_id.keys())
ner_model = AutoModelForTokenClassification.from_pretrained(
    "ai4bharat/IndicBERTv2-MLM-Sam-TLM", num_labels=len(label_list)
)
weights_path = "/home/sakshamt/SPIRE_TN/TN_Models/Indic_Bert/ner_weights_kan/model.safetensors"

state_dict = load_file(weights_path)
ner_model.load_state_dict(state_dict)

ner_model.eval()
print("Loaded safetensors weights successfully!")


def get_hint_from_span(span: str, lang="en"):
    """
    Enhanced version that handles various formats better.
    """
    span = span.strip()
    if not span:
        return span, False
    
    lang_map = {'hi': 'hi', 'en': 'en', 'te': 'te', 'kan': 'kn'}
    target_lang = lang_map.get(lang, 'en')
    
    try:
        # Try to handle as pure number first
        if span.isdigit():
            return num2words(int(span), lang=target_lang), False
        
        # Handle decimal numbers
        if re.fullmatch(r"\d+\.\d+", span):
            integer, fractional = span.split(".")
            int_part = num2words(int(integer), lang=target_lang)
            frac_part = " ".join(num2words(int(d), lang=target_lang) for d in fractional)
            return f"{int_part} point {frac_part}", False
        
        # Handle fractions (e.g., "2/3")
        if re.fullmatch(r"\d+/\d+", span):
            numerator, denominator = span.split("/")
            num_words = num2words(int(numerator), lang=target_lang)
            denom_words = num2words(int(denominator), lang=target_lang)
            
            if target_lang == 'en':
                # For English: "two thirds" or "two over three"
                denom_suffix = ""
                if denominator == "2":
                    denom_suffix = "half"
                elif denominator == "3":
                    denom_suffix = "third"
                elif denominator == "4":
                    denom_suffix = "quarter"
                else:
                    denom_suffix = f"{denom_words}th"
                
                if numerator == "1":
                    return f"one {denom_suffix}", False
                else:
                    return f"{num_words} {denom_suffix}s", False
            else:
                # For other languages, use simple format
                return f"{num_words} by {denom_words}", False
        
        # Handle phone numbers and other complex patterns
        # Split the span into tokens while preserving special characters
        tokens = re.findall(r'(\d+|\D+)', span)
        hint_tokens = []
        
        for token in tokens:
            if token.isdigit():
                try:
                    num_words = num2words(int(token), lang=target_lang)
                    hint_tokens.append(num_words)
                except:
                    hint_tokens.append(token)
            else:
                # Handle special characters like ## in "##30"
                if '#' in token:
                    # Convert numbers within tokens like "##30"
                    sub_tokens = re.findall(r'(\d+|#+)', token)
                    sub_hints = []
                    for sub in sub_tokens:
                        if sub.isdigit():
                            try:
                                sub_hints.append(num2words(int(sub), lang=target_lang))
                            except:
                                sub_hints.append(sub)
                        else:
                            sub_hints.append(sub)
                    hint_tokens.append(''.join(sub_hints))
                else:
                    hint_tokens.append(token)
        
        hint_text = ''.join(hint_tokens)
        
        # Clean up: remove extra spaces around punctuation
        hint_text = re.sub(r'\s+([/+\-:])', r'\1', hint_text)
        hint_text = re.sub(r'([/+\-:])\s+', r'\1', hint_text)
        
        # Check if conversion actually happened
        if hint_text == span:
            return span, True  # No conversion happened
        else:
            return hint_text, False
        
    except Exception as e:
        print(f"Error converting '{span}': {e}")
        return span, True
    
class BertDecoderOnlyLM(torch.nn.Module):
    def __init__(
        self,
        vocab_size,
        num_languages,                # Number of language tokens.
        num_entity_types,             # Number of entity types.
        tokenizer,
        model_name = "google/muril-base-cased",
        max_len=512,
        sep_token_id=None,
        padding_token_id=None,
    ):
        super(BertDecoderOnlyLM, self).__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.tokenizer = tokenizer
        
        # Initialize configuration first
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.is_decoder = True
        self.config.add_cross_attention = False
        self.hidden_size = self.config.hidden_size  # Should be equal to n_embd ideally

        # Load the decoder model (BERT repurposed)
        self.decoder = BertModel.from_pretrained(model_name, config=self.config)
        self.decoder.resize_token_embeddings(len(self.tokenizer))
        
        # Positional Embedding for the constructed prompt
        # self.pos_embedding = nn.Embedding(max_len, self.hidden_size)

        # --- Final Output Head ---
        self.ln_f = nn.LayerNorm(self.hidden_size)
        self.head = nn.Linear(self.hidden_size, vocab_size, bias=False)
        # Tie the head weight to the token embeddings from the decoder.
        self.head.weight = self.decoder.embeddings.word_embeddings.weight

        # --- Additional Conditioning Embeddings ---
        self.lang_embedding = nn.Embedding(num_languages, self.hidden_size)
        self.entity_embedding = nn.Embedding(num_entity_types, self.hidden_size)

        # Set SEP and PAD token IDs.
        self.sep_token_id = sep_token_id if sep_token_id is not None else 105 
        self.padding_token_id = padding_token_id if padding_token_id is not None else 0

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=self.padding_token_id)

    def forward(
        self,
        unnorm_span_ids,   # Tokens for the unnormalized span.
        hint_input_ids,      # Tokens for the hint (e.g., numbers converted to words).
        decoder_input_ids,   # Decoder input tokens (e.g., target sequence with BOS).
        entity_ids,          # Entity type indices.
        language_ids,        # Language indices.
        labels=None,          # Optional target tokens for loss computation.
        **kwargs
    ):
        """
        Forward pass for the decoder-only model.
        The input prompt is constructed as:
          [Language Token] + [Entity Token] + [CLS] + [Hint Span] + [SEP] + [CLS] + [Unnormalized Span] + [SEP] + [CLS] + [Decoder Input] + [SEP]
        The model outputs the normalized span.
        """
        # breakpoint()
        batch_size = unnorm_span_ids.size(0)
        
        # --- Conditioning Embeddings ---
        # Obtain language and entity embeddings in hidden_size.
        lang_emb = self.lang_embedding(language_ids)   # (B, hidden_size)
        if len(lang_emb.shape) == 2:
            lang_emb = lang_emb.unsqueeze(1)                # (B, 1, hidden_size)
        ent_emb = self.entity_embedding(entity_ids)       # (B, hidden_size)
        if len(ent_emb.shape) == 2:
            ent_emb = ent_emb.unsqueeze(1)                  # (B, 1, hidden_size)

        # --- Token Embeddings ---
        hint_emb = self.decoder.embeddings(hint_input_ids)    # (B, L_hint, hidden_size)
        span_emb = self.decoder.embeddings(unnorm_span_ids)   # (B, L_span, hidden_size)
        dec_emb = self.decoder.embeddings(decoder_input_ids)    # (B, L_dec, hidden_size)

        # --- Construct Input Prompt ---
        # Prompt: [Language Token] + [Entity Token] + [CLS] + [Hint Span] + [SEP] 
        # # hint contains [CLS] and [SEP] as it is from the tokenizer without excluding special tokens, so no need to explicitly add them.
        prompt_seq = torch.cat([lang_emb, ent_emb, hint_emb], dim=1)  # (B, L_prompt, hidden_size)
        # Full input: [Prompt Seq] + [CLS] + [Unnormalized Span] + [SEP] + [CLS] + [Decoder Input] + [SEP]
        # full_input = torch.cat([prompt_seq, sep_emb, span_emb, sep_emb, dec_emb], dim=1)  # (B, T, hidden_size)
        full_input = torch.cat([prompt_seq, span_emb, dec_emb], dim=1)  # (B, T, hidden_size)
        
        seq_len = full_input.size(1)
        # --- Positional Embeddings ---
        # positions = torch.arange(0, seq_len, device=full_input.device).unsqueeze(0)
        # pos_emb = self.pos_embedding(positions)  # (1, T, hidden_size)
        # full_input = full_input + pos_emb

        # --- Causal Mask ---
        # causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=full_input.device))

        # --- Transformer Decoder ---
        # decoder_output = self.decoder(inputs_embeds=full_input, attention_mask=causal_mask)[0]  # (B, T, hidden_size)
        decoder_output = self.decoder(inputs_embeds=full_input)[0]  # (B, T, hidden_size)

        # --- Final Output Head ---
        x_norm = self.ln_f(decoder_output)
        logits_all = self.head(x_norm)  # (B, T, vocab_size)

        # Compute context length: length of [Prompt Seq] + [SEP] + [Unnormalized Span] + [SEP]
        L_prompt = prompt_seq.size(1)
        L_span = span_emb.size(1)
        # context_len = L_prompt + 1 + L_span + 1
        context_len = L_prompt + L_span
        
        # breakpoint()
        # Extract logits corresponding to the decoder input portion.
        logits = logits_all[:, context_len:, :]
        
        if labels is not None:
            target = labels[:, 1:].contiguous()
            shifted_logits = logits[:, :-1, :]
            loss = self.loss_fct(shifted_logits.reshape(-1, shifted_logits.size(-1)), target.reshape(-1))
            return {"loss": loss, "logits": logits}
    
        return {"logits": logits}
language_mapping = {"en": 0, "hi": 1, "te": 2, "kan": 3}
device="cuda:0"
# Entity type mapping
entity_type_mapping = {
    "CARDINAL": 0,
    "DATE": 1,
    "FRACTION": 2,
    "MONEY": 3,
    "TELEPHONE": 4,
    "MEASURE": 5,
    "TIME": 6,
    "DECIMAL": 7
}
ib_decoder = BertDecoderOnlyLM(
    vocab_size=len(tokenizer),
    num_languages=len(language_mapping),
    num_entity_types=len(entity_type_mapping),
    tokenizer=tokenizer,
    model_name="ai4bharat/IndicBERTv2-MLM-Sam-TLM"
)

decoder_weights=torch.load("/home/sakshamt/SPIRE_TN/TN_Models/Indic_Bert/weights_kan/IB_decoder_weights_kan_updated.pt")
ib_decoder.load_state_dict(decoder_weights)
ib_decoder=ib_decoder.to(device)
model_name = "google/mt5-small"  # or mt5-base, mt5-large if resources permit
tokenizer_mt5 = MT5Tokenizer.from_pretrained(model_name)
mt5 = MT5ForConditionalGeneration.from_pretrained(model_name)
checkpoint_path = "/home/sakshamt/SPIRE_TN/checkpoints_kan/checkpoint-step110000.pt"
mt5.load_state_dict(torch.load(checkpoint_path, map_location=device))
id_to_class={v:k for k ,v in class_to_id.items()}
ner_model=ner_model.to(device)
# -------------------------------
# 1. Batch NER
# -------------------------------
def ner_batch(texts: List[str], batch_size=16):
    all_entities = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
            return_offsets_mapping=True
        )
        offset_mapping = inputs.pop("offset_mapping")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = ner_model(**inputs)

        predictions = torch.argmax(outputs.logits, dim=2).cpu().numpy()
        input_ids = inputs["input_ids"].cpu().numpy()

        for idx in range(len(batch_texts)):
            tokens = tokenizer.convert_ids_to_tokens(input_ids[idx])
            labels = [id_to_class[p] for p in predictions[idx]]
            offsets = offset_mapping[idx].tolist()

            entities = []
            current = None
            for (token, label, (start, end)) in zip(tokens, labels, offsets):
                if token in ["[CLS]", "[SEP]", "[PAD]"]:
                    continue

                if label.startswith("B-"):
                    if current:
                        entities.append(current)
                    current = {"text": token.replace("##",""),
                               "start": start, "end": end, "type": label[2:]}
                elif label.startswith("I-") and current and current["type"] == label[2:]:
                    current["text"] += token.replace("##","")
                    current["end"] = end
                else:
                    if current:
                        entities.append(current)
                        current = None
            if current:
                entities.append(current)

            all_entities.append(entities)

    return all_entities


# -------------------------------
# 2. Batch Decoder for spans
# -------------------------------
# -------------------------------
# 2. Batch Decoder for spans with CARDINAL decimal restriction
# -------------------------------
def decode_spans_batch(spans: List[str], entity_types: List[str], language="kan", max_new_tokens=20, debug=False):
    ib_decoder.eval()
    batch_size = len(spans)

    # Build hints
    hint_texts = [get_hint_from_span(s, lang=language)[0] for s in spans]
    hint_input_ids = tokenizer(
        hint_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=16
    ).input_ids.to(device)

    unnorm_span_ids = tokenizer(
        spans,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=32
    ).input_ids.to(device)

    decoder_input_ids = torch.full((batch_size, 1), tokenizer.cls_token_id, dtype=torch.long, device=device)
    entity_ids = torch.tensor([entity_type_mapping[t] for t in entity_types], dtype=torch.long, device=device)
    language_ids = torch.tensor([language_mapping[language]] * batch_size, dtype=torch.long, device=device)

    generated = decoder_input_ids.clone()
    finished = [False] * batch_size

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = ib_decoder(
                unnorm_span_ids=unnorm_span_ids,
                hint_input_ids=hint_input_ids,
                decoder_input_ids=generated,
                entity_ids=entity_ids,
                language_ids=language_ids
            )

            logits = outputs["logits"]
            next_token_ids = torch.argmax(logits[:, -1, :], dim=-1)

            generated = torch.cat([generated, next_token_ids.unsqueeze(1)], dim=1)

            # Check for SEP token
            for i in range(batch_size):
                token_str = tokenizer.convert_ids_to_tokens(next_token_ids[i].item())
                if token_str in ["[SEP]", "[PAD]"]:
                    finished[i] = True

            if all(finished):
                break

    def wordpiece_detokenize(tokens):
        out = []
        for tok in tokens:
            if tok.startswith("##") and len(out) > 0:
                out[-1] = out[-1] + tok[2:]
            else:
                out.append(tok)
        return " ".join(out)

    # -------------------------------
    # Detokenize & apply CARDINAL restriction
    # -------------------------------
    decimal_markers = ["ಪಾಯಿಂಟ್", "point", "dot", "दशमलव", "decimal"]

    output_texts = []
    for i in range(batch_size):
        tokens = tokenizer.convert_ids_to_tokens(generated[i][1:])  # drop CLS
        tokens = [t for t in tokens if t not in ["[SEP]", "[PAD]"]]
        text = wordpiece_detokenize(tokens)

        # CARDINAL: remove anything after decimal marker
        if entity_types[i] == "CARDINAL":
            for marker in decimal_markers:
                if marker in text:
                    if debug:
                        print(f"⚠️ CARDINAL decimal detected, trimming: {text}")
                    text = text.split(marker)[0].strip()
                    break

        output_texts.append(text)

    return output_texts

# -------------------------------
# 3. Batch normalize sentences
# -------------------------------
def normalize_text_batch(texts: List[str], ner_batch_size=16):
    all_entities = ner_batch(texts, batch_size=ner_batch_size)
    normalized_texts = []

    for text, entities in zip(texts, all_entities):
        if not entities:
            normalized_texts.append(text)
            continue

        normalized_text = text
        offset_shift = 0

        # Extract spans and types
        spans = [ent["text"] for ent in entities]
        types = [ent["type"] for ent in entities]

        # Batch decode
        norm_spans = decode_spans_batch(spans, types)

        for ent, norm in zip(entities, norm_spans):
            start = ent["start"] + offset_shift
            end = ent["end"] + offset_shift
            normalized_text = normalized_text[:start] + norm + normalized_text[end:]
            offset_shift += len(norm) - (end - start)

        normalized_texts.append(normalized_text)

    return normalized_texts


def generate_output_mt5_batch(input_texts: list, batch_size: int = 16):
    """
    Generate outputs from mT5 for a list of input texts in batches.

    Args:
        input_texts (list[str]): List of input sentences.
        batch_size (int): Number of sentences per batch.

    Returns:
        List[str]: Generated output sentences.
    """
    outputs = []

    # Process in batches
    for i in range(0, len(input_texts), batch_size):
        batch_texts = input_texts[i:i+batch_size]

        # Tokenize batch
        inputs = tokenizer_mt5(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=500
        )

        input_ids = inputs["input_ids"].to(mt5.device)
        attention_mask = inputs["attention_mask"].to(mt5.device)

        # Generate output
        with torch.no_grad():
            output_ids = mt5.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=500,
                num_beams=1,
                early_stopping=True
            )

        # Decode all outputs in batch
        batch_outputs = [tokenizer_mt5.decode(ids, skip_special_tokens=True) for ids in output_ids]
        outputs.extend(batch_outputs)

    return outputs
tag_pattern = re.compile(r"<([^>]+)><[A-Z]+>")
class NormalizationDataset(Dataset):
    def __init__(self, file_path):
        """
        Args:
            file_path (str): path to the file containing JSON lines
        """
        self.data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                translated = record.get("translated_tagged", "")
                normalized = record.get("normalized_output", "")
                ID=record.get("id","")

                # Remove tags and keep only the value
                translated_clean = tag_pattern.sub(r"\1", translated)
                normalized_clean = tag_pattern.sub(r"\1", normalized)

                self.data.append({
                    "unnorm": translated_clean,
                    "tagged_output":translated,
                    "norm": normalized,
                    "id":ID
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item
dataset_train=NormalizationDataset("/home/sakshamt/SPIRE_TN/DATASET/Dataset_ver.2/aug_kan.txt")

import json
from tqdm import tqdm

batch_size = 16
output_path = "/home/sakshamt/SPIRE_TN/DATASET/Dataset_ver.2/temp_kan.txt"   # change as needed

with open(output_path, "w", encoding="utf-8") as f_out:

    for start in tqdm(range(0, len(dataset_train), batch_size), desc="Processing batches"):
        end = min(start + batch_size, len(dataset_train))
        batch = dataset_train[start:end]

        # Extract fields
        unnorm_texts = [item["unnorm"] for item in batch]
        norm_texts   = [item["norm"] for item in batch]
        tagged_outputs   = [item["tagged_output"] for item in batch]
        id_texts     = [item["id"] for item in batch]

        # 1️ IB normalization (batch)
        hints_ib_texts = normalize_text_batch(
            unnorm_texts,
            ner_batch_size=batch_size
        )

        # 2️ mT5 generation (batch)
        mt5_texts = generate_output_mt5_batch(
            unnorm_texts,
            batch_size=batch_size
        )

        # 3️ Write output (one dict per line)
        for i in range(len(unnorm_texts)):
            record = {
                "tagged_output": tagged_outputs[i],
                "norm": norm_texts[i],
                "mt5": mt5_texts[i],
                "hints_ib": hints_ib_texts[i],
               # "id": id_texts[i]
            }

            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f" Finished! Output saved to: {output_path}")
