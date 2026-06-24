## Synthetic Data Generation Pipeline

The synthetic data generation process consists of three stages:

### Step 1: Extraction of Tagged Data

Sentences are extracted from the Kestrel dataset such that each sentence contains a primary unnormalized entity belonging to a specific category (e.g., DATE, CARDINAL, MONEY, etc.). The sentence may either:

* Contain unnormalized entities from only that category (**Only**), or
* Contain a primary entity from the target category along with entities from other categories (**Mixed**).

The extracted files are stored as:

```text
OP_<LANGUAGE>_<KESTREL_FILE_NO>_<CATEGORY>_<Mixed/Only>.txt
```

### Step 2: Translation to Target Indic Language

The extracted sentences are translated into the target Indic language using the NLLB-1B model. Since machine translation models typically leave numeric and other non-standard tokens unchanged, the resulting sentences generally preserve the original unnormalized entities while translating the surrounding text.

The translated files are stored as:

```text
OP_<LANGUAGE>_<KESTREL_FILE_NO>_<CATEGORY>_<Mixed/Only>_translated.txt
```

### Step 3: Generation of Normalized Data

The translated sentences are then provided to Gemini along with category-specific prompts. Gemini generates the fully normalized spoken-form representation of the unnormalized entities in the target language.

The normalized files are stored as:

```text
OP_<LANGUAGE>_<KESTREL_FILE_NO>_<CATEGORY>_<Mixed/Only>_norm.txt
```

### Directory Flow

```text
Extracted Tagged Data
└── OP_<LANGUAGE>_<KESTREL_FILE_NO>_<CATEGORY>_<Mixed/Only>.txt

Translated Data
└── OP_<LANGUAGE>_<KESTREL_FILE_NO>_<CATEGORY>_<Mixed/Only>_translated.txt

Normalized Data
└── OP_<LANGUAGE>_<KESTREL_FILE_NO>_<CATEGORY>_<Mixed/Only>_norm.txt
```
Fisrst extraction_tagging.py is used for the extraction part
Second translated_tagging.py  is used for the translation part
Third gemini_norm.py is used for obtaining the complete normalized and unnormalized data.
