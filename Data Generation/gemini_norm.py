import os
import ast
import json
import time
import threading
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import google.genai as genai

# API_KEY = "AIzaSyDrxD7mxu95GjF_6QU9oGv3182wkQ3F748"
client = genai.Client(api_key=API_KEY)

MODEL = "gemini-2.0-flash"

INPUT_FILE = "/raid/home/rizwank/Normalization/data_generation/Post_processed_Data/Kanada/test_remaining.txt"
OUTPUT_FILE = "/raid/home/rizwank/Normalization/data_generation/Post_processed_Data/Kanada/TEMP_remaining_norm.txt"

BATCH_SIZE = 20
MAX_WORKERS = 4

def build_prompt_1(batch):
    rules = """
You are a Hindi text normalizer.

RULES:
1. Convert ONLY the content inside < > into natural spoken Hindi.
2. Do NOT remove, add, reorder, or rename any tags.
3. Do NOT change anything outside the tags.
4. Keep the sentence structure, words, punctuation, and spacing EXACTLY the same as in "translated_tagged".
5. "normalized_output" must be identical to "translated_tagged" except for the text inside the tags.
6. Output ONLY a valid JSON array.
7. Each item must contain exactly these fields:
   - "tagged"
   - "translated_tagged"
   - "normalized_output"

IMPORTANT (ALL TAGS):
For every tag (<DATE>, <TIME>, <MONEY>, <DECIMAL>, <FRACTION>, <MEASURE>, <TELEPHONE>, etc.),
only replace the numeric/text value inside the tag with its spoken Hindi form.
Do NOT modify any surrounding words.

STRICT RULE:
normalized_output = translated_tagged
with ONLY the tagged values replaced by their spoken Hindi forms.

{
  "tagged": "The package weighs <2.5 kg><MEASURE>.",
  "translated_tagged": "पैकेज का वजन <2.5 kg><MEASURE> है।",
  "normalized_output": "पैकेज का वजन <दो दशमलव पाँच किलोग्राम><MEASURE> है।"
}
{
  "tagged": "The table is <120 cm><MEASURE> long.",
  "translated_tagged": "मेज की लंबाई <120 cm><MEASURE> है।",
  "normalized_output": "मेज की लंबाई <एक सौ बीस सेंटीमीटर><MEASURE> है।"
}

{
  "tagged": "He bought <3><CARDINAL> kg rice for <250><MONEY> on <12/10/2023><DATE> at <10:30><TIME>.",
  "translated_tagged": "<12/10/2023><DATE> को <10:30><TIME> बजे उसने <3><CARDINAL> किलो चावल <250><MONEY> में खरीदा।",
  "normalized_output": "<बारह अक्टूबर दो हजार तेईस><DATE> को <दस बजकर तीस मिनट><TIME> बजे उसने <तीन><CARDINAL> किलो चावल <दो सौ पचास रुपये><MONEY> में खरीदा।"
}
{
  "tagged": "She paid <₹750><MONEY> for the dress on <05/11/2024><DATE> at <18:45><TIME>.",
  "translated_tagged": "<05/11/2024><DATE> को <18:45><TIME> बजे उसने कपड़े के लिए <₹750><MONEY> भुगतान किया।",
  "normalized_output": "<पाँच नवंबर दो हजार चौबीस><DATE> को <अठारह बजकर पैंतालीस मिनट><TIME> बजे उसने कपड़े के लिए <सात सौ पचास रुपये><MONEY> भुगतान किया।"
}

{
  "tagged": "She paid <1500><MONEY> for <2><CARDINAL> tickets at <6:45><TIME> on <01/01/2024><DATE>.",
  "translated_tagged": "<01/01/2024><DATE> को <6:45><TIME> बजे <2><CARDINAL> टिकटों के लिए <1500><MONEY> भुगतान किया।",
  "normalized_output": "<एक जनवरी दो हजार चौबीस><DATE> को <छह बजकर पैंतालीस मिनट><TIME> बजे <दो><CARDINAL> टिकटों के लिए <एक हजार पाँच सौ रुपये><MONEY> भुगतान किया।"
}

{
  "tagged": "The box weighs <2.5><DECIMAL> kg and has <4><CARDINAL> items costing <750><MONEY>.",
  "translated_tagged": "डिब्बे का वजन <2.5><DECIMAL> किलो है और इसमें <4><CARDINAL> वस्तुएँ हैं जिनकी कीमत <750><MONEY> है।",
  "normalized_output": "डिब्बे का वजन <दो पॉइंट पाँच><DECIMAL> किलो है और इसमें <चार><CARDINAL> वस्तुएँ हैं जिनकी कीमत <सात सौ पचास रुपये><MONEY> है।"
}

{
  "tagged": "Call me at <9876543210><TELEPHONE> after <9:15><TIME> with <2><CARDINAL> updates.",
  "translated_tagged": "<9:15><TIME> बजे के बाद <9876543210><TELEPHONE> पर <2><CARDINAL> अपडेट के साथ कॉल करें।",
  "normalized_output": "<नौ बजकर पंद्रह मिनट><TIME> बजे के बाद <नौ आठ सात छह पाँच चार तीन दो एक शून्य><TELEPHONE> पर <दो><CARDINAL> अपडेट के साथ कॉल करें।"
}

{
  "tagged": "He paid <1200><MONEY> on <15/08/2023><DATE> at <10:45><TIME> for <3><CARDINAL> tickets.",
  "translated_tagged": "<15/08/2023><DATE> को <10:45><TIME> बजे <3><CARDINAL> टिकटों के लिए <1200><MONEY> भुगतान किया।",
  "normalized_output": "<पंद्रह अगस्त दो हजार तेईस><DATE> को <दस बजकर पैंतालीस मिनट><TIME> बजे <तीन><CARDINAL> टिकटों के लिए <एक हजार दो सौ रुपये><MONEY> भुगतान किया।"
}

{
  "tagged": "He ate <1/2><FRACTION> of the cake and gave <3><CARDINAL> pieces to friends.",
  "translated_tagged": "उसने केक का <1/2><FRACTION> हिस्सा खाया और दोस्तों को <3><CARDINAL> टुकड़े दिए।",
  "normalized_output": "उसने केक का <एक बटा दो><FRACTION> हिस्सा खाया और दोस्तों को <तीन><CARDINAL> टुकड़े दिए।"
}
"""
    return rules + "\n\nBatch:\n" + json.dumps(batch, ensure_ascii=False, indent=2)

def build_prompt(batch):
    rules = """
You are a Kannada text normalizer.

RULES:
1. Convert ONLY the content inside < > into natural spoken Kannada.
2. Do NOT remove, add, reorder, or rename any tags.
3. Do NOT change anything outside the tags.
4. Keep the sentence structure, words, punctuation, and spacing EXACTLY the same as in "translated_tagged".
5. "normalized_output" must be identical to "translated_tagged" except for the text inside the tags.
6. Output ONLY a valid JSON array.
7. Each item must contain exactly these fields:
   - "tagged"
   - "translated_tagged"
   - "normalized_output"

.

IMPORTANT (ALL TAGS):
For every tag (<DATE>, <TIME>, <MONEY>, <DECIMAL>, <FRACTION>, <MEASURE>, <TELEPHONE>, etc.),
only replace the numeric/text value inside the tag with its spoken Kannada form.
Do NOT modify any surrounding words.

STRICT RULE:
normalized_output = translated_tagged
with ONLY the tagged values replaced by their spoken Kannada forms.

Examples of output


TELEPHONE examples

{
  "tagged": "Call me at <9876543210><TELEPHONE>.",
  "translated_tagged": "ನನಗೆ <9876543210><TELEPHONE> ಗೆ ಕರೆ ಮಾಡಿ.",
  "normalized_output": "ನನಗೆ <ಒಂಬತ್ತು ಎಂಟು ಏಳು ಆರು ಐದು ನಾಲ್ಕು ಮೂರು ಎರಡು ಒಂದು ಶೂನ್ಯ><TELEPHONE> ಗೆ ಕರೆ ಮಾಡಿ."
}
{
  "tagged": "Customer care number is <1800123456><TELEPHONE>.",
  "translated_tagged": "ಗ್ರಾಹಕ ಸೇವೆ ಸಂಖ್ಯೆ <1800123456><TELEPHONE> ಆಗಿದೆ.",
  "normalized_output": "ಗ್ರಾಹಕ ಸೇವೆ ಸಂಖ್ಯೆ <ಒಂದು ಎಂಟು ಶೂನ್ಯ ಶೂನ್ಯ ಒಂದು ಎರಡು ಮೂರು ನಾಲ್ಕು ಐದು ಆರು><TELEPHONE> ಆಗಿದೆ."
}
{
  "tagged": "ISBN <978-0-323-09139-8><TELEPHONE> ; Access provided by the University of Pittsburgh .",
  "translated_tagged": "ISBN <978-0-323-09139-8><TELEPHONE> ; ಪ್ರವೇಶವನ್ನು ಪಿಟ್ಸ್ಬರ್ಗ್ ವಿಶ್ವವಿದ್ಯಾಲಯ ಒದಗಿಸಿದೆ.",
  "normalized_output": "ISBN <ಒಂಬತ್ತು ಏಳು ಎಂಟು ಡ್ಯಾಶ್ ಶೂನ್ಯ ಡ್ಯಾಶ್ ಮೂರು ಎರಡು ಮೂರು ಡ್ಯಾಶ್ ಶೂನ್ಯ ಒಂಬತ್ತು ಒಂದು ಮೂರು ಒಂಬತ್ತು ಡ್ಯಾಶ್ ಎಂಟು><TELEPHONE> ; ಪ್ರವೇಶವನ್ನು ಪಿಟ್ಸ್ಬರ್ಗ್ ವಿಶ್ವವಿದ್ಯಾಲಯ ಒದಗಿಸಿದೆ."
}

"""

    return rules + "\n\nBatch:\n" + json.dumps(batch, ensure_ascii=False, indent=2)



def call_gemini(batch):
    max_retries = 3

    for attempt in range(max_retries):
        try:
            prompt = build_prompt(batch)

            resp = client.models.generate_content(
                model=MODEL,
                contents=[{"role": "user", "parts": [{"text": prompt}]}],
            )

            text = resp.text.strip()

            if text.startswith("```"):
                text = text.strip("`")
                lines = text.split("\n")
                for i, line in enumerate(lines):
                    if line.strip() and line.strip().lower() not in ["json", "python"]:
                        text = "\n".join(lines[i:])
                        break

            return json.loads(text)

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2**attempt)
                continue

            return [
                {
                    "tagged": item.get("tagged", ""),
                    "translated_tagged": item.get("translated_tagged", ""),
                    "normalized_output": item.get("translated_tagged", "")
                }
                for item in batch
            ]


def read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield ast.literal_eval(line)
            except Exception:
                continue


def main():
    print("🚀 Starting Gemini normalization")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    buffer = []
    all_batches = []

    for item in read_lines(INPUT_FILE):
        buffer.append({
            "tagged": item.get("tagged", ""),
            "translated_tagged": item.get("translated_tagged", "")
        })

        if len(buffer) == BATCH_SIZE:
            all_batches.append(buffer)
            buffer = []

    if buffer:
        all_batches.append(buffer)

    total_batches = len(all_batches)
    print(f"Total batches: {total_batches}")

    start = time.time()
    write_lock = threading.Lock()
    completed = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(call_gemini, b): i for i, b in enumerate(all_batches)}

            with tqdm(total=total_batches, desc="Processing") as pbar:
                for future in as_completed(futures):
                    result = future.result()

                    with write_lock:
                        for d in result:
                            fout.write(str(d) + "\n")

                    completed += 1
                    pbar.update(1)

                    elapsed = time.time() - start
                    avg = elapsed / completed
                    eta = (total_batches - completed) * avg

                    print(
                        f"{completed}/{total_batches} | "
                        f"Avg: {avg:.2f}s | ETA: {eta/60:.2f}m   ",
                        end=""
                    )

    elapsed = time.time() - start
    print(f"\n\n Done in {elapsed:.2f}s ({elapsed/60:.2f}m)")


if __name__ == "__main__":
    main()
