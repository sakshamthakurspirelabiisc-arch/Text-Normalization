from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import ast
import os
import sys

# Initialize model
model_name = "facebook/nllb-200-1.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="cuda:6")

device = "cuda:6" 
model = model.to(device)

def nllb_translate(text, src_lang, tgt_lang):
    """Translate text using NLLB model."""
    try:
        # Set the correct source language
        tokenizer.src_lang = src_lang

        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)

        # Use convert_tokens_to_ids to get the target language token ID
        forced_bos = tokenizer.convert_tokens_to_ids(tgt_lang)

        # Generate
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos,
            max_length=256,
            num_beams=4,
            early_stopping=True
        )

        return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    except Exception as e:
        print(f"Error translating: {e}")
        return text  # Return original text on error

def dump(data, op_file_path, append=False):
    """
    Dump data to output file.
    
    Args:
        data: List of dictionaries to save
        op_file_path: Output file path
        append: If True, append to existing file
    """
    try:
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(op_file_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created directory: {output_dir}")
        
        # Open file in append or write mode
        mode = 'a' if append else 'w'
        with open(op_file_path, mode, encoding='utf-8') as f:
            for obj in data:
                # Write each dictionary as a string on its own line
                f.write(str(obj) + '\n')
        
        action = "Appended" if append else "Saved"
        print(f"{action} {len(data)} objects to {op_file_path}")
        
    except Exception as e:
        print(f"Error in dump function: {e}")

def process_file(input_file_path, output_file_path, batch_size=10):
    """
    Process input file, translate tagged text, and save to output file.
    
    Args:
        input_file_path: Path to input text file
        output_file_path: Path to output text file
        batch_size: Number of objects to accumulate before saving
    """
    # Check if input file exists
    if not os.path.exists(input_file_path):
        print(f"Error: Input file not found: {input_file_path}")
        sys.exit(1)
    
    # Check if output file already exists
    if os.path.exists(output_file_path):
        print(f"Warning: Output file already exists: {output_file_path}")
        overwrite = input("Do you want to overwrite? (y/n): ")
        if overwrite.lower() != 'y':
            print("Aborting...")
            sys.exit(0)
        else:
            print("Will overwrite existing file.")
    
    # Language settings
    src_lang = "eng_Latn"
    tgt_lang = "kan_Knda"
    
    print(f"Starting translation process...")
    print(f"Input file: {input_file_path}")
    print(f"Output file: {output_file_path}")
    print(f"Source language: {src_lang}")
    print(f"Target language: {tgt_lang}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")
    print("-" * 60)
    
    batch_data = []
    total_processed = 0
    total_lines = 0
    
    # Count total lines for progress
    with open(input_file_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    
    print(f"Total lines to process: {total_lines}")
    
    # Process file
    with open(input_file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Show progress
            if line_num % 100 == 0:
                print(f"Processing line {line_num}/{total_lines}...")
            
            try:
                # Parse as Python dictionary
                obj = ast.literal_eval(line)
                
                # Check if it has 'tagged' field
                if 'tagged' in obj:
                    # Get original tagged text
                    tagged_text = obj["tagged"]
                    
                    # Translate
                    print(f"Translating line {line_num}...")
                    translated = nllb_translate(tagged_text, src_lang, tgt_lang)
                    
                    # Add translated field
                    obj["translated_tagged"] = translated
                    
                    # Add to batch
                    batch_data.append(obj)
                    total_processed += 1
                    
                    # Show sample of first few translations
                    if line_num <= 3:
                        print(f"\nSample {line_num}:")
                        print(f"  Original tagged: {tagged_text[:80]}...")
                        print(f"  Translated: {translated[:80]}...")
                    
                    # Dump batch when it reaches batch_size
                    if len(batch_data) >= batch_size:
                        dump(batch_data, output_file_path, append=(line_num > batch_size))
                        batch_data = []  # Reset batch
                        
                else:
                    print(f"Warning: Line {line_num} has no 'tagged' field")
                    
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                print(f"Line content: {line[:200]}...")
                continue
    
    # Dump any remaining data in the batch
    if batch_data:
        dump(batch_data, output_file_path, append=(total_processed > batch_size))
    
    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE!")
    print(f"{'='*60}")
    print(f"Total objects processed: {total_processed}")
    print(f"Output saved to: {output_file_path}")
    
    # Show final statistics
    if os.path.exists(output_file_path):
        with open(output_file_path, 'r', encoding='utf-8') as f:
            output_lines = sum(1 for _ in f)
        print(f"Output file has {output_lines} lines")
        
        # Show first 3 lines of output
        print("\nFirst 3 lines of output:")
        with open(output_file_path, 'r', encoding='utf-8') as f:
            for i in range(3):
                line = f.readline().strip()
                if line:
                    print(f"Line {i+1}: {line[:150]}...")

# Main execution
if __name__ == "__main__":
    # File paths
    input_file_path = "/raid/home/rizwank/Normalization/data_generation/PreProcessed_Data_tagged/OP_21_MONEY_only_1.txt"
    output_file_path = "/raid/home/rizwank/Normalization/data_generation/Translated_tagged/Kannada/checking_delete_later.txt"
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")
    
    # Process the file
    process_file(input_file_path, output_file_path, batch_size=10)
    
    # Alternatively, use command line arguments
    # if len(sys.argv) >= 3:
    #     input_file = sys.argv[1]
    #     output_file = sys.argv[2]
    #     batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    #     process_file(input_file, output_file, batch_size)
    # else:
    #     print("Usage: python script.py <input_file> <output_file> [batch_size]")
    #     print(f"Example: python script.py input.txt output.txt 10")