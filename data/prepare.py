import os
import requests
import tiktoken
import numpy as np
import argparse

def download_file(url: str, filepath: str):
    """
    Download a file from a URL to a local path.
    
    This uses 'streaming' to handle large files. Instead of trying to download 
    10GB into RAM at once, it downloads it in little 8KB chunks and writes 
    them to disk immediately.
    """
    if not os.path.exists(filepath):
        print(f"Downloading {url} to {filepath}...")
        # stream=True is crucial for large files!
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("Download complete.")
    else:
        print(f"{filepath} already exists. Skipping download.")

def prepare_tinyshakespeare(data_dir: str = "data"):
    """
    Main logic to prepare the TinyShakespeare dataset.
    
    Steps:
    1. Download raw text (input.txt).
    2. Split into Train (90%) and Validation (10%) sets.
    3. Tokenize (Convert text -> integers) using GPT-2 BPE.
    4. Save as efficient binary files (.bin).
    """
    # Create the 'data' folder if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    input_file_path = os.path.join(data_dir, 'input.txt')
    url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    
    # 1. Download the raw text file
    download_file(url, input_file_path)

    # Read the text file into a simple string
    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = f.read()
    
    # 2. Split Data: Train vs Validation
    # We train on 90% of the data, and we test the model on the remaining 10%
    # to make sure it isn't just "memorizing" the answers (overfitting).
    n = len(data)
    train_data = data[:int(n*0.9)]
    val_data = data[int(n*0.9):]

    # 3. Encode (Tokenize)
    # We use "gpt2" encoding. This is Byte Pair Encoding (BPE).
    # It turns words/sub-words into specific integers.
    enc = tiktoken.get_encoding("gpt2")
    
    # encode_ordinary is a faster version of encode() that ignores special tokens
    train_ids = enc.encode_ordinary(train_data)
    val_ids = enc.encode_ordinary(val_data)
    
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    # 4. Export to Binary (.bin)
    # KEY LEARNING MOMENT: Why uint16?
    # The GPT-2 vocabulary size is roughly 50,257 tokens.
    # - uint8 (0 to 255) is too small.
    # - uint16 (0 to 65,535) is PERFECT.
    # - int32 or int64 would work but would take up 2x or 4x more disk space/RAM.
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    
    # .tofile() writes the raw bytes to disk.
    # This creates the exact format that np.memmap() (in the DataLoader) expects to read.
    train_ids.tofile(os.path.join(data_dir, 'train.bin'))
    val_ids.tofile(os.path.join(data_dir, 'val.bin'))
    print("Saved train.bin and val.bin")

if __name__ == "__main__":
    # This allows you to run the script from the command line.
    # Example: python prepare.py --dataset tinyshakespeare
    parser = argparse.ArgumentParser(description="Prepare dataset")
    parser.add_argument("--dataset", type=str, default="tinyshakespeare", help="Dataset to prepare")
    args = parser.parse_args()

    if args.dataset == "tinyshakespeare":
        prepare_tinyshakespeare()
    else:
        print(f"Unknown dataset: {args.dataset}")