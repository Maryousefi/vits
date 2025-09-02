import os
import random

dataset_dir = "dataset/amir"
wavs_dir = os.path.join(dataset_dir, "wavs")
transcript_file = os.path.join(dataset_dir, "transcripts.txt")

# Read transcripts
with open(transcript_file, encoding="utf-8") as f:
    lines = [line.strip().split("|") for line in f]

data = [os.path.join(wavs_dir, wav) + "|" + text for wav, text in lines]

# Shuffle and split
random.shuffle(data)
split_idx = int(0.9 * len(data))
train, val = data[:split_idx], data[split_idx:]

# Save
os.makedirs("filelists", exist_ok=True)
with open("filelists/fa_amir_train.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(train))
with open("filelists/fa_amir_val.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(val))

print(f"Created {len(train)} train and {len(val)} val samples.")
