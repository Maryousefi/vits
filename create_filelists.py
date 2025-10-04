# create_filelists.py
import os
import random

# Paths
base_dir = "dataset/amir"
wav_dir = os.path.join(base_dir, "wavs")
transcript_path = os.path.join(base_dir, "transcripts.txt")
out_dir = "filelists/fa_single"

os.makedirs(out_dir, exist_ok=True)

# Read transcripts
with open(transcript_path, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

pairs = []
for line in lines:
    try:
        fname, text = line.split("|", 1)
    except ValueError:
        print(f"Skipping malformed line: {line}")
        continue

    wav_path = os.path.join(wav_dir, fname)
    if not os.path.exists(wav_path):
        print(f"Warning: missing wav file {wav_path}")
        continue

    pairs.append(f"{wav_path}|{text}")

# Shuffle for randomness
random.shuffle(pairs)

# Split 80/20
split_idx = int(len(pairs) * 0.8)
train_pairs = pairs[:split_idx]
val_pairs = pairs[split_idx:]

# Save
train_out = os.path.join(out_dir, "train.txt")
val_out = os.path.join(out_dir, "val.txt")

with open(train_out, "w", encoding="utf-8") as f:
    f.write("\n".join(train_pairs) + "\n")

with open(val_out, "w", encoding="utf-8") as f:
    f.write("\n".join(val_pairs) + "\n")

print(f"Wrote {len(train_pairs)} training lines to {train_out}")
print(f"Wrote {len(val_pairs)} validation lines to {val_out}")
