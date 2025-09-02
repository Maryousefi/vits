import os

# Paths
base_dir = "data/amir"
wav_dir = os.path.join(base_dir, "wavs")
transcript_path = os.path.join(base_dir, "transcripts.txt")
out_dir = "filelists/fa_single"

os.makedirs(out_dir, exist_ok=True)

# Read transcripts
with open(transcript_path, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

# Expand to absolute or relative wav paths
pairs = []
for line in lines:
    fname, text = line.split("|", 1)
    wav_path = os.path.join(wav_dir, fname)
    pairs.append(f"{wav_path}|{text}")

# Split 80/20 for train/val
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

print(f" Wrote {len(train_pairs)} training lines to {train_out}")
print(f" Wrote {len(val_pairs)} validation lines to {val_out}")
