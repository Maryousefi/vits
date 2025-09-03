import os

# Paths
base_dir = "/data/amir"  # absolute path to dataset
wav_dir = os.path.join(base_dir, "wavs")
out_dir = "filelists/fa_single"

os.makedirs(out_dir, exist_ok=True)

# Read transcripts directly from wav folder + inline mapping
lines = [
    "000001.wav|تو اپیزود اول از خلاصه کتاب خودآموز دیکتاتورها",
    "000002.wav|گفتیم که نویسندگان این کتاب اومدن داستان به قدرت رسیدن",
    "000003.wav|و نحوه حکومت دیکتاتورها رو دقیق و با جزئیات بررسی کردن",
    "000004.wav|فصل مشترک زندگی و حکومت دیکتاتورها رو دراودن",
    "000005.wav|و با زبان طنز اومدن کتاب رو برای کسانی که دوست دارن دیکتاتور بشن نوشتن",
    "000006.wav|هرکی دوست داره دیکتاتور بشه",
    "000007.wav|میتونه از تجربیات دیکتاتورهای دیگه",
    "000008.wav|که تو این کتاب جمع آوری شده استفاده کنه",
    "000009.wav|تو اپیزود قبل درباره نحوه به قدرت رسیدن دیکتاتورها صحبت کردیم",
    "000010.wav|و بعد از کیش شخصیت بی همتایی که دیکتاتورها دوست دارن از خودشون بسازن گفتیم",
]

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
with open(os.path.join(out_dir, "train.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(train_pairs) + "\n")

with open(os.path.join(out_dir, "val.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(val_pairs) + "\n")

print(f" Wrote {len(train_pairs)} training lines")
print(f" Wrote {len(val_pairs)} validation lines")
