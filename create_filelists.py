"""
Create train/validation filelists for Persian VITS training.
Handles Persian text encoding and validates audio files.
"""

import os
import random
import argparse
from pathlib import Path
import soundfile as sf

def validate_audio_file(audio_path, min_duration=0.5, max_duration=10.0):
    """
    Validate audio file duration and format.
    
    Args:
        audio_path: Path to audio file
        min_duration: Minimum duration in seconds
        max_duration: Maximum duration in seconds
    
    Returns:
        tuple: (is_valid, duration)
    """
    try:
        info = sf.info(audio_path)
        duration = info.duration
        
        if min_duration <= duration <= max_duration:
            return True, duration
        else:
            return False, duration
    except Exception as e:
        print(f"Error reading {audio_path}: {e}")
        return False, 0

def clean_persian_text(text):
    """
    Basic Persian text cleaning for filelist creation.
    """
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove empty text
    if not text.strip():
        return None
        
    return text.strip()

def create_filelists_from_directory(audio_dir, output_dir, speaker_name="amir", 
                                  train_ratio=0.9, min_duration=0.5, max_duration=10.0):
    """
    Create filelists from a directory structure.
    Expected structure: audio_dir/speaker_name/wavs/*.wav with corresponding .txt files
    """
    wavs_dir = Path(audio_dir) / speaker_name / "wavs"
    
    if not wavs_dir.exists():
        raise ValueError(f"Audio directory not found: {wavs_dir}")
    
    # Find all audio files
    audio_files = list(wavs_dir.glob("*.wav"))
    
    if len(audio_files) == 0:
        raise ValueError(f"No WAV files found in {wavs_dir}")
    
    valid_pairs = []
    skipped = 0
    
    print(f"Processing {len(audio_files)} audio files...")
    
    for audio_file in audio_files:
        # Look for corresponding text file
        txt_file = audio_file.with_suffix('.txt')
        
        if not txt_file.exists():
            print(f"Warning: No text file for {audio_file.name}")
            skipped += 1
            continue
        
        # Validate audio
        is_valid, duration = validate_audio_file(audio_file, min_duration, max_duration)
        
        if not is_valid:
            print(f"Skipping {audio_file.name}: duration {duration:.2f}s")
            skipped += 1
            continue
        
        # Read and clean text
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            
            cleaned_text = clean_persian_text(text)
            if cleaned_text is None:
                print(f"Skipping {audio_file.name}: empty text")
                skipped += 1
                continue
                
            # Store relative path from project root
            relative_audio_path = f"dataset/{speaker_name}/wavs/{audio_file.name}"
            valid_pairs.append((relative_audio_path, cleaned_text))
            
        except Exception as e:
            print(f"Error reading text file {txt_file}: {e}")
            skipped += 1
            continue
    
    print(f"Valid samples: {len(valid_pairs)}, Skipped: {skipped}")
    
    if len(valid_pairs) == 0:
        raise ValueError("No valid audio-text pairs found")
    
    # Shuffle and split
    random.seed(1234)
    random.shuffle(valid_pairs)
    
    split_idx = int(len(valid_pairs) * train_ratio)
    train_pairs = valid_pairs[:split_idx]
    val_pairs = valid_pairs[split_idx:]
    
    print(f"Train samples: {len(train_pairs)}")
    print(f"Validation samples: {len(val_pairs)}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Write train filelist
    train_file = output_path / "train.txt"
    with open(train_file, 'w', encoding='utf-8') as f:
        for audio_path, text in train_pairs:
            f.write(f"{audio_path}|{text}\n")
    
    # Write validation filelist
    val_file = output_path / "val.txt"
    with open(val_file, 'w', encoding='utf-8') as f:
        for audio_path, text in val_pairs:
            f.write(f"{audio_path}|{text}\n")
    
    print(f"Created filelists:")
    print(f"  Train: {train_file}")
    print(f"  Validation: {val_file}")
    
    return train_file, val_file

def create_filelists_from_csv(csv_file, audio_base_dir, output_dir, 
                             audio_col='filename', text_col='text',
                             train_ratio=0.9, min_duration=0.5, max_duration=10.0):
    """
    Create filelists from a CSV file.
    """
    import pandas as pd
    
    df = pd.read_csv(csv_file)
    
    if audio_col not in df.columns or text_col not in df.columns:
        raise ValueError(f"CSV must contain '{audio_col}' and '{text_col}' columns")
    
    valid_pairs = []
    skipped = 0
    
    print(f"Processing {len(df)} CSV entries...")
    
    for idx, row in df.iterrows():
        audio_filename = row[audio_col]
        text = row[text_col]
        
        # Construct full audio path
        audio_path = Path(audio_base_dir) / audio_filename
        
        if not audio_path.exists():
            print(f"Audio file not found: {audio_path}")
            skipped += 1
            continue
        
        # Validate audio
        is_valid, duration = validate_audio_file(audio_path, min_duration, max_duration)
        
        if not is_valid:
            print(f"Skipping {audio_filename}: duration {duration:.2f}s")
            skipped += 1
            continue
        
        # Clean text
        cleaned_text = clean_persian_text(str(text))
        if cleaned_text is None:
            print(f"Skipping {audio_filename}: empty text")
            skipped += 1
            continue
        
        # Store relative path
        relative_path = str(Path(audio_base_dir).name / audio_filename)
        valid_pairs.append((relative_path, cleaned_text))
    
    print(f"Valid samples: {len(valid_pairs)}, Skipped: {skipped}")
    
    if len(valid_pairs) == 0:
        raise ValueError("No valid audio-text pairs found")
    
    # Shuffle and split
    random.seed(1234)
    random.shuffle(valid_pairs)
    
    split_idx = int(len(valid_pairs) * train_ratio)
    train_pairs = valid_pairs[:split_idx]
    val_pairs = valid_pairs[split_idx:]
    
    # Create output directory and write files
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    train_file = output_path / "train.txt"
    val_file = output_path / "val.txt"
    
    with open(train_file, 'w', encoding='utf-8') as f:
        for audio_path, text in train_pairs:
            f.write(f"{audio_path}|{text}\n")
    
    with open(val_file, 'w', encoding='utf-8') as f:
        for audio_path, text in val_pairs:
            f.write(f"{audio_path}|{text}\n")
    
    print(f"Created filelists:")
    print(f"  Train: {train_file} ({len(train_pairs)} samples)")
    print(f"  Validation: {val_file} ({len(val_pairs)} samples)")
    
    return train_file, val_file

def main():
    parser = argparse.ArgumentParser(description="Create filelists for Persian VITS training")
    parser.add_argument('--mode', choices=['directory', 'csv'], required=True,
                        help='Mode: directory or csv')
    parser.add_argument('--audio_dir', type=str, required=True,
                        help='Base directory containing audio files')
    parser.add_argument('--output_dir', type=str, default='filelists/fa_single',
                        help='Output directory for filelists')
    parser.add_argument('--speaker_name', type=str, default='amir',
                        help='Speaker name (for directory mode)')
    parser.add_argument('--csv_file', type=str,
                        help='CSV file path (for csv mode)')
    parser.add_argument('--audio_col', type=str, default='filename',
                        help='CSV column name for audio filename')
    parser.add_argument('--text_col', type=str, default='text',
                        help='CSV column name for text')
    parser.add_argument('--train_ratio', type=float, default=0.9,
                        help='Train/validation split ratio')
    parser.add_argument('--min_duration', type=float, default=0.5,
                        help='Minimum audio duration in seconds')
    parser.add_argument('--max_duration', type=float, default=10.0,
                        help='Maximum audio duration in seconds')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'directory':
            create_filelists_from_directory(
                args.audio_dir, args.output_dir, args.speaker_name,
                args.train_ratio, args.min_duration, args.max_duration
            )
        elif args.mode == 'csv':
            if not args.csv_file:
                raise ValueError("--csv_file is required for csv mode")
            create_filelists_from_csv(
                args.csv_file, args.audio_dir, args.output_dir,
                args.audio_col, args.text_col,
                args.train_ratio, args.min_duration, args.max_duration
            )
        
        print("\nFilelist creation completed successfully!")
        print("Next steps:")
        print("1. Check the generated filelists for correctness")
        print("2. Preprocess audio if needed: python preprocess_audio.py")
        print("3. Test compatibility: python test_compatibility.py")
        print("4. Start training: python train.py -c configs/fa_single_speaker.json -m fa_amir")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

# Example usage:
# python create_filelists.py --mode directory --audio_dir dataset --speaker_name amir
# python create_filelists.py --mode csv --csv_file data.csv --audio_dir dataset/amir/wavs
