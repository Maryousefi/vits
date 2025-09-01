"""
Text processing module for Persian VITS.
Handles symbol loading and text-to-sequence conversion.
"""

import os
import sys
import importlib

# Add the text directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import cleaners
from .cleaners_fa import persian_cleaners, basic_persian_cleaners

# Cleaner mapping
_cleaners = {
    'persian_cleaners': persian_cleaners,
    'basic_persian_cleaners': basic_persian_cleaners,
}

def get_symbols(cleaner_names):
    """
    Get symbols based on the cleaner names.
    """
    if isinstance(cleaner_names, str):
        cleaner_names = [cleaner_names]
    
    # For Persian cleaners, always use Persian symbols
    if any('persian' in name for name in cleaner_names):
        from .symbols_fa import get_persian_symbols
        return get_persian_symbols()
    else:
        # Fallback to basic Persian symbols
        from .symbols_fa import get_persian_symbols
        return get_persian_symbols()

def text_to_sequence(text, cleaner_names):
    """
    Converts text to a sequence of IDs corresponding to the symbols.
    
    Args:
        text: string to convert to a sequence
        cleaner_names: list of cleaner names to apply
        
    Returns:
        List of integers corresponding to the symbols
    """
    if isinstance(cleaner_names, str):
        cleaner_names = [cleaner_names]
    
    # Clean the text
    cleaned_text = text
    for cleaner_name in cleaner_names:
        if cleaner_name in _cleaners:
            cleaned_text = _cleaners[cleaner_name](cleaned_text)
        else:
            print(f"Warning: Cleaner '{cleaner_name}' not found, using persian_cleaners")
            cleaned_text = persian_cleaners(cleaned_text)
    
    # Get symbols based on cleaners
    symbols = get_symbols(cleaner_names)
    symbol_to_id = {s: i for i, s in enumerate(symbols)}
    
    # Convert text to sequence
    sequence = []
    for char in cleaned_text:
        if char in symbol_to_id:
            sequence.append(symbol_to_id[char])
        else:
            # Use pad token for unknown characters
            sequence.append(0)
            print(f"Warning: Unknown character '{char}' (ord: {ord(char)}) replaced with pad token")
    
    return sequence

def sequence_to_text(sequence, cleaner_names):
    """
    Converts a sequence of IDs back to text.
    
    Args:
        sequence: list of integers
        cleaner_names: list of cleaner names (for symbol loading)
        
    Returns:
        String representation
    """
    symbols = get_symbols(cleaner_names)
    result = ''
    for symbol_id in sequence:
        if 0 <= symbol_id < len(symbols):
            result += symbols[symbol_id]
    return result

# Legacy support 
def _clean_text(text, cleaner_names):
    """Apply text cleaners."""
    return text_to_sequence(text, cleaner_names)

# For backward compatibility
from .symbols_fa import get_persian_symbols

def test_text_processing():
    """Test the text processing pipeline."""
    print("Testing Persian text processing...")
    
    test_texts = [
        "سلام دنیا!",
        "این یک تست است.",
        "۱۲۳ عدد فارسی",
        "Mixed text with فارسی"
    ]
    
    for text in test_texts:
        print(f"Original: '{text}'")
        sequence = text_to_sequence(text, ['persian_cleaners'])
        reconstructed = sequence_to_text(sequence, ['persian_cleaners'])
        print(f"Sequence length: {len(sequence)}")
        print(f"Reconstructed: '{reconstructed}'")
        print()

if __name__ == "__main__":
    test_text_processing()
