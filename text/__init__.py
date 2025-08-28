"""
Text processing for VITS with Persian/Farsi support.
"""

from text import cleaners
from text import cleaners_fa
from text.symbols import symbols as en_symbols
from text.symbols_fa import symbols as fa_symbols

# Map cleaner names to their functions
_cleaner_functions = {
    "english_cleaners": cleaners.english_cleaners,
    "english_cleaners2": cleaners.english_cleaners2,
    "transliteration_cleaners": cleaners.transliteration_cleaners,
    "basic_cleaners": cleaners.basic_cleaners,
    "persian_cleaners": cleaners_fa.persian_cleaners,
    "basic_persian_cleaners": cleaners_fa.basic_persian_cleaners,
}

# Default: English symbols  
symbols = en_symbols

def get_symbols(cleaner_name: str = None):
    """
    Return appropriate symbol set based on cleaner name.
    Updates the global symbols variable.
    """
    global symbols
    
    if cleaner_name and "persian" in cleaner_name.lower():
        symbols = fa_symbols
    else:
        symbols = en_symbols
    
    return symbols

def text_to_sequence(text, cleaners_names):
    """
    Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
        text: string to convert to a sequence
        cleaners_names: names of the cleaner functions to run the text through
    Returns:
        List of integers corresponding to the symbols in the text
    """
    sequence = []
    
    # Apply cleaners
    if isinstance(cleaners_names, str):
        cleaners_names = [cleaners_names]
    
    for cleaner_name in cleaners_names:
        if cleaner_name in _cleaner_functions:
            text = _cleaner_functions[cleaner_name](text)
        else:
            raise ValueError(f"Unknown cleaner: {cleaner_name}")
    
    # Convert text to symbol IDs
    for symbol in text:
        if symbol in symbols:
            sequence.append(symbols.index(symbol))
        else:
            # Handle unknown symbols - you can modify this behavior
            print(f"Warning: Unknown symbol '{symbol}' (ord: {ord(symbol)})")
            # Skip unknown symbols or replace with space
            if " " in symbols:
                sequence.append(symbols.index(" "))
    
    return sequence

def sequence_to_text(sequence):
    """
    Converts a sequence of IDs back to a string
    """
    result = ""
    for symbol_id in sequence:
        if symbol_id < len(symbols):
            result += symbols[symbol_id]
    return result

# For backward compatibility and convenience
def _clean_text(text, cleaners):
    """Apply cleaners to text."""
    if isinstance(cleaners, str):
        cleaners = [cleaners]
    
    for cleaner_name in cleaners:
        if cleaner_name in _cleaner_functions:
            text = _cleaner_functions[cleaner_name](text)
    return text
