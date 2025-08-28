"""
Text processing for VITS with Persian/Farsi support.
"""
from text import cleaners

# Try to import Persian cleaners
try:
    from text import cleaners_fa
except ImportError:
    print("Warning: Persian cleaners not available")
    cleaners_fa = None

# Import symbol sets
try:
    from text.symbols import symbols as en_symbols
except ImportError:
    # Fallback English symbols
    en_symbols = ['_', '-', '!', "'", '(', ')', ',', '.', ':', ';', '?', ' '] + \
                 [chr(i) for i in range(ord('A'), ord('Z')+1)] + \
                 [chr(i) for i in range(ord('a'), ord('z')+1)]

try:
    from text.symbols_fa import symbols as fa_symbols
except ImportError:
    print("Warning: Persian symbols not available")
    fa_symbols = en_symbols  # fallback

# Map cleaner names to their functions
_cleaner_functions = {
    "english_cleaners": cleaners.english_cleaners,
    "transliteration_cleaners": cleaners.transliteration_cleaners,
    "basic_cleaners": cleaners.basic_cleaners,
}

# Add Persian cleaners if available
if cleaners_fa:
    _cleaner_functions.update({
        "persian_cleaners": cleaners_fa.persian_cleaners,
        "basic_persian_cleaners": cleaners_fa.basic_persian_cleaners,
    })

# Default: English symbols  
symbols = en_symbols

def get_symbols(cleaner_names=None):
    """
    Return appropriate symbol set based on cleaner name.
    Updates the global symbols variable.
    """
    global symbols
    
    if cleaner_names:
        if isinstance(cleaner_names, str):
            cleaner_names = [cleaner_names]
        
        # Check if any cleaner is Persian
        for cleaner_name in cleaner_names:
            if cleaner_name and "persian" in cleaner_name.lower():
                symbols = fa_symbols
                return symbols
    
    # Default to English
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
    # Update symbols based on cleaners
    current_symbols = get_symbols(cleaners_names)
    
    sequence = []
    
    # Apply cleaners
    if isinstance(cleaners_names, str):
        cleaners_names = [cleaners_names]
    
    for cleaner_name in cleaners_names:
        if cleaner_name in _cleaner_functions:
            text = _cleaner_functions[cleaner_name](text)
        else:
            print(f"Warning: Unknown cleaner '{cleaner_name}', skipping...")
    
    # Convert text to symbol IDs
    for symbol in text:
        if symbol in current_symbols:
            sequence.append(current_symbols.index(symbol))
        else:
            # Handle unknown symbols
            print(f"Warning: Unknown symbol '{symbol}' (ord: {ord(symbol)})")
            # Replace with space if available, otherwise skip
            if " " in current_symbols:
                sequence.append(current_symbols.index(" "))
    
    return sequence

def sequence_to_text(sequence):
    """
    Converts a sequence of IDs back to a string
    """
    result = ""
    for symbol_id in sequence:
        if 0 <= symbol_id < len(symbols):
            result += symbols[symbol_id]
        else:
            result += "?"  # Unknown symbol placeholder
    return result

# For backward compatibility and convenience
def clean_text(text, cleaners_names):
    """Apply cleaners to text."""
    if isinstance(cleaners_names, str):
        cleaners_names = [cleaners_names]
    
    for cleaner_name in cleaners_names:
        if cleaner_name in _cleaner_functions:
            text = _cleaner_functions[cleaner_name](text)
    return text

# Export main symbols for external use
__all__ = ['symbols', 'text_to_sequence', 'sequence_to_text', 'clean_text', 'get_symbols']
