"""
Persian/Farsi symbols for VITS TTS model.
Defines the set of symbols used in text input to the model for Persian language.
"""

# Padding symbol
_pad = "_"

# End of sequence symbol
_eos = "~"

# Persian letters (main alphabet)
_persian_letters = [
    "آ", "ا", "ب", "پ", "ت", "ث", "ج", "چ", "ح", "خ", 
    "د", "ذ", "ر", "ز", "ژ", "س", "ش", "ص", "ض", "ط", 
    "ظ", "ع", "غ", "ف", "ق", "ک", "گ", "ل", "م", "ن", 
    "و", "ه", "ی"
]

# Latin letters (for mixed Persian-English text)
_latin_letters = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", 
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", 
    "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"
]

# Digits
_digits = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

# Persian and common punctuation
_punctuation = [
    "!", "\"", "'", "(", ")", ",", "-", ".", ":", ";", "?", " ",
    "،", "؛", "؟"  # Persian-specific punctuation
]

# Additional Persian characters that might appear
_persian_extras = [
    "ء",  # hamza (though we usually remove it)
    "ة",  # taa marbutah (usually converted to ه)
    "ك",  # Arabic kaf (usually converted to ک)  
    "ي"   # Arabic yaa (usually converted to ی)
]

# Export all symbols
symbols = (
    [_pad, _eos] + 
    _persian_letters + 
    _latin_letters + 
    _digits + 
    _punctuation
)

# Special symbol indices for easy access
SPACE_ID = symbols.index(" ")
PAD_ID = symbols.index(_pad)
EOS_ID = symbols.index(_eos)

# Symbol mappings for debugging
symbol_to_id = {s: i for i, s in enumerate(symbols)}
id_to_symbol = {i: s for i, s in enumerate(symbols)}

def symbols_count():
    """Return the total number of symbols."""
    return len(symbols)

def print_symbols():
    """Print all symbols with their indices for debugging."""
    for i, symbol in enumerate(symbols):
        print(f"{i:3d}: '{symbol}'")

if __name__ == "__main__":
    print(f"Total symbols: {len(symbols)}")
    print(f"Persian letters: {len(_persian_letters)}")
    print(f"Latin letters: {len(_latin_letters)}")
    print(f"Digits: {len(_digits)}")
    print(f"Punctuation: {len(_punctuation)}")
    print("\nFirst 20 symbols:")
    for i in range(min(20, len(symbols))):
        print(f"{i:2d}: '{symbols[i]}'")
