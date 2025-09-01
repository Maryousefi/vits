"""
Persian/Farsi symbols for VITS TTS model.
Compatible with PyTorch 2.2.2 and hazm text processing.
"""

# Base symbols
_pad = '_'
_punctuation = ';:,.!?¡¿—…"«»"" '
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

# Persian letters (main Persian alphabet)
_persian_letters = "ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی"

# Persian specific characters and diacritics
_persian_diacritics = "َُِّْ"  # Fatha, Damma, Kasra, Shadda, Sukun
_persian_special = "ءآأإئؤةك"  # Hamza and special forms

# Numbers (both Persian/Arabic and English)
_numbers = "۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩0123456789"

# English letters (for mixed content)
_english_letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Combine all symbols for Persian VITS
persian_symbols = [_pad] + list(_punctuation) + list(_persian_letters) + list(_persian_diacritics) + list(_persian_special) + list(_numbers) + list(_english_letters)

# Export for use in other modules
symbols = persian_symbols

# Symbol mappings
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

def symbol_to_id(symbol):
    """Convert symbol to ID."""
    return _symbol_to_id.get(symbol, 0)  # Return 0 (pad) if symbol not found

def id_to_symbol(id):
    """Convert ID to symbol."""
    return _id_to_symbol.get(id, _pad)

def get_persian_symbols():
    """Return the Persian symbol set."""
    return symbols

def print_symbols():
    """Print all symbols with their IDs for debugging."""
    print(f"Total symbols: {len(symbols)}")
    for i, symbol in enumerate(symbols):
        print(f"{i:3d}: '{symbol}'")

if __name__ == "__main__":
    print_symbols()
