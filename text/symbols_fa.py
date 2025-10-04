"""
Persian/Farsi symbols for VITS TTS model.
Compatible with PyTorch and hazm text processing.
"""

# Base symbols
_pad = "_"
_punctuation = ';:,.!?¡¿—…"«»"\'()[] '
# Persian letters (main Persian alphabet)
_persian_letters = "ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی"
# Persian diacritics (keep optional ones)
_persian_diacritics = "َُِّْ"  # Fatha, Damma, Kasra, Shadda, Sukun
# Persian special characters
_persian_special = "ءآأإئؤةك"  # Hamza and special forms
# English digits only (your cleaner normalizes all digits here)
_numbers = "0123456789"
# English letters (for mixed content)
_english_letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
# (Optional) IPA phoneme set if you want phoneme-level training later
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

# Final symbol list
symbols = (
    [_pad]
    + list(_punctuation)
    + list(_persian_letters)
    + list(_persian_diacritics)
    + list(_persian_special)
    + list(_numbers)
    + list(_english_letters)
    # If you want phonemes, uncomment the next line:
    # + list(_letters_ipa)
)

# Symbol mappings
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

def symbol_to_id(symbol):
    """Convert symbol to ID."""
    return _symbol_to_id.get(symbol, 0)  # fallback to pad (0)

def id_to_symbol(id):
    """Convert ID to symbol."""
    return _id_to_symbol.get(id, _pad)

def get_persian_symbols():
    """Return the Persian symbol set."""
    return symbols

def print_symbols():
    """Debug: print all symbols with IDs."""
    print(f"Total symbols: {len(symbols)}")
    for i, s in enumerate(symbols):
        print(f"{i:3d}: '{s}'")

if __name__ == "__main__":
    print_symbols()
