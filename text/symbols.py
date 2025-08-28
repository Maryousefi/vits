""" 
English symbols for VITS TTS model.
Based on the original tacotron symbols.
"""

# Padding symbol
_pad = '_'

# Punctuation 
_punctuation = ';:,.!?¡¿—…"«»"" '

# Letters
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

# IPA symbols for English phonemes
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

# Export all symbols
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)

# Special symbol IDs
SPACE_ID = symbols.index(" ")
PAD_ID = symbols.index(_pad)

# Helper functions
def symbol_to_id(symbol):
    """Convert symbol to ID."""
    return symbols.index(symbol) if symbol in symbols else symbols.index(' ')

def id_to_symbol(symbol_id):
    """Convert ID to symbol."""
    return symbols[symbol_id] if 0 <= symbol_id < len(symbols) else ' '
