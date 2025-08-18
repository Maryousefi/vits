
_pad = "_"
_eos = "~"

# Persian alphabet (grapheme-based)
_persian_letters = list("آابپتثجچحخدذرزسشصضطظعغفقکگلمنوهی")

# Latin letters and digits
_latin_letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
_digits = list("0123456789")

# Basic punctuation
_punctuation = list("!\"'(),-.:;? ")

# Combine all
symbols = [_pad, _eos] + _persian_letters + _latin_letters + _digits + _punctuation

