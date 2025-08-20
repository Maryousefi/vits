from text import cleaners
from text import cleaners_fa
from text.symbols import symbols as en_symbols
from text.symbols_fa import symbols as fa_symbols

# Map cleaner names to functions
cleaner_names = {
    "english_cleaners": cleaners.english_cleaners,
    "english_cleaners2": cleaners.english_cleaners2,
    "persian_cleaners": cleaners_fa.persian_cleaners,
}

# Default: English symbols
symbols = en_symbols


def get_symbols(cleaner_name: str):
    """Return symbol set depending on cleaner."""
    global symbols
    if cleaner_name == "persian_cleaners":
        symbols = fa_symbols
    else:
        symbols = en_symbols
    return symbols
