"""
text/__init__.py

Text processing bridge for Persian VITS.
Exposes:
 - get_symbols(cleaner_names) -> list of symbols
 - text_to_sequence(text, cleaner_names) -> list[int]
 - sequence_to_text(sequence, cleaner_names) -> str

This module is defensive: if an English `text/symbols.py` exists it will use it for non-Persian cleaners.
Otherwise it falls back to Persian symbols or a minimal ASCII symbol set.
"""

import os
import sys

# make sure text package directory is importable
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

# Import Persian cleaners
from .cleaners_fa import (
    persian_cleaners,
    basic_persian_cleaners,
    minimal_persian_cleaners
)

_cleaners = {
    "persian_cleaners": persian_cleaners,
    "basic_persian_cleaners": basic_persian_cleaners,
    "minimal_persian_cleaners": minimal_persian_cleaners,
}


def _load_english_symbols():
    """
    Try to import the original English symbols (text/symbols.py).
    If not available, return a simple fallback list.
    """
    try:
        from .symbols import symbols as en_symbols  # original repo symbols.py
        return en_symbols
    except Exception:
        # basic fallback (pad + punctuation + lowercase ascii + digits)
        _pad = "_"
        _punctuation = "!?.,:;\"'()[] "
        _letters = "abcdefghijklmnopqrstuvwxyz"
        _numbers = "0123456789"
        return [_pad] + list(_punctuation) + list(_letters) + list(_numbers)


def get_symbols(cleaner_names):
    """
    Return the symbol list for the provided cleaner(s).
    cleaner_names may be a string or a list of strings.
    """
    if isinstance(cleaner_names, str):
        cleaner_names = [cleaner_names]

    # if any cleaner name contains 'persian' assume Persian symbol set
    for name in cleaner_names:
        if "persian" in name.lower():
            try:
                from .symbols_fa import get_persian_symbols
                return get_persian_symbols()
            except Exception as e:
                # fallback if symbols_fa.py is missing
                print(" Warning: Persian symbols not found, falling back to English-like set:", e)
                return _load_english_symbols()

    # otherwise return english symbols if available
    return _load_english_symbols()


def text_to_sequence(text, cleaner_names):
    """
    Convert text to numeric sequence (list of symbol IDs) using specified cleaners.
    """
    if isinstance(cleaner_names, str):
        cleaner_names = [cleaner_names]

    cleaned = text
    for cname in cleaner_names:
        if cname in _cleaners:
            cleaned = _cleaners[cname](cleaned)
        else:
            # unknown cleaner -> warn and fallback to persian_cleaners
            print(f" Warning: cleaner '{cname}' not found. Falling back to persian_cleaners.")
            cleaned = persian_cleaners(cleaned)

    symbols = get_symbols(cleaner_names)
    symbol_to_id = {s: i for i, s in enumerate(symbols)}

    seq = []
    for ch in cleaned:
        if ch in symbol_to_id:
            seq.append(symbol_to_id[ch])
        else:
            # unknown char -> use pad id (0)
            seq.append(0)
    return seq


def sequence_to_text(sequence, cleaner_names):
    """
    Convert numeric sequence back to text (useful for debugging).
    """
    symbols = get_symbols(cleaner_names)
    id_to_symbol = {i: s for i, s in enumerate(symbols)}
    out = []
    for idx in sequence:
        if 0 <= int(idx) < len(symbols):
            out.append(id_to_symbol[int(idx)])
    return "".join(out)


# expose common names used in repo:
__all__ = [
    "get_symbols",
    "text_to_sequence",
    "sequence_to_text",
    "persian_cleaners",
    "basic_persian_cleaners",
    "minimal_persian_cleaners",
]
