import re
import unicodedata

# Arabic to Persian character mapping
AR2FA_MAP = {
    "ك": "ک",
    "ي": "ی",
    "ۀ": "ه",
    "ة": "ه",
    "ؤ": "و",
    "إ": "ا",
    "أ": "ا",
    "ٱ": "ا",
}

# Safe punctuation (kept in transcripts)
SAFE_PUNCTUATION = "،؛؟!.,؟!?"

def normalize_persian(text: str) -> str:
    """Normalize Persian text (chars, diacritics, spaces, punctuation)."""
    # Convert Arabic characters to Persian
    for ar, fa in AR2FA_MAP.items():
        text = text.replace(ar, fa)

    # Remove diacritics (tashdid, fatha, kasra, etc.)
    text = "".join(ch for ch in unicodedata.normalize("NFD", text)
                   if unicodedata.category(ch) != "Mn")

    # Keep only Persian/Latin letters, digits, and safe punctuation
    text = re.sub(r"[^آ-یA-Za-z0-9" + SAFE_PUNCTUATION + r"\s]", " ", text)

    # Normalize spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def persian_cleaners(text: str) -> str:
    """Pipeline for Persian text cleaning (grapheme-based)."""
    text = normalize_persian(text)
    return text

