import re
import unicodedata

AR2FA_MAP = {
    "ك": "ک", "ي": "ی", "ۀ": "ه", "ة": "ه",
    "ؤ": "و", "إ": "ا", "أ": "ا", "ٱ": "ا",
}

SAFE_PUNCTUATION = "،؛؟!.,؟!?"

def normalize_persian(text: str) -> str:
    for ar, fa in AR2FA_MAP.items():
        text = text.replace(ar, fa)
    text = "".join(ch for ch in unicodedata.normalize("NFD", text)
                   if unicodedata.category(ch) != "Mn")
    text = re.sub(r"[^آ-یA-Za-z0-9" + SAFE_PUNCTUATION + r"\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def persian_cleaners(text: str) -> str:
    return normalize_persian(text)
