import re
import unicodedata
from unidecode import unidecode

try:
    from hazm import Normalizer, word_tokenize
    HAZM_AVAILABLE = True

    # Try modern hazm args, fallback gracefully
    try:
        _normalizer = Normalizer(
            remove_diacritics=True,
            persian_numbers=True,
            unicodes_replacement=True,
            character_refinement=True,
        )
    except TypeError:
        # For older hazm
        _normalizer = Normalizer()

except ImportError:
    HAZM_AVAILABLE = False
    _normalizer = None
    print(" hazm not available, falling back to manual Persian cleaning.")

# Arabic → Farsi mapping
AR2FA_MAP = {
    "ك": "ک", "ي": "ی", "ۀ": "ه", "ة": "ه",
    "ؤ": "و", "إ": "ا", "أ": "ا", "ٱ": "ا",
    "ئ": "ی", "ء": "", "ى": "ی", "ۃ": "ه",
}

SAFE_PUNCTUATION = "،؛؟!.\"'(),-:; "

FARSI_DIGITS = {
    '۰': '0', '۱': '1', '۲': '2', '۳': '3', '۴': '4',
    '۵': '5', '۶': '6', '۷': '7', '۸': '8', '۹': '9'
}
ARABIC_DIGITS = {
    '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
    '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9'
}


def normalize_digits(text: str) -> str:
    for fa, en in FARSI_DIGITS.items():
        text = text.replace(fa, en)
    for ar, en in ARABIC_DIGITS.items():
        text = text.replace(ar, en)
    return text


def normalize_persian_manual(text: str) -> str:
    for ar, fa in AR2FA_MAP.items():
        text = text.replace(ar, fa)

    text = normalize_digits(text)

    # Handle zero-width chars
    text = text.replace('\u200c', '\u200c')  # keep ZWNJ
    text = text.replace('\u200d', '')
    text = text.replace('\u200b', '')
    text = text.replace('\ufeff', '')

    allowed_pattern = (
        r"[^\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFFA-Za-z0-9\u200c"
        + re.escape(SAFE_PUNCTUATION)
        + r"]"
    )
    text = re.sub(allowed_pattern, " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_extra_spaces(text: str) -> str:
    text = re.sub(r'[\t\n\r]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def lowercase_latin(text: str) -> str:
    return "".join(c.lower() if 'A' <= c <= 'Z' else c for c in text)


def persian_cleaners(text: str) -> str:
    if not text or not text.strip():
        return " "

    if HAZM_AVAILABLE and _normalizer:
        try:
            text = _normalizer.normalize(text)
        except Exception as e:
            print(f" Hazm normalization failed: {e}, using manual.")
            text = normalize_persian_manual(text)
    else:
        text = normalize_persian_manual(text)

    text = lowercase_latin(text)
    text = remove_extra_spaces(text)

    return text if text.strip() else " "


def basic_persian_cleaners(text: str) -> str:
    if not text or not text.strip():
        return " "

    if HAZM_AVAILABLE and _normalizer:
        try:
            text = _normalizer.character_refinement(text)
            text = _normalizer.affix_spacing(text)
        except Exception as e:
            print(f" Basic hazm failed: {e}")

    text = normalize_digits(text)
    text = remove_extra_spaces(text)

    return text if text.strip() else " "


def minimal_persian_cleaners(text: str) -> str:
    if not text or not text.strip():
        return " "
    text = normalize_digits(text)
    text = remove_extra_spaces(text)
    return text if text.strip() else " "


def test_cleaners():
    test_texts = [
        "سلام دنیا! چطوری؟",
        "این متن فارسی است با ۱۲۳ عدد",
        "Mixed text with English and فارسی",
        "ك ي ۀ",  # Arabic forms
        "متن    با    فاصله‌های    زیاد",
        "text with ١٢٣ arabic digits",
        "", "   "
    ]

    cleaners = [
        ("persian_cleaners", persian_cleaners),
        ("basic_persian_cleaners", basic_persian_cleaners),
        ("minimal_persian_cleaners", minimal_persian_cleaners)
    ]

    print("Testing Persian cleaners:\n" + "=" * 50)
    for name, func in cleaners:
        print(f"\n{name}:\n" + "-" * 30)
        for i, txt in enumerate(test_texts):
            try:
                print(f"  {i+1:02d}: '{txt}' -> '{func(txt)}'")
            except Exception as e:
                print(f"  {i+1:02d}: ERROR - {e}")


__all__ = [
    "persian_cleaners",
    "basic_persian_cleaners",
    "minimal_persian_cleaners",
    "normalize_digits",
    "normalize_persian_manual",
]

if __name__ == "__main__":
    test_cleaners()
