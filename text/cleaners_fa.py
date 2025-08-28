import re
import unicodedata

# Arabic to Farsi character mapping
AR2FA_MAP = {
    "ك": "ک", "ي": "ی", "ۀ": "ه", "ة": "ه",
    "ؤ": "و", "إ": "ا", "أ": "ا", "ٱ": "ا",
    "ئ": "ی", "ء": "",  # Remove hamza
}

# Safe punctuation that should be preserved
SAFE_PUNCTUATION = "،؛؟!.\"'"

# Farsi digits to English digits
FARSI_DIGITS = {
    '۰': '0', '۱': '1', '۲': '2', '۳': '3', '۴': '4',
    '۵': '5', '۶': '6', '۷': '7', '۸': '8', '۹': '9'
}

# Arabic digits to English digits  
ARABIC_DIGITS = {
    '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
    '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9'
}

def normalize_digits(text: str) -> str:
    """Convert Farsi and Arabic digits to English digits."""
    for fa_digit, en_digit in FARSI_DIGITS.items():
        text = text.replace(fa_digit, en_digit)
    for ar_digit, en_digit in ARABIC_DIGITS.items():
        text = text.replace(ar_digit, en_digit)
    return text

def normalize_persian(text: str) -> str:
    """Normalize Persian text by converting Arabic chars and cleaning."""
    # Convert Arabic characters to Farsi
    for ar, fa in AR2FA_MAP.items():
        text = text.replace(ar, fa)
    
    # Normalize digits
    text = normalize_digits(text)
    
    # Remove diacritics (optional - you might want to keep some)
    text = "".join(ch for ch in unicodedata.normalize("NFD", text)
                   if unicodedata.category(ch) != "Mn")
    
    # Keep only Farsi letters, English letters, digits, and safe punctuation
    text = re.sub(r"[^آ-یA-Za-z0-9" + re.escape(SAFE_PUNCTUATION) + r"\s]", " ", text)
    
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

def remove_extra_spaces(text: str) -> str:
    """Remove extra whitespace."""
    return re.sub(r'\s+', ' ', text).strip()

def lowercase_latin(text: str) -> str:
    """Convert Latin characters to lowercase while preserving Persian."""
    result = ""
    for char in text:
        if 'A' <= char <= 'Z':
            result += char.lower()
        else:
            result += char
    return result

def persian_cleaners(text: str) -> str:
    """Main Persian text cleaning pipeline."""
    text = normalize_persian(text)
    text = lowercase_latin(text)
    text = remove_extra_spaces(text)
    return text

def basic_persian_cleaners(text: str) -> str:
    """Basic Persian cleaning without aggressive normalization."""
    text = normalize_digits(text)
    text = remove_extra_spaces(text)
    return text
