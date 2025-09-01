import re
import unicodedata
from unidecode import unidecode
try:
    from hazm import Normalizer, word_tokenize
    HAZM_AVAILABLE = True
    # Initialize normalizer globally to avoid repeated initialization
    _normalizer = Normalizer(
        persian_style=True,
        persian_digit=True,
        remove_diacritics=False,  # Keep some diacritics for better pronunciation
        remove_specials_chars=False,
        decrease_repeated_chars=True
    )
except ImportError:
    HAZM_AVAILABLE = False
    _normalizer = None
    print("Warning: hazm not available, using basic Persian cleaning")

# Arabic to Farsi character mapping (comprehensive)
AR2FA_MAP = {
    "ك": "ک", "ي": "ی", "ۀ": "ه", "ة": "ه",
    "ؤ": "و", "إ": "ا", "أ": "ا", "ٱ": "ا",
    "ئ": "ی", "ء": "",  # Remove standalone hamza
    "ى": "ی", "ة": "ه", "ۃ": "ه"
}

# Safe punctuation that should be preserved
SAFE_PUNCTUATION = "،؛؟!.\"'(),-:; "

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

def normalize_persian_manual(text: str) -> str:
    """Manual Persian text normalization when hazm is not available."""
    # Convert Arabic characters to Farsi
    for ar, fa in AR2FA_MAP.items():
        text = text.replace(ar, fa)
    
    # Normalize digits
    text = normalize_digits(text)
    
    # Handle Zero Width Non-Joiner (ZWNJ) - keep it for proper Persian rendering
    text = text.replace('\u200c', '\u200c')  # Preserve ZWNJ
    
    # Remove other zero-width characters
    text = text.replace('\u200d', '')  # Zero Width Joiner
    text = text.replace('\u200b', '')  # Zero Width Space
    text = text.replace('\ufeff', '')  # Byte Order Mark
    
    # Keep only allowed characters: Persian letters, English letters, digits, and safe punctuation
    # Persian Unicode ranges: 
    # \u0600-\u06FF (Arabic/Persian block)
    # \u0750-\u077F (Arabic Supplement)
    # \uFB50-\uFDFF (Arabic Presentation Forms-A)
    # \uFE70-\uFEFF (Arabic Presentation Forms-B)
    allowed_pattern = r"[^\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFFA-Za-z0-9\u200c" + re.escape(SAFE_PUNCTUATION) + r"]"
    text = re.sub(allowed_pattern, " ", text)
    
    # Collapse multiple spaces but preserve single spaces
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

def remove_extra_spaces(text: str) -> str:
    """Remove extra whitespace and normalize spacing."""
    # Replace tabs and newlines with spaces
    text = re.sub(r'[\t\n\r]+', ' ', text)
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing spaces
    return text.strip()

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
    """
    Main Persian text cleaning pipeline using hazm if available.
    This is the primary cleaner for Persian VITS.
    """
    if not text or not text.strip():
        return " "  # Return single space for empty text
    
    if HAZM_AVAILABLE and _normalizer:
        try:
            # Use hazm normalizer for proper Persian text handling
            text = _normalizer.normalize(text)
        except Exception as e:
            print(f"Hazm normalization failed: {e}, falling back to manual")
            text = normalize_persian_manual(text)
    else:
        # Fallback to manual normalization
        text = normalize_persian_manual(text)
    
    # Convert Latin to lowercase
    text = lowercase_latin(text)
    
    # Remove extra spaces
    text = remove_extra_spaces(text)
    
    # Final validation - ensure text is not empty
    if not text.strip():
        text = " "
    
    return text

def basic_persian_cleaners(text: str) -> str:
    """
    Basic Persian cleaning without aggressive normalization.
    Useful for preserving more of the original text structure.
    """
    if not text or not text.strip():
        return " "
    
    if HAZM_AVAILABLE and _normalizer:
        try:
            # Use only basic hazm functions
            text = _normalizer.character_refinement(text)
            text = _normalizer.affix_spacing(text)
        except Exception as e:
            print(f"Basic hazm processing failed: {e}")
    
    # Always apply digit normalization
    text = normalize_digits(text)
    
    # Basic space cleanup
    text = remove_extra_spaces(text)
    
    # Ensure text is not empty
    if not text.strip():
        text = " "
        
    return text

def minimal_persian_cleaners(text: str) -> str:
    """
    Minimal cleaning - only essential normalization.
    Use when you want to preserve most of the original text.
    """
    if not text or not text.strip():
        return " "
    
    # Only normalize digits and basic spacing
    text = normalize_digits(text)
    text = remove_extra_spaces(text)
    
    if not text.strip():
        text = " "
        
    return text

def test_cleaners():
    """Test function to verify all cleaners work correctly."""
    test_texts = [
        "سلام دنیا! چطوری؟",
        "این متن فارسی است با ۱۲۳ عدد",
        "Mixed text with English and فارسی",
        "ك ي ۀ",  # Arabic characters that should be converted
        "متن    با    فاصله‌های    زیاد",
        "text with ١٢٣ arabic digits",
        "",  # Empty text test
        "   ",  # Whitespace only test
    ]
    
    cleaners = [
        ("persian_cleaners", persian_cleaners),
        ("basic_persian_cleaners", basic_persian_cleaners),
        ("minimal_persian_cleaners", minimal_persian_cleaners)
    ]
    
    print("Testing Persian cleaners:")
    print("=" * 50)
    
    for cleaner_name, cleaner_func in cleaners:
        print(f"\n{cleaner_name}:")
        print("-" * 30)
        for i, text in enumerate(test_texts):
            try:
                cleaned = cleaner_func(text)
                print(f"  Test {i+1:2d}: '{text}' -> '{cleaned}'")
            except Exception as e:
                print(f"  Test {i+1:2d}: ERROR - {e}")

# Export all cleaners for use in other modules
__all__ = [
    'persian_cleaners', 
    'basic_persian_cleaners', 
    'minimal_persian_cleaners',
    'normalize_digits',
    'normalize_persian_manual'
]

if __name__ == "__main__":
    test_cleaners()
