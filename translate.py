# Install: pip install indic-transliteration

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

def urdu_to_roman_indic(text):
    """
    Using indic-transliteration library
    """
    # First clean Urdu text
    
    # Transliterate
    roman = transliterate(
        text, 
        sanscript.,  # Source script
        sanscript.ITRANS  # Target roman scheme
        # Other schemes: HK, IAST, DEVANAGARI
    )
    return roman

# Usage
urdu_text = "غلام حسین، سیالکوٹ"
roman = urdu_to_roman_indic(urdu_text)
# Result: "ghulaam husain, siyaalkoT"