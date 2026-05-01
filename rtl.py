from bidi.algorithm import get_display
import arabic_reshaper

def to_rtl(text: str) -> str:
    if not text:
        return text

    # reshape Arabic/Urdu letters
    reshaped = arabic_reshaper.reshape(text)

    # apply RTL display order
    rtl_text = get_display(reshaped)

    return rtl_text


# Example usage
sample = "موجودہ پت:مکان نمبر 1184، محلّہ عبداللہ بنگلوز سرجانی\nﯽﻄﺳﻭ ﯽﭘﺍﺮﮐ ،ﻥﺅﺎﭨ\nمستفل پتہ: مکان نمبر 114-۶، محلّہ عبداللہ بنگلوز سرجانی\nﯽﻄﺳﻭ ﯽﭘﺍﺮﮐ ،ﻥﺅﺎﻧ\nﻻﻮﮩﺼﻌﻣ ﻻ۔۸ﺎﻨﻌﻌﻣ\n2000 ۵ ۱ا0il aadarpe۲ AeAa0 ۵ (2)0 A0-66 1010ا 2c05ا"
print("Original:", sample)
print("RTL:", to_rtl(sample))