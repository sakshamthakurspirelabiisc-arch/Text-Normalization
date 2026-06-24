import re
from indic_numtowords import num2words

# ----------------------------
# Kannada month names
# ----------------------------
months_kannada = {
    1: "ಜನವರಿ", 2: "ಫೆಬ್ರವರಿ", 3: "ಮಾರ್ಚ್", 4: "ಏಪ್ರಿಲ್",
    5: "ಮೇ", 6: "ಜೂನ್", 7: "ಜುಲೈ", 8: "ಆಗಸ್ಟ್",
    9: "ಸೆಪ್ಟೆಂಬರ್", 10: "ಅಕ್ಟೋಬರ್", 11: "ನವೆಂಬರ್", 12: "ಡಿಸೆಂಬರ್"
}

# English month mapping
month_name_map = {
    "january": 1,"february":2,"march":3,"april":4,"may":5,"june":6,
    "july":7,"august":8,"september":9,"october":10,"november":11,"december":12
}

# ----------------------------
# Number to Kannada words
# ----------------------------
# ----------------------------
# Number to Kannada words (decimal-safe)
# ----------------------------
def convert_number_to_words(num_str):
    """
    Converts a number (int or decimal as string) into Kannada words.
    Handles:
      - Integers up to 999,999
      - Decimal numbers (3.14 → ಮೂರು ದಶಮಾಂಶ ಒಂದು ನಾಲ್ಕು)
      - Works for string or int inputs
    """
    num_str = str(num_str).strip()
    
    if not num_str:
        return ""
    
    # Handle decimal numbers
    if '.' in num_str:
        integer_part, decimal_part = num_str.split('.')
        integer_part = integer_part if integer_part else "0"
        
        # Convert integer part recursively
        int_words = convert_number_to_words(integer_part) if integer_part.isdigit() else integer_part
        
        # Convert decimal part digit by digit
        decimal_words = " ".join(num2words(int(d), lang='kn') for d in decimal_part if d.isdigit())
        
        if int_words and decimal_words:
            return f"{int_words} ದಶಮಾಂಶ {decimal_words}".strip()
        elif int_words:
            return int_words.strip()
        elif decimal_words:
            return f"ಸೊನ್ನೆ ದಶಮಾಂಶ {decimal_words}".strip()
        else:
            return ""
    
    # Handle integer numbers
    elif num_str.isdigit():
        num = int(num_str)
        if num == 0:
            return "ಸೊನ್ನೆ"
        
        words = []
        
        # Handle thousands (up to 999,999)
        if num >= 1000:
            thousands = num // 1000
            words.append(f"{convert_number_to_words(thousands)} ಸಾವಿರ")
            num = num % 1000
        
        # Handle hundreds
        if num >= 100:
            hundreds = num // 100
            words.append(f"{num2words(hundreds, lang='kn')} ನೂರು")
            num = num % 100
        
        # Remaining tens and units
        if num > 0:
            words.append(num2words(num, lang='kn'))
        
        return " ".join(words).strip()
    
    # Fallback: non-digit string
    else:
        return num_str.strip()


# ----------------------------
# Phone number to Kannada words
# ----------------------------
def number_to_kannada_phone(num_str):
    output_parts = []
    if num_str.startswith("+"):
        output_parts.append("ಪ್ಲಸ್")
        num_str = num_str[1:]
    output_parts.extend(num2words(int(d), lang='kn') for d in num_str)
    return " ".join(output_parts)

# ----------------------------
# Year normalization
# ----------------------------
def normalize_year(year: int) -> int:
    if year < 100:
        return 2000 + year if year <= 30 else 1900 + year
    if year < 1000:
        return 2000 + (year % 100)
    return year

def convert_year_to_words(year: int) -> str:
    year = normalize_year(year)
    if year < 1000:
        return convert_number_to_words(year)
    thousands = year // 1000
    remainder = year % 1000
    parts = []
    if thousands > 0:
        parts.append(f"{convert_number_to_words(thousands)} ಸಾವಿರ")
    if remainder >= 100:
        hundreds = remainder // 100
        parts.append(f"{convert_number_to_words(hundreds)} ನೂರು")
        remainder = remainder % 100
    if remainder > 0:
        parts.append(convert_number_to_words(remainder))
    return " ".join(parts)

# ----------------------------
# Time normalization (fixed pm/am stuck to numbers)
# ----------------------------
def replace_time(text):
    # Match only HH:MM or HH:MM:SS with optional AM/PM
    time_pattern = r"\b(\d{1,2}):(\d{2})(?::(\d{2}))?\s*(am|pm|AM|PM)?\b"
    
    def repl(match):
        hour, minute, sec, meridian = int(match.group(1)), int(match.group(2)), match.group(3), match.group(4)
        words = ""
        if meridian:
            period = "ಬೆಳಗ್ಗೆ" if meridian.lower() == "am" else "ಸಂಜೆ"
            words += f"{period} "
        words += f"{convert_number_to_words(hour)} ಗಂಟೆಗೆ {convert_number_to_words(minute)} ನಿಮಿಷ"
        if sec:
            words += f" {convert_number_to_words(int(sec))} ಸೆಕೆಂಡ್"
        return words

    return re.sub(time_pattern, repl, text)


# ----------------------------
# Measurements
# ----------------------------
def replace_measurements(text):
    measure_units = {
        "kg": "ಕಿಲೋಗ್ರಾಂ",
        "m\\^2": "ವರ್ಗ ಮೀಟರ್",
        "km\\^2": "ವರ್ಗ ಕಿಲೋಮೀಟರ್",
        "m\\^3": "ಘನ ಮೀಟರ್",
        "km\\^3": "ಘನ ಕಿಲೋಮೀಟರ್",
        "km/hr": "ಕಿಲೋಮೀಟರ್ ಪ್ರತಿ ಗಂಟೆ",
        "m/s": "ಮೀಟರ್ ಪ್ರತಿ ಸೆಕೆಂಡ್",
        "km/s": "ಕಿಲೋಮೀಟರ್ ಪ್ರತಿ ಸೆಕೆಂಡ್",
        "m/min": "ಮೀಟರ್ ಪ್ರತಿ ನಿಮಿಷ",
        "km/min": "ಕಿಲೋಮೀಟರ್ ಪ್ರತಿ ನಿಮಿಷ",
        "m": "ಮೀಟರ್",
        "km": "ಕಿಲೋಮೀಟರ್",
        "ft": "ಫೀಟ್",
        "ಫೀಟ್": "ಫೀಟ್",
        "ಮೀಟರ್": "ಮೀಟರ್"
    }
    for unit, kannada_unit in measure_units.items():
        pattern = rf"\b([0-9೦-೯]+(?:\.[0-9೦-೯]+)?)\s*{unit}\b"
        def repl(m):
            num = m.group(1)
            return f"{convert_number_to_words(num)} {kannada_unit}"
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    return text

# ----------------------------
# Percentages
# ----------------------------
def replace_percentage(text):
    percent_pattern = r"(\d+(?:\.\d+)?)\s*%"
    return re.sub(percent_pattern, lambda m: f"{convert_number_to_words(m.group(1))} ಪ್ರತಿಶತ", text)

# ----------------------------
# Fractions
# ----------------------------
def replace_fractions_or_dates(text):
    fraction_pattern = r"(\d+)/(\d+)"
    return re.sub(fraction_pattern, lambda m: f"{convert_number_to_words(int(m.group(2)))}ರಲ್ಲಿ {convert_number_to_words(int(m.group(1)))}", text)

# ----------------------------
# Money normalization
# ----------------------------
def replace_money(text):
    money_pattern = r"(₹|Rs\.?|Rs|\$|£|€)\s*(\d+(?:,\d{2,3})*(?:\.\d+)?)"
    
    def repl(match):
        symbol, num_str = match.groups()
        num_str = num_str.replace(",", "")  # Remove commas
        word_num = convert_number_to_words(num_str)  # Convert number to words

        # Currency mapping
        currency_word = ""
        if symbol in ['₹', 'Rs', 'Rs.']:
            currency_word = "ರೂಪಾಯಿ"
        elif symbol == '$':
            currency_word = "ಡಾಲರ್"
        elif symbol == '£':
            currency_word = "ಪೌಂಡ್"
        elif symbol == '€':
            currency_word = "ಯೂರೋ"

        return f"{word_num} {currency_word}".strip()
    
    return re.sub(money_pattern, repl, text)


# ----------------------------
# Dates normalization
# ----------------------------
def replace_dates(text):
    # dd/mm/yyyy or dd-mm-yyyy
    pattern = r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b"
    def repl(m):
        d, mth, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if len(str(y)) == 2:
            y = normalize_year(y)
        if not (1 <= d <= 31 and 1 <= mth <= 12):
            return m.group(0)
        return f"{convert_number_to_words(d)} {months_kannada[mth]} {convert_year_to_words(y)}"
    text = re.sub(pattern, repl, text)
    return text

# ----------------------------
# Hyphenated numbers
# ----------------------------
def replace_hyphen_numbers(text):
    def repl(m):
        parts = m.group(0).split("-")
        return " - ".join(" ".join(num2words(int(d), lang='kn') for d in part) for part in parts)
    return re.sub(r"\b(?:\d+-){2,}\d+\b", repl, text)

# ----------------------------
# Plus sign in numbers
# ----------------------------
def replace_plus_sign(text):
    return re.sub(r"(?<=\d)\+(?=\d)", "ಪ್ಲಸ್", text)
# ----------------------------
# Replace standalone numbers (including decimals)
# ----------------------------
def replace_standalone_numbers(text):
    # Matches integers or decimals not already processed
    pattern = r"\b\d+\.\d+|\b\d+\b"
    def repl(m):
        return convert_number_to_words(m.group(0))
    return re.sub(pattern, repl, text)


# ----------------------------
# Large numbers / phone numbers
# ----------------------------
def replace_large_numbers_and_phones(text):
    phone_keywords = ["ಫೋನ್", "ಮೊಬೈಲ್", "ನಂಬರ್", "ಕಾಲ್", "telephone", "contact"]
    tokens = text.split()
    new_tokens = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        j = i
        num_parts = []
        has_plus = tok.startswith("+")
        while j < len(tokens) and re.match(r"^\+?\d+$", tokens[j]):
            num_parts.append(tokens[j].lstrip("+"))
            j += 1
        if num_parts:
            merged = "".join(num_parts)
            prev_word = tokens[i-1] if i > 0 else ""
            next_word = tokens[j] if j < len(tokens) else ""
            is_phone = len(merged) >= 10 or prev_word in phone_keywords or next_word in phone_keywords
            if is_phone:
                phone_str = []
                if has_plus:
                    phone_str.append("ಪ್ಲಸ್")
                for part in num_parts:
                    phone_str.extend(num2words(int(d), lang='kn') for d in part)
                new_tokens.append(" ".join(phone_str))
            else:
                new_tokens.append(convert_number_to_words(merged))
            i = j
            continue
        new_tokens.append(tok)
        i += 1
    return " ".join(new_tokens)

# ----------------------------
# Main normalization pipeline
# ----------------------------
def normalize_sent_kan(text):
    text = replace_plus_sign(text)
    text = replace_hyphen_numbers(text)
    text = replace_dates(text)
    text = replace_time(text)
    text = replace_percentage(text)
    text = replace_measurements(text)
    text = replace_fractions_or_dates(text)
    text = replace_money(text)
    text = replace_large_numbers_and_phones(text)
    # NEW STEP: convert any remaining standalone numbers (including decimals)
    text = replace_standalone_numbers(text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
