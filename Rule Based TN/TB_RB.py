import re
from indic_numtowords import num2words

# Hindi month names
months_hindi = {
    1: "जनवरी", 2: "फ़रवरी", 3: "मार्च", 4: "अप्रैल",
    5: "मई", 6: "जून", 7: "जुलाई", 8: "अगस्त",
    9: "सितंबर", 10: "अक्टूबर", 11: "नवंबर", 12: "दिसंबर"
}

# Month name mapping for text dates
month_name_map = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12
}

def convert_number_to_words(num_str):
    """Converts integer or decimal to Hindi words."""
    num_str = str(num_str)
    if '.' in num_str:
        integer_part, decimal_part = num_str.split('.')
        integer_words = num2words(int(integer_part), lang='hi') if integer_part else ""
        decimal_words = " ".join(num2words(int(d), lang='hi') for d in decimal_part)
        result = f"{integer_words} दशमलव {decimal_words}".strip()
    else:
        # Ensure the integer is converted back to a string before passing to num2words
        result = num2words(str(int(num_str)), lang='hi')
    return f" {result} "

def number_to_hindi_phone(num_str):
    """Pronounce '+' as 'प्लस' and each digit separately in Hindi."""
    output_parts = []
    if num_str.startswith("+"):
        output_parts.append("प्लस")
        num_str = num_str[1:]
    output_parts.extend(num2words(int(d), lang='hi') for d in num_str)
    return " ".join(output_parts)

def replace_measurements(text):
    measure_units = {
        "फीट": "फीट",
        "मीटर": "मीटर",
        "m\\^2": "वर्ग मीटर",
        "km\\^2": "वर्ग किलोमीटर",
        "m\\^3": "घन मीटर",
        "km\\^3": "घन किलोमीटर",
        "km/hr": "किलोमीटर प्रति घंटा",
        "m/s": "मीटर प्रति सेकंड",
        "km/s": "किलोमीटर प्रति सेकंड",
        "m/min": "मीटर प्रति मिनट",
        "km/min": "किलोमीटर प्रति मिनट",
        "m": "मीटर",
        "km": "किलोमीटर",
    }

    for unit, hi_unit in measure_units.items():
        pattern = rf"\b([0-9०-९]+(?:\.[0-9०-९]+)?)\s*{unit}\b"

        def repl(m):
            num = m.group(1)
            # 🚨 Add markers so this chunk won’t be reprocessed
            return f"{convert_number_to_words(num).strip()} {hi_unit}"

        text = re.sub(pattern, repl, text)
    return text


def replace_percentage(text):
    percent_pattern = r"(\d+(?:\.\d+)?)\s*%"
    return re.sub(percent_pattern, lambda m: f"{convert_number_to_words(m.group(1))} प्रतिशत ", text)
def replace_time(text):
    # Case 1: hh:mm or hh.mm with mandatory AM/PM
    time_pattern = r"\b(\d{1,2})[:\.](\d{2})(?:[:\.](\d+))?\s*(am|pm|AM|PM)\b"

    def time_repl(match):
        hour, minute, sec, meridian = int(match.group(1)), int(match.group(2)), match.group(3), match.group(4)

        # Convert hour to 12-hour format
        hour_12 = hour if 1 <= hour <= 12 else hour - 12
        if hour_12 == 0:
            hour_12 = 12

        # Build words
        time_words = f"{convert_number_to_words(hour_12)}बजकर {convert_number_to_words(minute)}मिनट"

        if sec:
            seconds = int(sec)
            time_words += f" {convert_number_to_words(seconds)}सेकंड"

        if meridian:
            period = "सुबह" if meridian.lower() == "am" else "शाम"
            time_words = f"{period} {time_words}"

        return f" {time_words} "

    text = re.sub(time_pattern, time_repl, text)

    # Case 2: only hh AM/PM (like "5am")
    hour_only_pattern = r"\b(\d{1,2})\s*(am|pm|AM|PM)\b"

    def hour_only_repl(match):
        hour, meridian = int(match.group(1)), match.group(2)
        hour_12 = hour if 1 <= hour <= 12 else hour - 12
        if hour_12 == 0:
            hour_12 = 12
        period = "सुबह" if meridian.lower() == "am" else "शाम"
        return f" {period} {convert_number_to_words(hour_12)}बजे "

    text = re.sub(hour_only_pattern, hour_only_repl, text)

    # Case 3: hh:mm or hh.mm without AM/PM
    simple_time_pattern = r"\b(\d{1,2})[:\.](\d{2})(?:[:\.](\d+))?\b"

    def simple_time_repl(match):
        hour, minute, sec = int(match.group(1)), int(match.group(2)), match.group(3)

        time_words = f"{convert_number_to_words(hour)} {convert_number_to_words(minute)}"
        if sec:
            seconds = int(sec)
            time_words += f" {convert_number_to_words(seconds)}"
        return f" {time_words} "

    text = re.sub(simple_time_pattern, simple_time_repl, text)

    return text





def normalize_year(year: int) -> int:
    """Expand 2-digit or 3-digit year properly into full year."""
    if year < 100:  # yy → 19yy or 20yy
        return 2000 + year if year <= 30 else 1900 + year
    if year < 1000:  # 025 → 2025
        return 2000 + (year % 100)
    return year

def convert_year_to_words(year: int) -> str:
    year = normalize_year(year)

    if 1000 <= year < 2000:
        last_two = year % 100
        if year >= 1900:
            prefix = "उन्नीस सौ"
        elif year >= 1800:
            prefix = "अठारह सौ"
        elif year >= 1700:
            prefix = "सत्रह सौ"
        else:
            prefix = f"{num2words(year // 100, lang='hi')} सौ"
        return f"{prefix} {convert_number_to_words(last_two).strip()}" if last_two else prefix

    elif 2000 <= year < 2100:
        remainder = year % 1000
        return "दो हज़ार" if remainder == 0 else f"दो हज़ार {convert_number_to_words(remainder).strip()}"

    elif 2100 <= year < 4000:
        prefix = f"{num2words(year // 100, lang='hi')} सौ"
        last_two = year % 100
        return f"{prefix} {convert_number_to_words(last_two).strip()}" if last_two else prefix

    else:  # 4000+
        thousands = year // 1000
        remainder = year % 1000
        prefix = f"{convert_number_to_words(thousands).strip()} हज़ार"
        return f"{prefix} {convert_number_to_words(remainder).strip()}" if remainder else prefix

months_hindi = {
    1:"जनवरी",2:"फ़रवरी",3:"मार्च",4:"अप्रैल",5:"मई",6:"जून",7:"जुलाई",
    8:"अगस्त",9:"सितंबर",10:"अक्टूबर",11:"नवंबर",12:"दिसंबर"
}

# English to month number mapping
month_name_map = {
    "january":1,"february":2,"march":3,"april":4,"may":5,"june":6,"july":7,
    "august":8,"september":9,"october":10,"november":11,"december":12
}

def replace_dates(text):
    # 1. Numeric formats dd/mm/yyyy or dd-mm-yyyy
    date_pattern = r"\b(\d{1,4})[/-](\d{1,2})[/-](\d{2,4})\b"
    def date_repl(match):
        p1, p2, p3 = match.groups()
        # Determine day, month, year based on length
        if len(p1) == 4:  # yyyy-mm-dd
            year, month, day = int(p1), int(p2), int(p3)
        else:  # dd-mm-yyyy
            day, month, year = int(p1), int(p2), int(p3) if len(p3) == 4 else (1900 + int(p3))
        # Check valid day/month ranges
        if not (1 <= day <= 31 and 1 <= month <= 12):
            return match.group(0)  # invalid, return original
        return f"{convert_number_to_words(day)} {months_hindi[month]} {convert_year_to_words(year)}"
    text = re.sub(date_pattern, date_repl, text)

    # 2. Year first: yyyy m d
    ymd_pattern = r"\b(\d{4})\s+(\d{1,2})\s+(\d{1,2})\b"
    def ymd_repl(m):
        year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if not (1 <= day <= 31 and 1 <= month <= 12):
            return m.group(0)
        return f"{convert_number_to_words(day)} {months_hindi[month]} {convert_year_to_words(year)}"
    text = re.sub(ymd_pattern, ymd_repl, text)

    # 3. Text month formats: 5 May 2025 OR 5 अगस्त 1989
    text_month_pattern = r"(\d{1,2})\s+([A-Za-z\u0900-\u097F]+)\s+(\d{2,4})"
    def text_month_repl(m):
        day = int(m.group(1))
        month_raw = m.group(2).lower()
        year = int(m.group(3))
        # Validate day
        if not (1 <= day <= 31):
            return m.group(0)
        # Determine month number
        if month_raw in month_name_map:
            month = months_hindi[month_name_map[month_raw]]
        elif month_raw in months_hindi.values():
            month = month_raw
        else:
            return m.group(0)
        return f"{convert_number_to_words(day)} {month} {convert_year_to_words(year)}"
    text = re.sub(text_month_pattern, text_month_repl, text)

    # 4. Month first formats: May 5, 1989 OR अगस्त 5, 1989
    month_text_pattern = r"([A-Za-z\u0900-\u097F]+)\s+(\d{1,2}),?\s*(\d{2,4})"
    def month_text_repl(m):
        month_raw = m.group(1).lower()
        day = int(m.group(2))
        year = int(m.group(3))
        # Validate day
        if not (1 <= day <= 31):
            return m.group(0)
        if month_raw in month_name_map:
            month = months_hindi[month_name_map[month_raw]]
        elif month_raw in months_hindi.values():
            month = month_raw
        else:
            return m.group(0)
        return f"{convert_number_to_words(day)} {month} {convert_year_to_words(year)}"
    text = re.sub(month_text_pattern, month_text_repl, text)

    return text
def replace_hyphen_numbers(text):
    """
    Treat only sequences with 3 or more hyphens as digit-wise IDs.
    Example:
      12-01-12-2-3-215689-12 →
      बारह - शून्य एक - बारह - दो - तीन - दो एक पाँच छह आठ नौ - बारह
    But:
      1-12-2010 → stays untouched (so date logic can process it).
    """
    def repl(m):
        parts = m.group(0).split("-")
        out_parts = []
        for part in parts:
            # Speak each digit separately
            spoken = " ".join(num2words(int(d), lang='hi') for d in part)
            out_parts.append(spoken)
        return " - ".join(out_parts)

    # Require at least 3 hyphens (so normal dates like 1-12-2010 are not caught)
    return re.sub(r"\b(?:\d+-){3,}\d+\b", repl, text)

import random

def replace_fractions_or_dates(text):
    motion_verbs = ["गया", "आई", "रहा", "बैठा", "चल पड़ा", "निकला", "आया"]
    tokens = text.split()
    new_tokens = []

    for i, tok in enumerate(tokens):
        if re.match(r"^\d+/\d+$", tok):
            a, b = map(int, tok.split('/'))
            prev_word = tokens[i-1] if i > 0 else ""
            next_word = tokens[i+1] if i < len(tokens)-1 else ""

            # --- If looks like a date ---
            if next_word in ["को", "में", "पर"] or prev_word in motion_verbs:
                month_num = b if 1 <= b <= 12 else b
                replacement = f"{convert_number_to_words(a)} {months_hindi.get(month_num, '')}"
            else:
                # --- Randomly choose 'भाग' or 'बटे' ---
                connector = random.choice(["भाग", "बटे"])
                replacement = f"{convert_number_to_words(a)} {connector} {convert_number_to_words(b)}"

            new_tokens.append(replacement)
        else:
            new_tokens.append(tok)
    return " ".join(new_tokens)


def replace_money(text):
    money_pattern = r"(\₹|Rs\.?|Rs|\$|£|€)?\s*(\d+(?:,\d{2,3})*(?:\.\d+)?)(?:\s*(million|billion|thousand|lakh|crore))?"
    def money_repl(match):
        symbol, num_str, multiplier = match.groups()
        num_str = num_str.replace(",", "")
        word_num = convert_number_to_words(num_str).strip()
        currency_word = ""
        if symbol in ['₹', 'Rs', 'Rs.']:
            currency_word = "रुपये"
        elif symbol == '$':
            currency_word = "डॉलर"
        elif symbol == '£':
            currency_word = "पाउंड"
        elif symbol == '€':
            currency_word = "यूरो"
        multiplier_word = ""
        if multiplier:
            multiplier_lower = multiplier.lower()
            if multiplier_lower == "million":
                multiplier_word = " मिलियन"
            elif multiplier_lower == "billion":
                multiplier_word = " बिलियन"
            elif multiplier_lower == "thousand":
                multiplier_word = " हज़ार"
            elif multiplier_lower == "lakh":
                multiplier_word = " लाख"
            elif multiplier_lower == "crore":
                multiplier_word = " करोड़"
        result = f"{word_num}{multiplier_word} {currency_word}".strip()
        return f" {result} "
    return re.sub(money_pattern, money_repl, text)

def fix_money_order(text):
    pattern = r"(\b[\w\d\d०-९]+)\s+(डॉलर|रुपये|पाउंड|यूरो)\s+(मिलियन|बिलियन|हज़ार|लाख|करोड़)\b"
    def repl(match):
        number, currency, multiplier = match.groups()
        return f"{number} {multiplier} {currency}"
    return re.sub(pattern, repl, text)

def replace_large_numbers_and_phones(text):
    phone_keywords = ["फोन", "मोबाइल", "नंबर", "कॉल", "telephone", "contact"]

    tokens = text.split()
    new_tokens = []
    i = 0

    while i < len(tokens):
        tok = tokens[i]

        # Gather consecutive numeric tokens (with optional + at start)
        j = i
        num_parts = []
        has_plus = tokens[i].startswith("+")
        while j < len(tokens) and re.match(r"^\+?\d+$", tokens[j]):
            num_parts.append(tokens[j].lstrip("+"))
            j += 1

        if num_parts:
            merged = "".join(num_parts)
            prev_word = tokens[i - 1] if i > 0 else ""
            next_word = tokens[j] if j < len(tokens) else ""

            # --- PHONE detection ONLY if ---
            #   1) Length >= 8 digits, OR
            #   2) Surrounded by phone-related keywords
            is_phone = (
                len(merged) >= 8 or
                prev_word in phone_keywords or
                next_word in phone_keywords
            )

            if is_phone:
                phone_str = []
                if has_plus:
                    phone_str.append("प्लस")
                for part in num_parts:
                    phone_str.extend(num2words(int(d), lang='hi') for d in part)
                new_tokens.append(" ".join(phone_str))
            else:
                # Always treat shorter numbers as normal numbers
                new_tokens.append(convert_number_to_words(merged))

            i = j
            continue

        new_tokens.append(tok)
        i += 1

    return " ".join(new_tokens)


def replace_plus_sign(text):
    # + को "प्लस" से बदलें केवल तब जब ये नंबर या telephone prefix में हो
    text = re.sub(r"\+", "प्लस ", text)
    return text



def normalize_sent(text, category=None):
    try:
        text = replace_plus_sign(text)
        text = replace_time(text)
        text = replace_hyphen_numbers(text)

        tokens = text.split()

        #  Case: pure number input
        if len(tokens) == 1 and re.match(r"^\+?\d+$", tokens[0]):
            return convert_number_to_words(tokens[0]).strip()

        #  Case: sequence of numbers -> check for phone
        if all(re.match(r"^\+?\d+$", t) for t in tokens):
            phone_str = []
            has_plus = tokens[0].startswith("+")
            if has_plus:
                phone_str.append("प्लस")
                tokens[0] = tokens[0][1:]
            for tok in tokens:
                phone_str.extend([num2words(int(d), lang='hi') for d in tok])
            return " ".join(phone_str)

        # Normal processing
        text = replace_dates(text)
        text = replace_time(text)
        text = replace_percentage(text)
        text = replace_measurements(text)
        text = replace_fractions_or_dates(text)
        text = replace_money(text)
        text = replace_large_numbers_and_phones(text)
        text = fix_money_order(text)

        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception as e:
        # If any error occurs, return the original sentence
        print(f"Error processing sentence: {text} | Error: {e}")
        return text