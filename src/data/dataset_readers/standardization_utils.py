import html
import re
import sys


whitespaces = re.findall(
    r"\s", u"".join(chr(c) for c in range(sys.maxunicode + 1)), re.UNICODE
)
empty_chars = ["\u200b", "\ufeff", "\u2061"]  # zero width space, byte order mark


def standardize_text_simple(text, output_offset=False):
    for whitespace in whitespaces:
        if whitespace == "\n" or whitespace == "\t":
            continue
        text = text.replace(whitespace, " ")

    for empty_char in empty_chars:
        text = text.replace(empty_char, " ")

    stripped_text = text.strip()
    offset = len(stripped_text) - len(text.rstrip())
    return (stripped_text, offset) if output_offset else stripped_text


def standardize_text_advanced(text, output_offset=False):
    text = html.unescape(text)
    text = standardize_text_simple(text)
    text = " ".join(
        text.split()
    )  # use ' ' for all spaces and replace sequence of spaces with single space

    # There is a pattern that repeats itself 97 times in the train set and 16 in the
    # dev set: "<letters>.:<digits>". It originates from the method of parsing the
    # Wikipedia pages. In such an occurrence, "<letters>." is the last word of a
    # sentence, followed by a period. Then, in the wikipedia page, follows a superscript
    # of digits within square brackets, which is a hyperlink to a reference. After the
    # hyperlink there is a colon, ":", followed by <digits>. These digits are the page
    # within the reference.
    # Example: https://en.wikipedia.org/wiki/Polish%E2%80%93Ottoman_War_(1672%E2%80%931676)
    if ".:" in text:
        text = re.sub("\.:\d+(-\d+)*", ".", text)

    # offset from here shouldn't be used!
    return (text, 10000000) if output_offset else text


def fix_references(string):
    return re.sub(r'#([1-9][0-9]?)', '@@\g<1>@@', string)


def fix_references_back(string):
    return re.sub(r'@@([1-9][0-9]?)@@', '#\g<1>', string)


