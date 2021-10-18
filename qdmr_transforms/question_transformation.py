import re
import spacy

nlp = spacy.load("en")

FLIP_COMPARISON_OP = {'young': 'old',
                      'first': 'second',
                      'short': 'long',
                      'lower': 'bigger',
                      'smaller': 'larger',
                      'small': 'big',
                      'low': 'high',
                      'less': 'more'}
FLIP_COMPARISON_OP.update({v: k for k, v in FLIP_COMPARISON_OP.items()})
FLIP_COMPARISON_OP['fewer'] = 'more'
FLIP_COMPARISON_OP['large'] = 'small'


def key_is_str_prefix(dictionary, string):
    keys = dictionary.keys()
    string = string.lower()
    for key in keys:
        if string.startswith(key.lower()):
            return key
    return None

def transform_comparison_question(question):
    """Flip the comparison operator in the question with a
    ne operator. E.g., young --> old, first --> second"""
    for token in question.split():
        op = key_is_str_prefix(FLIP_COMPARISON_OP, token)
        if op is not None:
            return question.replace(op, FLIP_COMPARISON_OP[op])
    return None

def transform_append_boolean_question(question, new_condition):
    """Transform a counting question to a boolean comparison question.
    E.g.,
        How many species are in the plant family that includes the vine Cucumis argenteus?
        --> If more than 2 are in the plant family that includes the vine Cucumis argenteus?
    """

    def lowercase_first_token(phrase):
        tokens = phrase.split()
        tokens[0] = tokens[0].lower()
        return " ".join(tokens)

    def switch_were_there(in_question):
        if "were there" in in_question:
            in_question = "there were " + in_question.replace(" were there", "")
        return in_question

    # so we don't get an uppercase word in the middle of the new question
    # question = lowercase_first_token(question)
    trigger = "how many"
    if trigger not in question.lower():
        return None
    parsed = nlp(question)
    how_many_start_index = [
        t.i for i, t in enumerate(parsed)
        if (i < len(parsed) - 1 and
            parsed[i:i + 2].text.lower() == "how many")
    ]
    if len(how_many_start_index) == 1:
        # check if the question has a do/did/does structure.
        how_many_start_index = how_many_start_index[0]
        do_index = [
            parsed[i].i for i in range(how_many_start_index + 2, len(parsed))
            if parsed[i].lemma_ == "do"
        ]

        # simple structure manipulation where we remove "how many" and "did/does/etc.",
        # and append the condition at the end.
        if len(do_index) == 1:
            do_index = do_index[0]
            # Tomer: handle crash when how_many_start_index==0
            text_before_how_many = parsed[:how_many_start_index].text if how_many_start_index != 0 else ""
            subject_text = parsed[how_many_start_index + 2: do_index].text
            if question.endswith("?"):
                text_after_do = parsed[do_index + 1:-1].text
            else:
                text_after_do = parsed[do_index + 1:].text

            if text_before_how_many == "":
                parts = ["If", text_after_do, new_condition, subject_text + "?"]
            else:
                parts = ["If", text_before_how_many, subject_text, text_after_do, new_condition + "?"]
            return " ".join([part for part in parts if part != ""])

        # simple heuristic, append "If" & replace "how many" with condition
        else:
            cased_trigger = original_substring_case(question, trigger)
            new_question = question.replace(cased_trigger, new_condition)
            new_question = switch_were_there(new_question)
            return "If " + new_question

    return None

def original_substring_case(full_string, substring):
    if re.search(substring, full_string, re.IGNORECASE):
        return re.search(substring, full_string, re.IGNORECASE).group(0)
    return None
 
def transform_replace_boolean_question(question):
    """Transform boolean question with two true conditions to double negation
    E.g.,
        Are both genera Silphium and Heliotropium, genera of flowering plants ?
        --> Are neither genera Silphium nor Heliotropium, genera of flowering plants ?

        Did John Updike and Tom Clancy both publish more than 15 bestselling novels?
        --> Did neither John Updike nor Tom Clancy publish more than 15 bestselling novels
    """
    def double_negation(phrase):
        phrase = phrase.replace(" and ", " nor ").replace("And ", "Nor ").replace(" both ", " ")
        tokens = phrase.split()
        tokens[0] = "Do" if tokens[0].lower() == "can" else tokens[0]
        tokens[0] = tokens[0] + " neither"
        return " ".join(tokens)

    if " and " not in question and "And " not in question:
        return None
    return double_negation(question)
