
from difflib import SequenceMatcher

comparison_tags = ["JJR", "RBR", "JJS", "RBS", "RB"]


# E.g. Which group from the census is larger: Irish or danish?
def comparison_parse_structure_1(question_parsed):
    res = None
    comp_idx = -1
    sep_idx = -1
    cc_idx = -1

    for token in question_parsed:
        # first identify a comparative word
        if comp_idx == -1 and token.tag_ in comparison_tags:
            comp_idx = token.i

        # then, identify a separator token (a comma or a colon)
        elif (comp_idx > -1 and sep_idx == -1
              and token.text in [",", ":", ";"]):
            sep_idx = token.i

        # then, identify an "or" conjunction
        elif (comp_idx > -1 and sep_idx > -1 and cc_idx == -1
              and token.text == "or"
                # and token.dep_ == "cc"
        ):
            cc_idx = token.i
            res = {
                "comp": comp_idx,
                "sep": sep_idx,
                "cc": cc_idx,
            }
            break

    return res


# E.g. Were there more Latinos or English?
# and Were there more robberies in Harlem in 1981 or 1990?
def comparison_parse_structure_2(question_parsed):
    res = None
    comp_idx = -1
    sep_idx = -1
    cc_idx = -1

    for token in question_parsed:
        # first identify a comparative word
        if comp_idx == -1 and token.tag_ in comparison_tags:
            comp_idx = token.i

        # then, identify an "or" conjunction that is headed by the same token of the following token.
        elif (comp_idx > -1 and cc_idx == -1 and token.text == "or" and token.i < len(question_parsed)-1 and
              token.head.i == question_parsed[token.i+1].head.i == token.i-1
        ):
            cc_idx = token.i
            res = {
                "comp": comp_idx,
                "sep": sep_idx,
                "cc": cc_idx,
            }
            break

    return res


def comparison_parse_structure(question_parsed):
    res = comparison_parse_structure_1(question_parsed)
    if res is None:
        res = comparison_parse_structure_2(question_parsed)

    return res


def get_comparison_answer_candidates_from_indices(question_parsed, indices):
    if indices["sep"] > -1:
        first_candidate = question_parsed[indices["sep"] + 1: indices["cc"]].text
        second_candidate = question_parsed[indices["cc"] + 1:-1].text
    else:
        first_candidate = question_parsed[indices["cc"]-1].text
        second_candidate = question_parsed[indices["cc"]+1].text

    return first_candidate, second_candidate


def comparison_get_opposite_answer(question_parsed, orig_answer_span, indices):
    first_candidate, second_candidate = get_comparison_answer_candidates_from_indices(question_parsed, indices)
    first_candidate_ratio = SequenceMatcher(None, first_candidate, orig_answer_span).ratio()
    second_candidate_ratio = SequenceMatcher(None, second_candidate, orig_answer_span).ratio()

    if first_candidate_ratio >= second_candidate_ratio:
        return second_candidate
    else:
        return first_candidate


def comparison_get_answer_from_step_index(question_parsed, answer_step_index, indices):
    first_candidate, second_candidate = get_comparison_answer_candidates_from_indices(question_parsed, indices)

    if answer_step_index == 0:
        return first_candidate
    elif answer_step_index == 1:
        return second_candidate
    else:
        raise RuntimeError

