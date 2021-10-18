
import dateutil.parser as date_parser
import datetime
import spacy

from example_generation.answer_generators.auxilary_methods import \
    comparison_parse_structure, comparison_get_answer_from_step_index
from qdmr_transforms.qdmr_identifier import StepIdentifier
from src.reference_utils import MAX_STEPS

nlp = spacy.load('en')
DEFAULT_DATE = date_parser.parse("0001-01-01")


def clean_string_marks(s):
    # for word in ["hundred", "thousand", "million", "billion", "trillion"]:
    #     s = s.replace(word, '')
    for punc in [',', '%', '£', '$']:
        s = s.replace(punc, '')

    return s.strip()


def clean_candidate_arg(arg):
    arg_clean = None
    arg_type = "none"
    if type(arg) == str:
        # float / int
        if arg_clean is None:
            try:
                arg_clean = float(clean_string_marks(arg))
                if arg_clean.is_integer():
                    arg_clean = int(arg_clean)
                    arg_type = "int"
                else:
                    arg_type = "float"
            except:
                arg_clean = None

        # date
        if arg_clean is None:
            try:
                arg_clean = date_parser.parse(arg, default=DEFAULT_DATE, fuzzy_with_tokens=True)
                arg_clean = arg_clean[0]    # when fuzzy_with_tokens is set to True it returns a tuple.
                arg_type = "date"
            except:
                arg_clean = None

        # float / int - heuristic
        if arg_clean is None:
            try:
                arg_clean = float(clean_string_marks(arg.split(" ")[0]))
                if arg_clean.is_integer():
                    arg_clean = int(arg_clean)
                    arg_type = "int"
                else:
                    arg_clean = round(arg_clean, 2)
                    arg_type = "float"

            except:
                arg_clean = None

    elif type(arg) == int:
        return arg, "int"

    elif type(arg) == float:
        return arg, "float"

    elif type(arg) == datetime.datetime:
        return arg, "date"

    if arg_clean is not None:
        return arg_clean, arg_type
    return arg, arg_type


def eval_select_step(qdmr, step_i, step, step_refs, evaluated_step_answers):
    assert len(step_refs) == 0
    evaluated_step_answers[step_i] = qdmr["step_answers"][step_i]
    return 0


def eval_simple_aggregation_step(qdmr, step_i, step, step_refs, evaluated_step_answers):
    assert len(step_refs) == 1
    try:
        ref = step_refs[0]
        ans_count, ans_type = clean_candidate_arg(qdmr["step_answers"][ref])
        first_arg_lower = step.arguments[0].lower()

        if ans_type == "int" and (first_arg_lower == 'count' or
                                  'number of #ref' in first_arg_lower or
                                  'amount of #ref' in first_arg_lower):
            evaluated_step_answers[step_i] = ans_count

        elif ans_type == "float" and first_arg_lower == 'percentage of #ref':
            evaluated_step_answers[step_i] = ans_count

    except:
        return -1

    return 0


def eval_arithmetic_step(qdmr, step_i, step, step_refs, evaluated_step_answers):
    arg_ans_candidates = [clean_candidate_arg(evaluated_step_answers[ref]) for ref in step_refs]
    arg_ans = [arg[0] for arg in arg_ans_candidates]

    # check that all arguments are numeric and that they have different values (if the arguments
    # are the same then it is probably because of an error of the single-step QA model).
    if len([arg for arg in arg_ans_candidates if arg[1] in ["int", "float"]]) == len(arg_ans_candidates) and \
            arg_ans.count(arg_ans[0]) != len(arg_ans):
        try:
            if step.arguments[0] == 'difference':
                evaluated_step_answers[step_i] = abs(arg_ans[0] - sum(arg_ans[1:]))

            elif step.arguments[0] == 'sum':
                evaluated_step_answers[step_i] = sum(arg_ans)

            elif step.arguments[0] == 'multiplication':
                res = 1
                for ans in arg_ans:
                    res *= ans
                evaluated_step_answers[step_i] = res
        except:
            return -1

    return 0


def comparable_ans_candidates(arg_ans_candidates):
    res = True

    # check that all candidates have an identified answer type
    if len([arg for arg in arg_ans_candidates if arg[1] != "none"]) != len(arg_ans_candidates):
        res = False

    # if the candidates are dates and one of which includes a year, then check that all candidates include a year.
    if len([arg for arg in arg_ans_candidates if arg[1] == "date"]) == len(arg_ans_candidates):
        includes_year = arg_ans_candidates[0][0].year > 1
        if includes_year and \
                len([arg for arg in arg_ans_candidates if arg[0].year > 1]) != len(arg_ans_candidates):
            res = False

    return res


def eval_comparison_step(qdmr, step_i, step, step_refs, evaluated_step_answers):
    arg_ans_raw = [evaluated_step_answers[ref] for ref in step_refs]
    arg_ans_candidates = [clean_candidate_arg(ans_raw) for ans_raw in arg_ans_raw]
    arg_ans = [arg[0] for arg in arg_ans_candidates]

    # check that all arguments are comparable (date vs. date, number vs. number, etc.) and that
    # we are not comparing the same values (which is probably because of an error of the single-step QA model).
    if comparable_ans_candidates(arg_ans_candidates) and \
            arg_ans.count(arg_ans[0]) != len(arg_ans):
        try:
            if step.arguments[0] == 'min':
                if arg_ans[0] <= arg_ans[1]:
                    ans_idx = step_refs[0]
                else:
                    ans_idx = step_refs[1]
            elif step.arguments[0] == 'max':
                if arg_ans[0] <= arg_ans[1]:
                    ans_idx = step_refs[1]
                else:
                    ans_idx = step_refs[0]
            else:
                raise NotImplementedError

            # In comparison steps in Break, min/max actually mean argmin/argmax.
            question_parsed = nlp(qdmr["generated_question"])
            indices = comparison_parse_structure(question_parsed)
            answer = comparison_get_answer_from_step_index(question_parsed, ans_idx, indices)
            evaluated_step_answers[step_i] = answer

        except:
            return -1

    return 0


def eval_boolean_step(qdmr, step_i, step, step_refs, evaluated_step_answers):
    cond = step.arguments[1]
    if cond.rsplit(' ', 1)[0] in ["is higher than", "is lower than", "is the same as"]:
        assert len(step_refs) == 1
        arg_raw = evaluated_step_answers[step_refs[0]]
        arg_candidate = clean_candidate_arg(arg_raw)
        try:
            _, cond_op, _, value_str = cond.replace("the same", "the-same").split(' ')  # is higher than two
            if value_str[0] == "#":
                value_raw = evaluated_step_answers[int(value_str[1:]) - 1]
                value_candidate = clean_candidate_arg(value_raw)
            else:
                value_candidate = float(value_str)

            # Continue only if we are not comparing two reference value which are the same.
            # Such cases are probably because of incorrect predictions by the single-step qa model.
            if not (value_str[0] == "#" and arg_candidate == value_candidate):
                if cond_op == "higher":
                    if arg_candidate > value_candidate:
                        evaluated_step_answers[step_i] = "yes"
                    else:
                        evaluated_step_answers[step_i] = "no"
                elif cond_op == "lower":
                    if arg_candidate < value_candidate:
                        evaluated_step_answers[step_i] = "yes"
                    else:
                        evaluated_step_answers[step_i] = "no"
                elif cond_op == "the-same":
                    if arg_candidate == value_candidate:
                        evaluated_step_answers[step_i] = "yes"
                    else:
                        evaluated_step_answers[step_i] = "no"

                # Handle the special case (common in HotpotQA), where the arguments are
                # years but the comparison is actually of ages - simply swap the answer.
                step_ref_text = qdmr["transformed"][step_refs[0]].lower()
                if cond_op in ["higher", "lower"] and \
                        ("how young" in step_ref_text or "how old" in step_ref_text):
                    if evaluated_step_answers[step_i] == "yes":
                        evaluated_step_answers[step_i] = "no"
                    else:
                        assert evaluated_step_answers[step_i] == "no"
                        evaluated_step_answers[step_i] = "yes"

        except:
            return -1

    return 0


def eval_union_step(qdmr, step_i, step, step_refs, evaluated_step_answers):
    evaluated_step_answers[step_i] = [
        f"{evaluated_step_answers[ref]}" for ref in step_refs
    ]

    return 0


def evalute_qdmr(qdmr):
    # go over decomposition steps, for selection steps - use the answers generated by the single-step qa model,
    # for simple arithmetic/aggregation steps - calculate the answer directly, and ignore other steps.
    evaluated_step_answers = [None] * len(qdmr["transformed"])
    num_resolved_steps = len([step for step in evaluated_step_answers if step is not None])
    step_identifier = StepIdentifier()
    while True:
        for step_i, step_text in enumerate(qdmr["transformed"]):
            if evaluated_step_answers[step_i] is not None:
                continue
            step = step_identifier.identify(step_text)
            step_refs = [
                int(arg[1:].split(' ')[0])-1 for arg in step.arguments
                if len(arg) > 1 and arg[0] == '#' and arg[1].isdigit()
            ]
            # some decompositions might have sub-strings like #51 which are not real placeholders.
            step_refs = [ref for ref in step_refs if ref < MAX_STEPS]

            # selection step - use the answer generated by the single-step qa model.
            if step.operator == 'select':
                _ = eval_select_step(qdmr, step_i, step, step_refs, evaluated_step_answers)
                continue

            # if we don't have answers for all reference steps, skip this step.
            num_evaluated_refs = len([ref for ref in step_refs if evaluated_step_answers[ref] is not None])
            if num_evaluated_refs < len(step_refs):
                continue

            # we have all the references, now check that we can actually use them to compute the answer to this step.
            first_arg_lower = step.arguments[0].lower()
            if (step.operator == 'aggregate' and first_arg_lower == 'count') or \
                    (step.operator == 'project' and first_arg_lower == 'percentage of #ref') or \
                    (step.operator == 'project' and 'number of #ref' in first_arg_lower) or \
                    (step.operator == 'project' and 'amount of #ref' in first_arg_lower):
                res = eval_simple_aggregation_step(qdmr, step_i, step, step_refs, evaluated_step_answers)
                if res < 0:
                    continue

            elif step.operator == 'arithmetic':
                res = eval_arithmetic_step(qdmr, step_i, step, step_refs, evaluated_step_answers)
                if res < 0:
                    continue

            elif step.operator == 'comparison':
                res = eval_comparison_step(qdmr, step_i, step, step_refs, evaluated_step_answers)
                if res < 0:
                    continue

            elif step.operator == 'boolean':
                res = eval_boolean_step(qdmr, step_i, step, step_refs, evaluated_step_answers)
                if res < 0:
                    continue

            elif step.operator == 'union':
                res = eval_union_step(qdmr, step_i, step, step_refs, evaluated_step_answers)
                if res < 0:
                    continue

        # stop evaluation if there are no new evaluated answers, or if we generated the final answer
        final_answer = str(evaluated_step_answers[-1])
        if final_answer is not None and final_answer != "None":
            qdmr["transformed_answer"] = final_answer
            break

        new_num_resolved_steps = len([step for step in evaluated_step_answers if step is not None])
        if new_num_resolved_steps == num_resolved_steps:
            break
        num_resolved_steps = new_num_resolved_steps
