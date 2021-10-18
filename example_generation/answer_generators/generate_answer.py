
import spacy

from example_generation.answer_generators.auxilary_methods import \
    comparison_parse_structure, comparison_get_opposite_answer
from example_generation.answer_generators.qdmr_evaluator import evalute_qdmr, clean_candidate_arg
from qdmr_transforms.qdmr_example import get_transform_base_info, get_transform_base_info_parts
from src.data.dataset_readers.drop import DropReader
from src.reference_utils import get_references

nlp = spacy.load('en')
# nlp = spacy.load('en_core_web_lg')
reader = DropReader()


def gen_ans_op_replace_aggregate(qdmr):
    pass


def gen_ans_op_replace_arithmetic(qdmr):
    # assert len(qdmr["orig_answer_texts"]) == 1
    _, transform_info = get_transform_base_info(qdmr["transformation"])
    step, orig_op, transform_op = get_transform_base_info_parts(transform_info)
    transform_op = transform_op.strip("_VARIANT")
    if int(step) == len(qdmr["transformed"]) - 1:
        try:
            orig_answer_numeric = float(qdmr["orig_answer_texts"][0])
        except:
            return

        # discard examples where the original answer is an integer <10, because it might be
        # a question that requires counting, and therefore, extracting the arguments from the passage will
        # probably lead to an incorrect answer.
        if orig_answer_numeric.is_integer() and orig_answer_numeric < 10:
            return

        all_number_pairs = []
        numbers = qdmr["orig_numbers"] + [100.0]
        for i in range(len(numbers)):
            for j in range(i + 1, len(numbers)):
                if numbers[i] <= numbers[j]:
                    num_a = numbers[i]
                    num_b = numbers[j]
                else:
                    num_a = numbers[j]
                    num_b = numbers[i]

                if orig_op == "sum" and num_a + num_b == orig_answer_numeric:
                    step_refs = get_references(qdmr["transformed"][int(step)])[::-1]
                    candidate_inter_ans = [clean_candidate_arg(qdmr["step_answers"][ref])[0] for ref in step_refs]
                    # we would like to avoid questions with a negative answer, the make sure the question expresses
                    # the right order of compared entities (e.g. when asking "are there more X than Y?",
                    # we need to make sure that there is more X than Y, based on the context). To this end, we
                    # add a pair of numbers only if the numbers match the QA model's intermediate predictions
                    # in the expected order: We want the answer to the first step to be greater than
                    # the answer to the second step.
                    if (len(step_refs) == 2 and
                            num_b == candidate_inter_ans[0] and num_a == candidate_inter_ans[1]):
                        all_number_pairs.append((num_a, num_b))
                elif orig_op == "difference" and abs(num_a - num_b) == orig_answer_numeric:
                    all_number_pairs.append((num_a, num_b))
                elif orig_op == "multiplication" and num_a * num_b == orig_answer_numeric:
                    raise NotImplementedError
                elif orig_op == "division" and num_a != 0 and num_b != 0 and \
                        orig_answer_numeric in [num_a * 1.0 / num_b, num_b * 1.0 / num_a]:
                    raise NotImplementedError

        all_number_pairs = list(set(all_number_pairs))
        if len(all_number_pairs) == 1:
            (num_a, num_b) = all_number_pairs[0]
            # TODO(mega): should check if the changes for division make sense.
            if transform_op == "sum":
                qdmr["transformed_answer"] = num_a + num_b
            elif transform_op == "difference":
                qdmr["transformed_answer"] = abs(num_a - num_b)
            elif transform_op == "multiplication":
                qdmr["transformed_answer"] = num_a * num_b
            elif transform_op == "division":
                if num_a > 0:
                    qdmr["transformed_answer"] = num_b * 1.0 / num_a
            else:
                raise NotImplementedError


def gen_ans_op_replace_comparison(qdmr, second_attempt=False):
    _, transform_info = get_transform_base_info(qdmr["transformation"])
    step, orig_op, transform_op = get_transform_base_info_parts(transform_info)
    if int(step) == len(qdmr["transformed"]) - 1:
        if second_attempt is True:
            question_parsed = nlp(qdmr["generated_question"])
        else:
            question_parsed = nlp(qdmr["question"])
        indices = comparison_parse_structure(question_parsed)

        orig_answer_span = qdmr["orig_answer_texts"][0]
        if indices:
            opposite_answer = comparison_get_opposite_answer(
                question_parsed, orig_answer_span, indices
            )
            if opposite_answer is not None and opposite_answer != "":
                qdmr["transformed_answer"] = opposite_answer

        # If we failed to generate the answer from the original question,
        # try again with the generated question.
        if "transformed_answer" not in qdmr and second_attempt is False:
            gen_ans_op_replace_comparison(qdmr, second_attempt=True)


def gen_ans_op_replace_comparative(qdmr):
    pass


def gen_ans_op_replace_superlative(qdmr):
    gen_ans_op_replace_comparison(qdmr)


def gen_ans_op_replace_boolean(qdmr):
    assert len(qdmr["orig_answer_texts"]) == 1
    _, transform_info = get_transform_base_info(qdmr["transformation"])
    step, orig_op, transform_op = get_transform_base_info_parts(transform_info)

    # swap_answer = {
    #     "logical_and-logical_or": False,
    #     "logical_or-logical_and": None,
    #     "logical_and-true": True,
    #     "logical_and-false": True,
    #     "logical_or-true-no": True,
    #     "logical_or-false-no": True,
    # }

    if int(step) == len(qdmr["transformed"]) - 1:
        orig_ans = qdmr["orig_answer_texts"][0]
        if orig_ans == "yes":
            orig_ans_opposite = "no"
        elif orig_ans == "no":
            orig_ans_opposite = "yes"
        else:
            return

        if orig_op == "logical_and" and transform_op == "logical_or":
            qdmr["transformed_answer"] = orig_ans
        elif orig_op == "logical_and" and transform_op in ["true", "false"] and orig_ans.lower() == "yes":
            qdmr["transformed_answer"] = orig_ans_opposite
        elif orig_op == "logical_or" and transform_op in ["true", "false"] and orig_ans.lower() == "no":
            qdmr["transformed_answer"] = orig_ans_opposite


def is_two_step_number_of_structure(qdmr):
    if qdmr["transformed"][-1].lower() in \
            ["the number of #1", "number of #1", "numberof #1",
             "the amount of #1", "amount of #1"]:
        return True

    return False


def is_first_step_answer_valid(qdmr, numeric=False):
    if "step_answers" in qdmr and qdmr["step_answers"][0] not in [None, ""]:
        if numeric:
            try:
                _ = float(qdmr["step_answers"][0])
                return True
            except:
                pass
        else:
            return True

    return False


def gen_ans_prune_last_step(qdmr):
    if is_two_step_number_of_structure(qdmr) and is_first_step_answer_valid(qdmr, numeric=True):
        qdmr["transformed_answer"] = qdmr["step_answers"][0]

    elif "difference of 100 and #1" in qdmr["decomposition"][-1].lower():
        try:
            orig_answer_numeric = float(qdmr["orig_answer_texts"][0])
            qdmr["transformed_answer"] = 100 - orig_answer_numeric
        except:
            pass

    elif len(qdmr["transformed"]) == 1 and is_first_step_answer_valid(qdmr):
        qdmr["transformed_answer"] = qdmr["step_answers"][0]


def gen_ans_prune_last_step_rm_unused(qdmr):
    if is_two_step_number_of_structure(qdmr) and is_first_step_answer_valid(qdmr, numeric=True):
        qdmr["transformed_answer"] = qdmr["step_answers"][0]

    elif len(qdmr["transformed"]) == 1 and is_first_step_answer_valid(qdmr):
        qdmr["transformed_answer"] = qdmr["step_answers"][0]


def gen_ans_prune_step(qdmr):
    gen_ans_prune_last_step_rm_unused(qdmr)


def gen_ans_change_last_step(qdmr):
    pass


def gen_ans_append_boolean_step(qdmr):
    _, transform_info = get_transform_base_info(qdmr["transformation"])
    _, _, cond_op, value = get_transform_base_info_parts(transform_info)

    # extract the original numeric answer and the transformation numeric answer.
    orig_ans = qdmr["orig_answer_texts"][0]
    orig_ans_value = reader.extract_number_from_text(orig_ans)
    value_num = reader.extract_number_from_text(value)

    if orig_ans_value is not None and value_num is not None:
        if cond_op == "lower":
            if orig_ans_value < value_num:
                qdmr["transformed_answer"] = "yes"
            else:
                qdmr["transformed_answer"] = "no"

        elif cond_op == "higher":
            if orig_ans_value > value_num:
                qdmr["transformed_answer"] = "yes"
            else:
                qdmr["transformed_answer"] = "no"

        elif cond_op == "equal":
            if orig_ans_value == value_num:
                qdmr["transformed_answer"] = "yes"
            else:
                qdmr["transformed_answer"] = "no"


transform_to_ans_gen_func = {
    "op_replace_aggregate": gen_ans_op_replace_aggregate,
    "op_replace_arithmetic": gen_ans_op_replace_arithmetic,
    "op_replace_comparison": gen_ans_op_replace_comparison,
    "op_replace_comparative": gen_ans_op_replace_comparative,
    "op_replace_superlative": gen_ans_op_replace_superlative,
    "op_replace_boolean": gen_ans_op_replace_boolean,
    "prune_last_step": gen_ans_prune_last_step,
    "prune_last_step_rm_unused": gen_ans_prune_last_step_rm_unused,
    "prune_step": gen_ans_prune_step,
    "change_last_step": gen_ans_change_last_step,
    "append_boolean_step": gen_ans_append_boolean_step,
}


def is_valid_final_answer(qdmr, max_words=8):
    answer = qdmr["transformed_answer"]

    # check degenerate cases
    if answer is None or \
            (type(answer) in [str, list] and len(answer) == 0):
        return False

    # check if the answer is too long or includes only punctuation marks.
    if type(answer) == str:
        if len(answer.split(" ")) > max_words:
            return False
        if len(answer.strip()) == 0:
            return False
    elif type(answer) == list:
        if len([
            len(span.split(" ")) > max_words
            for span in answer
        ]) > 1:
            return False
        if len([
            len(span.strip()) == 0
            for span in answer
        ]) == len(answer):
            return False

    # in case of a boolean question, make sure the answer is yes/no.
    if qdmr["transformed"][-1].lower().startswith("if") and answer not in ["yes", "no"]:
        return False

    return True


def get_transformed_qdmr_answer(qdmr):
    transform_base, _ = get_transform_base_info(qdmr["transformation"])
    transform_to_ans_gen_func.get(transform_base, lambda q: 'invalid')(qdmr)

    # if the generated answer is not valid - discard it.
    if "transformed_answer" in qdmr and not is_valid_final_answer(qdmr):
        qdmr.pop("transformed_answer")
    if "transformed_answer" in qdmr:
        qdmr["transformed_answer_gen_func"] = transform_to_ans_gen_func[transform_base].__name__

    # try to evaluate the transformed qdmr directly from the step-answers.
    else:
        evalute_qdmr(qdmr)

        # if the generated answer is not valid - discard it.
        if "transformed_answer" in qdmr and not is_valid_final_answer(qdmr):
            qdmr.pop("transformed_answer")
        if "transformed_answer" in qdmr:
            qdmr["transformed_answer_gen_func"] = evalute_qdmr.__name__

    # if the answer is multi-span that includes just one span - make it a single-span answer.
    if "transformed_answer" in qdmr and \
        type(qdmr["transformed_answer"]) == list and \
        len(qdmr["transformed_answer"]) == 1:
        qdmr["transformed_answer"] = qdmr["transformed_answer"][0]
