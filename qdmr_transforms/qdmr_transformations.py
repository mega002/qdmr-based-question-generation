from qdmr_transforms.qdmr_editor import QDMREditor
from qdmr_transforms.qdmr_example import QDMRExample, TRANSFORM_SEP, TRANSFORM_INFO_SEP
from qdmr_transforms.qdmr_identifier import QDMRStep
from qdmr_transforms.question_transformation import transform_comparison_question, \
    transform_append_boolean_question, transform_replace_boolean_question
import re

import random
random.seed(42)

numbers = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
           "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
comparatives = {}
comparatives["BETWEEN"] = ["between"]
comparatives[">"] = ["more than", "above", "larger than", "larger",
                     "older than", "older", "higher than", "higher",
                     "greater than", "greater", "bigger than", "bigger",
                     "after", "over"]
comparatives[">="] = ["at least"]
comparatives["<"] = ["less than", "under", "lower than", "lower",
                     "younger than", "younger", "before", "below",
                     "smaller than", "smaller"]
comparatives["<="] = ["at most"]
comparatives["!="] = ["is not"]
comparatives["start"] = ['start with', 'starts with', 'begin']
comparatives["end"] = ['end with', 'ends with']
comparatives["LIKE"] = ["the letter", "the string", "the word", "the phrase",
                        "contain", "include", "has", "have",
                        "contains", "substring", "includes"]
comparatives["="] = ['is equal to', 'equal to', 'same as',
                     'is ', 'are ', 'was ']
unformatted = {}
unformatted[">="] = ["or later", "or more", "or after"]
unformatted["<="] = ["or earlier", "or less", "or before"]


def is_last_step(index, steps):
    return index == len(steps) - 1


def extract_comparator(condition):
    """
    Returns comparator and value of a
    QDMR comparative step condition
    Parameters
    ----------
    condition : str
        Phrase representing condition of a QDMR step
    Returns
    -------
    tuple
        (comparator, value)
    """
    condition = condition.replace(",", "")
    # extract comparative
    comp = None
    for c in comparatives.keys():
        if comp:
            break
        for trigger in comparatives[c]:
            if trigger in condition:
                comp = c
                break
    if comp:
        # extract value/reference
        value_phrase = condition.split(trigger)[1].strip()
        if comp == "BETWEEN":
            # "between num1 AND num2"
            return comp, value_phrase.upper()
        elif comp:
            # check for unformatted comparators in value phrase
            for c in unformatted.keys():
                for trigger in unformatted[c]:
                    if trigger in condition:
                        comp = c
                        value_phrase = condition.split(trigger)[0].strip()
                        break
        for tok in value_phrase.split():
            if tok.isnumeric():
                return comp, tok
            if tok in numbers.keys():
                return comp, numbers[tok]
        return comp, value_phrase
    return "=", None


def is_float(phrase):
    try:
        float(phrase)
        return True
    except ValueError:
        return False
    return False


# Input is a single QDMRExample
# Output is a list of transformed QDMRExample objects

import copy


def transformed_qdmr(original_example, new_qdmr, transform):
    new_example_id = original_example.example_id + TRANSFORM_SEP + transform
    return QDMRExample(example_id=new_example_id,
                       question=original_example.question,
                       qdmr=new_qdmr,
                       transform=transform)


class QDMRTransform(object):
    def __init__(self, qdmr_example):
        self.qdmr_example = qdmr_example

    def transformations(self):
        """list of transformed QDMRExample objects"""
        return self._transformations()

    def _transformations(self):
        raise NotImplementedError
        return True


class OpReplaceTransform(QDMRTransform):
    """
    Return list of transformed QDMR examples.
    Replaced operators for: aggregate, arithmetic, comparison,
    comparative, superlative & boolean QDMR steps.
    """

    def __init__(self, qdmr_example):
        super(OpReplaceTransform, self).__init__(qdmr_example)

    def _transformations(self):
        transforms = []
        for step in self.qdmr_example.steps:
            if step.operator == "aggregate":
                transforms += self.replace_aggregate(self.qdmr_example)
            elif step.operator == "arithmetic":
                transforms += self.replace_arithmetic(self.qdmr_example)
            elif step.operator == "comparison":
                transforms += self.replace_comparison(self.qdmr_example)
            elif step.operator == "comparative":
                transforms += self.replace_comparative(self.qdmr_example)
            elif step.operator == "superlative":
                transforms += self.replace_superlative(self.qdmr_example)
            elif step.operator == "boolean":
                transforms += self.replace_boolean(self.qdmr_example)
        return transforms

    def replace_aggregate(self, qdmr_example):
        """
        [min, max, sum, avg] --> [min, max, count]
        E.g.:
            "the highest of #2" --> "the number of #2"
        """
        transforms = []
        qdmr_program = qdmr_example.steps
        qdmr_copies = []
        for i in range(len(qdmr_program)):
            step = qdmr_program[i]
            if step.operator == "aggregate" and is_last_step(i, qdmr_program):
                original_op = step.arguments[0]
                if original_op in ["min", "max", "sum", "avg"]:
                    for op in ["min", "max", "count"]:
                        if op != original_op:
                            qdmr_copy = copy.deepcopy(qdmr_example)
                            qdmr_copy.steps[i].arguments[0] = op
                            transforms += [
                                transformed_qdmr(qdmr_example,
                                                 new_qdmr=qdmr_copy.qdmr_encoding(),
                                                 transform=f"op_replace_aggregate{TRANSFORM_SEP}{i}{TRANSFORM_INFO_SEP}{original_op}{TRANSFORM_INFO_SEP}{op}")
                            ]
        return transforms

    def replace_arithmetic(self, qdmr_example):
        """
        [+, -, *, /] --> [+, -, *, /]
        E.g.:
            "difference of #2 and #3" --> "sum of #2 and #3"
        """
        transforms = []
        qdmr_program = qdmr_example.steps
        for i in range(len(qdmr_program)):
            step = qdmr_program[i]
            if step.operator == "arithmetic" and is_last_step(i, qdmr_program):
                original_op = step.arguments[0]
                for op in ["sum", "difference", "multiplication", "division"]:
                    if op != original_op:
                        qdmr_copy = copy.deepcopy(qdmr_example)
                        qdmr_copy.steps[i].arguments[0] = op
                        transforms += [
                            transformed_qdmr(qdmr_example,
                                             new_qdmr=qdmr_copy.qdmr_encoding(),
                                             transform=f"op_replace_arithmetic{TRANSFORM_SEP}{i}{TRANSFORM_INFO_SEP}{original_op}{TRANSFORM_INFO_SEP}{op}")
                        ]
                        if (op == "difference" and
                                len(qdmr_program) in [3, 5] and
                                qdmr_program[0].operator == qdmr_program[1].operator == "select"):
                            # In case of sum --> difference switch generate 2 variant transformation
                            # (1) where first two steps are in original order
                            # (2) where first and second step are switched
                            # Meant to handle question generator producing illogical difference questions
                            #   where the answer is a negative number
                            #   (e.g., "how many more eyes does a person have than fingers?")
                            qdmr_copy_variant = copy.deepcopy(qdmr_copy)
                            first_step, second_step = qdmr_copy_variant.steps[:2]
                            qdmr_copy_variant.steps[0] = second_step
                            qdmr_copy_variant.steps[1] = first_step
                            transforms += [
                                transformed_qdmr(qdmr_example,
                                                 new_qdmr=qdmr_copy_variant.qdmr_encoding(),
                                                 transform=f"op_replace_arithmetic{TRANSFORM_SEP}{i}{TRANSFORM_INFO_SEP}{original_op}{TRANSFORM_INFO_SEP}{op}_VARIANT")
                            ]
        return transforms

    def replace_comparison(self, qdmr_example):
        """
        [min, max] --> [min, max]
        [true, false] --> [true, false]
        E.g.:
            "which is the highest of #1 , #2" --> "which is the lowest of #1 , #2"
            "which is true of #1 , #2" --> "which is false of #1 , #2"
        """
        transforms = []
        qdmr_program = qdmr_example.steps
        for i in range(len(qdmr_program)):
            step = qdmr_program[i]
            if step.operator == "comparison" and is_last_step(i, qdmr_program):
                original_op = step.arguments[0]
                ops = ["min", "max"] if original_op in ["min", "max"] else ["true", "false"]
                for op in ops:
                    if op != original_op:
                        qdmr_copy = copy.deepcopy(qdmr_example)
                        qdmr_copy.steps[i].arguments[0] = op
                        transformed_example = transformed_qdmr(qdmr_example,
                                                               new_qdmr=qdmr_copy.qdmr_encoding(),
                                                               transform=f"op_replace_comparison{TRANSFORM_SEP}{i}{TRANSFORM_INFO_SEP}{original_op}{TRANSFORM_INFO_SEP}{op}")
                        transformed_example.set_transformed_question(
                            transform_comparison_question(transformed_example.question))
                        transforms += [transformed_example]
        return transforms

    def replace_comparative(self, qdmr_example):
        """
        [>, >=, <, <=, !=, =] --> [>, >=, <, <=, !=, =]
        E.g.:
            "#1 where #2 is greater than 375" --> "#1 where #2 is at most 375"
        """
        transforms = []
        qdmr_program = qdmr_example.steps
        for i in range(len(qdmr_program)):
            step = qdmr_program[i]
            if step.operator == "comparative" and is_last_step(i, qdmr_program):
                condition = step.arguments[2]
                comp, value_phrase = extract_comparator(condition)
                if value_phrase is not None:
                    transform_operators = None
                    if ((comp in ["=", "!="] and is_float(value_phrase)) or
                            (comp in [">", ">=", "<", "<="])):
                        # numeric comparatives
                        transform_operators = ["=", "!=", ">", ">=", "<", "<="]
                    elif comp in ["=", "!="]:
                        transform_operators = ["=", "!="]
                    elif comp in ["start", "end"]:
                        transform_operators = ["start", "end"]
                    if transform_operators is not None:
                        for op in transform_operators:
                            if op != comp:
                                op_phrase = comparatives[op][0]
                                condition = f"{op_phrase} {value_phrase}"
                                qdmr_copy = copy.deepcopy(qdmr_example)
                                qdmr_copy.steps[i].arguments[2] = condition
                                transforms += [
                                    transformed_qdmr(qdmr_example,
                                                     new_qdmr=qdmr_copy.qdmr_encoding(),
                                                     transform=f"op_replace_comparative{TRANSFORM_SEP}{i}{TRANSFORM_INFO_SEP}{comp}{TRANSFORM_INFO_SEP}{op}")
                                ]
        return transforms

    def replace_superlative(self, qdmr_example):
        """
        [min, max] --> [min, max]
        E.g.:
            "#1 where #2 is the lowest" --> "#1 where #2 is the highest"
        """
        transforms = []
        qdmr_program = qdmr_example.steps
        for i in range(len(qdmr_program)):
            step = qdmr_program[i]
            if step.operator == "superlative" and is_last_step(i, qdmr_program):
                original_op = step.arguments[0]
                for op in ["min", "max"]:
                    if op != original_op:
                        qdmr_copy = copy.deepcopy(qdmr_example)
                        qdmr_copy.steps[i].arguments[0] = op
                        transforms += [
                            transformed_qdmr(qdmr_example,
                                             new_qdmr=qdmr_copy.qdmr_encoding(),
                                             transform=f"op_replace_superlative{TRANSFORM_SEP}{i}{TRANSFORM_INFO_SEP}{original_op}{TRANSFORM_INFO_SEP}{op}")
                        ]
        return transforms

    def replace_boolean(self, qdmr_example):
        """
        [logical_and, logical_or] --> [logical_and, logical_or]
        [true, fales] --> [true, fales]
        E.g.:
            "if both #3 and #4 are true" --> "if either #3 or #4 are true"
            "if both #3 and #4 are true" --> "if both #3 and #4 are false"
        """
        transforms = []
        qdmr_program = qdmr_example.steps
        for i in range(len(qdmr_program)):
            step = qdmr_program[i]
            if step.operator == "boolean" and is_last_step(i, qdmr_program):
                original_op = step.arguments[0]
                bool_expr = step.arguments[1]
                if original_op in ["logical_and", "logical_or"]:
                    for op in ["logical_and", "logical_or"]:
                        if op != original_op:
                            qdmr_copy = copy.deepcopy(qdmr_example)
                            qdmr_copy.steps[i].arguments[0] = op
                            transforms += [
                                transformed_qdmr(qdmr_example,
                                                 new_qdmr=qdmr_copy.qdmr_encoding(),
                                                 transform=f"op_replace_boolean{TRANSFORM_SEP}{i}{TRANSFORM_INFO_SEP}{original_op}{TRANSFORM_INFO_SEP}{op}")
                            ]
                    for op in ["true", "false"]:
                        if op != bool_expr:
                            qdmr_copy = copy.deepcopy(qdmr_example)
                            qdmr_copy.steps[i].arguments[1] = op
                            transformed_example = transformed_qdmr(qdmr_example,
                                                                   new_qdmr=qdmr_copy.qdmr_encoding(),
                                                                   transform=f"op_replace_boolean{TRANSFORM_SEP}{i}{TRANSFORM_INFO_SEP}{original_op}{TRANSFORM_INFO_SEP}{op}")
                            if op == "false":
                                # check if the question can be automatically transformed to a double negation question
                                transformed_question = transform_replace_boolean_question(transformed_example.question)
                                transformed_example.set_transformed_question(transformed_question)
                            transforms += [transformed_example]
        return transforms


class PruneLastTransform(QDMRTransform):
    """
    Return list of transformed QDMR examples.
    Remove last QDMR step if its operator is:
        filter, project, aggregate, superlative, comparative, sort
    """

    def __init__(self, qdmr_example):
        super(PruneLastTransform, self).__init__(qdmr_example)

    def _transformations(self):
        transforms = []
        if len(self.qdmr_example.steps) < 2:
            return transforms
        transforms += self.remove_last_step(self.qdmr_example)
        return transforms

    def remove_last_step(self, qdmr_example):
        transforms = []
        last_step = qdmr_example.steps[-1]
        remove_last_ops = ["filter", "project", "aggregate", "superlative", "comparative", "sort"]
        if last_step.operator in remove_last_ops:
            qdmr_copy = copy.deepcopy(qdmr_example)
            qdmr_copy.steps = qdmr_copy.steps[:-1]
            transforms = [transformed_qdmr(qdmr_example,
                                           new_qdmr=qdmr_copy.qdmr_encoding(),
                                           transform=f"prune_last_step{TRANSFORM_SEP}{last_step.operator}")]
        else:
            # remove step and resulting unused steps
            qdmr_editor = QDMREditor(qdmr_example.qdmr)
            qdmr_editor.remove_step(len(qdmr_example.steps))
            transforms = [transformed_qdmr(qdmr_example,
                                           new_qdmr=qdmr_editor.get_qdmr_text(),
                                           transform=f"prune_last_step_rm_unused{TRANSFORM_SEP}{last_step.operator}")]
        return transforms


class PruneStepTransform(QDMRTransform):
    """
    Return list of transformed QDMR examples.
    Remove any QDMR step that its operator is:
        filter, superlative, comparative
    """

    def __init__(self, qdmr_example):
        super(PruneStepTransform, self).__init__(qdmr_example)

    def _transformations(self):
        transforms = []
        last_step = self.qdmr_example.steps[-1]
        remove_step_ops = ["filter", "superlative", "comparative"]
        for i in range(len(self.qdmr_example.steps)):
            step = self.qdmr_example.steps[i]
            if step.operator in remove_step_ops:
                transforms += self.remove_step(self.qdmr_example, i + 1)
        return transforms

    def remove_step(self, qdmr_example, i):
        """Return new list of QDMRExample.
        List contains the input qdmr without its i'th step.
        Correct all references to #i to point to step ref[#i]
        Example:
            ['references', '#1 of Making database systems usable', 'number of #2']
            --> ['references', 'number of #1']
        """
        qdmr_editor = QDMREditor(qdmr_example.qdmr)
        qdmr_editor.remove_step(i)
        return [transformed_qdmr(qdmr_example,
                                 new_qdmr=qdmr_editor.get_qdmr_text(),
                                 transform=f"prune_step{TRANSFORM_SEP}{i}")]


class ChangeLastStepTransform(QDMRTransform):
    """
    Return list of transformed QDMR examples.
    Changes the last QDMR step to another step type
    Implemented step changes:
        arithmetic --> [union, comparison]
        comparison --> [union, arithmetic]
    """

    def __init__(self, qdmr_example):
        super(ChangeLastStepTransform, self).__init__(qdmr_example)
        self.change_step_ops = ["arithmetic", "union", "comparison"]

    def _transformations(self):
        transforms = []
        last_step = self.qdmr_example.steps[-1]
        if self.valid_step_to_change(last_step):
            transforms += self.change_last_step(self.qdmr_example)
        return transforms

    def change_last_step(self, qdmr_example):
        """Return new list of QDMRExample.
        List contains the input qdmr with its last step replaced.
        Changes only arithmetic and comparison steps.
        Example:
            ['matte cubes', 'shiny cubes', 'number of #1', 'number of #2', 'which is highest of #3 , #4']
            --> ['matte cubes', 'shiny cubes', 'number of #1', 'number of #2', 'sum of #3 and #4']
        """
        transforms = []
        last_step = self.qdmr_example.steps[-1]
        replacement_steps = self.replacement_union_steps(qdmr_example, last_step) + \
                            self.replacement_boolean_steps(qdmr_example, last_step)
        if last_step.operator == "arithmetic":
            replacement_steps += self.replacement_comparison_steps(qdmr_example, last_step)
        if last_step.operator == "comparison":
            replacement_steps += self.replacement_arithmetic_steps(qdmr_example, last_step)
        for replacement in replacement_steps:
            op_subtype, replacement = replacement[0], replacement[1]
            op_subtype = op_subtype.replace(" ", "_")
            qdmr_copy = copy.deepcopy(qdmr_example)
            qdmr_copy.steps[-1] = replacement
            n = len(qdmr_copy.steps)
            transforms += [
                transformed_qdmr(qdmr_example,
                                 new_qdmr=qdmr_copy.qdmr_encoding(),
                                 transform=f"change_last_step{TRANSFORM_SEP}{n}{TRANSFORM_INFO_SEP}{last_step.operator}{TRANSFORM_INFO_SEP}{replacement.operator}{TRANSFORM_INFO_SEP}{op_subtype}")
            ]
        return transforms

    def valid_step_to_change(self, qdmr_step):
        def two_refs(step):
            return len(get_reference_arguments(step.arguments)) == 2

        def valid_step_type(step):
            if step.operator not in ["arithmetic", "comparison"]:
                return False
            if step.operator == "comparison":
                op = step.arguments[0]
                if op not in ["max", "min"]:
                    return False
            return True

        return two_refs(qdmr_step) and valid_step_type(qdmr_step)

    def replacement_arithmetic_steps(self, qdmr_example, qdmr_step):
        arithmetic_ops = ["sum", "difference", "multiplication", "division"]
        replacement_steps = []
        refs = get_reference_arguments(qdmr_step.arguments)
        for op in arithmetic_ops:
            args = [op] + refs
            replacement_steps += [create_qdmr_step(qdmr_example, "arithmetic", args)]
        return list(zip(arithmetic_ops, replacement_steps))

    def replacement_union_steps(self, qdmr_example, qdmr_step):
        refs = get_reference_arguments(qdmr_step.arguments)
        replacement_steps = [create_qdmr_step(qdmr_example, "union", refs)]
        return list(zip(["union"], replacement_steps))

    def replacement_comparison_steps(self, qdmr_example, qdmr_step):
        comparison_ops = ["max", "min"]
        replacement_steps = []
        refs = get_reference_arguments(qdmr_step.arguments)
        for op in comparison_ops:
            args = [op] + refs
            replacement_steps += [create_qdmr_step(qdmr_example, "comparison", args)]
        return list(zip(comparison_ops, replacement_steps))

    def replacement_boolean_steps(self, qdmr_example, qdmr_step):
        comparatives = ["higher than", "lower than", "the same as"]
        replacement_steps = []
        refs = get_reference_arguments(qdmr_step.arguments)
        for comp in comparatives:
            cond = "is %s %s" % (comp, refs[1])
            args = [refs[0], cond]
            replacement_steps += [create_qdmr_step(qdmr_example, "boolean", args)]
        return list(zip(comparatives, replacement_steps))


def get_reference_arguments(qdmr_arguments):
    def is_reference(phrase):
        phrase = phrase.strip()
        return re.match("^#[0-9]*$", phrase)

    return list(filter(lambda x: is_reference(x), qdmr_arguments))


def create_qdmr_step(qdmr_example, operator, arguments):
    dummy_step = QDMRStep("", operator, arguments)
    qdmr_str = qdmr_example.step_to_text(dummy_step)
    return QDMRStep(qdmr_str, operator, arguments)


class AppendBooleanTransform(QDMRTransform):
    """
    Return list of transformed QDMR examples.
    Appends boolean steps comparing referenced step to a value.
    Implemented for QDMRs where last step is number, i.e., aggregate step:
    """

    def __init__(self, qdmr_example, limit=-1, numeric_qa_data=None):
        """numeric qa file is a json file from original datasets question ids
        to their numeric answers. file computed in pre-processing"""
        self.numeric_qa_data = numeric_qa_data
        self.limit = limit

        super(AppendBooleanTransform, self).__init__(qdmr_example)

    def _transformations(self):
        def original_question_numeric_answer():
            original_question_id = self.qdmr_example.example_id
            if original_question_id in self.numeric_qa_data:
                return self.numeric_qa_data[original_question_id]
            return None

        transforms = []
        last_step = self.qdmr_example.steps[-1]
        if last_step.operator == "aggregate":
            base_cond_values = ["two", 17]
            transformed_original_answer = []
            if self.numeric_qa_data is not None:
                # check if original dataset question has numeric answer in file
                numeric_val = original_question_numeric_answer()
                if numeric_val is not None and numeric_val >= 0:
                    for k in range(0, 3):
                        transformed_original_answer += [numeric_val/k] if k != 0 else []
                        transformed_original_answer += [numeric_val+k, abs(numeric_val-k), numeric_val*k]
                    # keep only integer values
                    transformed_original_answer = [str(int(x)) for x in transformed_original_answer]
            base_cond_values += transformed_original_answer
            # remove duplicate values
            base_cond_values = list(dict.fromkeys(base_cond_values))
            for val in base_cond_values:
                transforms += self.append_boolean_cond(self.qdmr_example, val)

        if self.limit > -1:
            # randomly sample the maximum allowed number of transformations.
            random.shuffle(transforms)
            transforms = transforms[:self.limit]

        return transforms

    def append_boolean_cond(self, qdmr_example, value):
        """Return new list of QDMRExample.
        List contains the input QDMR with boolean step appended.
        Boolean step compares ref to the previous last step to a number value.
        Changes only QDMR where the last step refers to number, i.e., aggregate.
        Example:
            ['times that David Carr was sacked', 'number of #1']
            --> ['times that David Carr was sacked', 'number of #1', 'if #2 is lower than two']
        """
        transforms = []
        last_step = self.qdmr_example.steps[-1]
        last_step_ref = "#%s" % len(self.qdmr_example.steps)
        for cond_phrase in ["lower than", "higher than", "equal to"]:
            cond = "is %s %s" % (cond_phrase, value)
            append = create_qdmr_step(qdmr_example, "boolean", [last_step_ref, cond])
            qdmr_copy = copy.deepcopy(qdmr_example)
            qdmr_copy.steps += [append]
            n = len(qdmr_copy.steps)
            cond_op = cond_phrase.split()[0]
            transformed_example = transformed_qdmr(qdmr_example,
                                                   new_qdmr=qdmr_copy.qdmr_encoding(),
                                                   transform=f"append_boolean_step{TRANSFORM_SEP}{n}{TRANSFORM_INFO_SEP}{last_step.operator}{TRANSFORM_INFO_SEP}{cond_op}{TRANSFORM_INFO_SEP}{value}")
            question_cond = cond
            if cond_phrase == "equal to":
                question_cond = "%s" % value
            if cond_phrase == "lower than":
                question_cond = "less than %s" % value
            if cond_phrase == "higher than":
                question_cond = "more than %s" % value
            transformed_question = transform_append_boolean_question(transformed_example.question, question_cond)
            transformed_example.set_transformed_question(transformed_question)
            transforms += [transformed_example]
        return transforms
