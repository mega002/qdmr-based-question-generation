from qdmr_transforms.qdmr_identifier import *
from src.models.iterative.reference_utils import get_references

BREAK_SEP = '_'
TRANSFORM_SEP = '+'
TRANSFORM_INFO_SEP = '-'


class QDMRExample:
    blank_ref = "#REF"
    aggregate_to_phrase = {"min": "lowest", \
                           "max": "highest", \
                           "count": "number", \
                           "sum": "sum", \
                           "avg": "average"}

    def __init__(self, example_id, question, qdmr, transform=None):
        self.example_id = example_id
        self.question = question
        self.qdmr = qdmr
        self.original_steps_text = parse_decomposition(qdmr)
        builder = QDMRProgramBuilder(qdmr)
        builder.build()
        self.steps = builder.steps
        self.transform = transform
        self.transformed_question = None
        
    def set_transformed_question(self, transformed_question):
        assert self.transform is not None
        self.transformed_question = transformed_question

    def qdmr_encoding(self):
        prefix = "return "
        return prefix + " ;return ".join(self.qdmr_steps_text())

    def qdmr_steps_text(self):
        step_phrases = [self.step_to_text(step) for step in self.steps]
        return step_phrases

    def step_to_text(self, qdmr_step):
        op = qdmr_step.operator
        if op == "select":
            return self.select_step_phrase(qdmr_step)
        elif op == "project":
            return self.project_step_phrase(qdmr_step)
        elif op == "filter":
            return self.filter_step_phrase(qdmr_step)
        elif op == "aggregate":
            return self.aggregate_step_phrase(qdmr_step)
        elif op == "group":
            return self.group_step_phrase(qdmr_step)
        elif op == "superlative":
            return self.superlative_step_phrase(qdmr_step)
        elif op == "comparative":
            return self.comparative_step_phrase(qdmr_step)
        elif op == "union":
            return self.union_step_phrase(qdmr_step)
        elif op == "intersection":
            return self.intersect_step_phrase(qdmr_step)
        elif op == "discard":
            return self.discard_step_phrase(qdmr_step)
        elif op == "sort":
            return self.sort_step_phrase(qdmr_step)
        elif op == "boolean":
            return self.boolean_step_phrase(qdmr_step)
        elif op == "arithmetic":
            return self.arithmetic_step_phrase(qdmr_step)
        elif op == "comparison":
            return self.comparison_step_phrase(qdmr_step)

    def select_step_phrase(self, step):
        return step.arguments[0]

    def project_step_phrase(self, step):
        projection, ref = step.arguments
        return projection.replace(self.blank_ref, ref)

    def filter_step_phrase(self, step):
        to_filter, filter_condition = step.arguments
        return "%s %s" % (to_filter, filter_condition)

    def aggregate_step_phrase(self, step):
        aggregate, ref = step.arguments
        assert aggregate in self.aggregate_to_phrase.keys()
        agg_phrase = self.aggregate_to_phrase[aggregate]
        return "%s of %s" % (agg_phrase, ref)

    def group_step_phrase(self, step):
        agg, val_ref, key_ref = step.arguments
        agg_phrase = self.aggregate_to_phrase[agg]
        return "%s of %s for each %s" % (agg_phrase, val_ref, key_ref)

    def superlative_step_phrase(self, step):
        agg, entity_ref, attribute_ref = step.arguments
        agg_phrase = self.aggregate_to_phrase[agg]
        return "%s where %s is %s" % (entity_ref, attribute_ref, agg_phrase)

    def comparative_step_phrase(self, step):
        to_filter, attribute, condition = step.arguments
        return "%s where %s %s" % (to_filter, attribute, condition)

    def union_step_phrase(self, step):
        return " , ".join(step.arguments)

    def intersect_step_phrase(self, step):
        projection = step.arguments[0]
        refs = step.arguments[1:]
        return "%s in both %s and %s" % (projection, refs[0], refs[1])

    def discard_step_phrase(self, step):
        set_1, set_2 = step.arguments
        return "%s besides %s" % (set_1, set_2)

    def sort_step_phrase(self, step):
        objects, order = step.arguments
        return "%s sorted by %s" % (objects, order)

    def boolean_step_phrase(self, step):
        op = step.arguments[0]
        if op in ["logical_and", "logical_or"]:
            bool_expr = step.arguments[1]
            refs = step.arguments[2:]
            prefix = "if both" if op == "logical_and" else "if either"
            op_phrase = "and" if op == "logical_and" else "or"
            refs_phrase = f" {op_phrase} ".join(refs)
            return f"{prefix} {refs_phrase} are {bool_expr}"
        elif op == "if_exist":
            ref, cond = step.arguments[1:]
            join_tok = "" if cond.startswith("are") or cond.startswith("is") else "is "
            return f"if any {ref} {join_tok}{cond}"
        ref, condition = step.arguments
        if self.blank_ref in condition:
            # project boolean
            condition = condition.replace(self.blank_ref, ref)
            return f"{condition}"
        # filter boolean
        return f"if {ref} {condition}"

    def arithmetic_step_phrase(self, step):
        op = step.arguments[0]
        refs = step.arguments[1:]
        refs_phrase = " and ".join(refs)
        return "the %s of %s" % (op, refs_phrase)

    def comparison_step_phrase(self, step):
        op = step.arguments[0]
        op_phrase = self.aggregate_to_phrase[op] if op in self.aggregate_to_phrase else op
        refs = step.arguments[1:]
        refs_phrase = " , ".join(refs)
        return "which is %s of %s" % (op_phrase, refs_phrase)

    def is_valid_qdmr(self):
        covered_steps = []
        for i, step in enumerate(self.steps):
            step_refs = get_references(step.step)
            for step_ref in step_refs:
                # only references to previous steps
                if step_ref >= i:
                    return False
                covered_steps.append(step_ref)

        # all nodes should be connected
        covered_steps = list(set(covered_steps))
        if len(covered_steps) != len(self.steps) - 1:
            return False

        return True


def get_orig_example_id(transformed_id, clean=False):
    orig_example_id = transformed_id.split(TRANSFORM_SEP, 1)[0]
    if clean is False:
        return orig_example_id
    else:
        # TODO(mega): handle the special structure of IIRC ids more systematically than these replacements.
        return orig_example_id.replace("_q_", "_q-").split(BREAK_SEP)[-1].replace("q-", "q_")


def get_transform_from_example_id(transformed_id):
    qid_parts = transformed_id.split(TRANSFORM_SEP, 1)
    if len(qid_parts) == 1:
        return ''
    else:
        return transformed_id.split(TRANSFORM_SEP, 1)[1]


def get_transform_base_info(transform):
    transform_parts = transform.split(TRANSFORM_SEP)
    assert len(transform_parts) == 2
    transform_base, transform_info = transform_parts
    return transform_base, transform_info


def get_transform_base_info_parts(transform_info):
    transform_parts = transform_info.split(TRANSFORM_INFO_SEP)
    return transform_parts
