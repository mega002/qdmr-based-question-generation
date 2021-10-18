from qdmr_transforms.qdmr_example import get_transform_base_info, get_transform_base_info_parts


def gen_ans_const_op_replace_aggregate(qdmr, transform_info):
    step, orig_op, transform_op = get_transform_base_info_parts(transform_info)
    if int(step) == len(qdmr["transformed"]) - 1:
        ops_to_const = {
            # when changing min to max, the new answer must be greater than or equal to the original answer.
            "min_max": ">=",
            # "min_count": None,
            "max_min": "<=",
            # "max_count": None,
            # assuming that there are no negative numbers.
            "sum_min": "<=",
            "sum_max": "<=",
            # "sum_count": None,
            "avg_min": "<=",
            "avg_max": ">=",
            # "avg_count": None,
        }
        if f"{orig_op}_{transform_op}" in ops_to_const:
            qdmr["transformed_answer_constraints"] = ops_to_const[f"{orig_op}_{transform_op}"]


def gen_ans_const_op_replace_arithmetic(qdmr, transform_info):
    step, orig_op, transform_op = get_transform_base_info_parts(transform_info)
    if int(step) == len(qdmr["transformed"]) - 1:
        # assuming that there are no negative numbers.
        ops_to_const = {
            # when changing sum to difference, the new answer must be smaller than or equal to the original answer.
            "sum_difference": "<=",
            # "sum_multiplication": None,
            # "sum_division": None,
            "difference_sum": ">=",
            # "difference_multiplication": None,
            # "difference_division": None,
            # "multiplication_sum": None,
            # "multiplication_difference": None,
            # "multiplication_division": None,
            # "division_sum": None,
            # "division_difference": None,
            # "division_multiplication": None
        }
        if f"{orig_op}_{transform_op}" in ops_to_const:
            qdmr["transformed_answer_constraints"] = ops_to_const[f"{orig_op}_{transform_op}"]


def gen_ans_const_op_replace_comparison(qdmr, transform_info):
    # When transforming the QDMR, we only have information about the relation between the compared entites,
    # but not their names. Extraction of the names in some cases in possible and is done when computing the answer.
    # operator = get_transform_base_info_parts(transform_info)
    pass


def gen_ans_const_op_replace_comparative(qdmr, transform_info):
    # When transforming the QDMR, we only have information about the relation between the compared entites,
    # but not their names. Extraction of the names in some cases in possible and is done when computing the answer.
    # operator = get_transform_base_info_parts(transform_info)
    pass


def gen_ans_const_op_replace_superlative(qdmr, transform_info):
    # When transforming the QDMR, we only have information about the relation between the compared entites,
    # but not their names. Extraction of the names in some cases in possible and is done when computing the answer.
    # operator = get_transform_base_info_parts(transform_info)
    pass


def gen_ans_const_op_replace_boolean(qdmr, transform_info):
    # No need to for answer constraints because we can generate the exact answer.
    # operator = get_transform_base_info_parts(transform_info)
    pass


def gen_ans_const_prune_last_step(qdmr, transform_info):
    # operator = get_transform_base_info_parts(transform_info)
    pass


def gen_ans_const_prune_last_step_rm_unused(qdmr, transform_info):
    # operator = get_transform_base_info_parts(transform_info)
    pass


def gen_ans_const_prune_step(qdmr, transform_info):
    # step = get_transform_base_info_parts(transform_info)
    pass


def gen_ans_const_change_last_step(qdmr, transform_info):
    _, _, transform_op, _ = get_transform_base_info_parts(transform_info)
    if transform_op in ["boolean", "arithmetic"]:
        qdmr["transformed_answer_constraints"] = transform_op


def gen_ans_const_append_boolean_step(qdmr, transform_info):
    # No need to for answer constraints because we can generate the exact answer.
    pass


transform_to_ans_const_func = {
    "op_replace_aggregate": gen_ans_const_op_replace_aggregate,
    "op_replace_arithmetic": gen_ans_const_op_replace_arithmetic,
    "op_replace_comparison": gen_ans_const_op_replace_comparison,
    "op_replace_comparative": gen_ans_const_op_replace_comparative,
    "op_replace_superlative": gen_ans_const_op_replace_superlative,
    "op_replace_boolean": gen_ans_const_op_replace_boolean,
    "prune_last_step": gen_ans_const_prune_last_step,
    "prune_last_step_rm_unused": gen_ans_const_prune_last_step_rm_unused,
    "prune_step": gen_ans_const_prune_step,
    "change_last_step": gen_ans_const_change_last_step,
    "append_boolean_step": gen_ans_const_append_boolean_step,
}


def get_transformed_qdmr_answer_constraints(qdmr):
    transform_base, transform_info = get_transform_base_info(qdmr["transformation"])
    transform_to_ans_const_func.get(transform_base, lambda q, i: 'invalid')(qdmr, transform_info)
