import pandas as pd
from typing import Dict
from tqdm import tqdm

from qdmr_transforms.qdmr_example import QDMRExample
from qdmr_transforms.qdmr_transformations import extract_comparator

OP_REPLACE_STEPS = ["aggregate", "arithmetic", "comparison", "comparative", "superlative", "boolean"]
REF = "#"
CONST = "const"


def read_qdmr_data(csv_file):
    data = pd.read_csv(csv_file)
    return data

def read_qdmr_example(example: Dict[str, str]):
    return QDMRExample(example['question_id'], example['question_text'], example['decomposition'])

def qdmr_dataset_distribution(dataset_file):
    """given a qdmr dataset, return pair of:
    (1) all qdmr programs appearing in dataset;
    (2) all qdmr operators appearing in dataset
    (3) operator distribution by example"""
    dataset_programs = []
    dataset_operators = []
    operator_counts = {}
    num_examples = 0
    qdmr_data = read_qdmr_data(dataset_file)
    for i in tqdm(range(len(qdmr_data)), desc="Loading…", ascii=False, ncols=75):
        try:
            qdmr_example = read_qdmr_example(qdmr_data.iloc[i])
            program = qdmr_program_encoding(qdmr_example.steps)
            dataset_programs += [' '.join(program)]
            dataset_operators += program
            operator_counts = update_operator_counts(operator_counts, program)
            num_examples += 1
        except:
            print("* Error with example: ", qdmr_example.qdmr)
            continue
    # remove duplicates
    dataset_programs = list(dict.fromkeys(dataset_programs))
    dataset_operators = list(dict.fromkeys(dataset_operators))
    print(
        f"* QDMR data contains {len(dataset_programs)} unique programs, {len(dataset_operators)} unique operators, from {num_examples} examples.")
    # normalize operator counts
    operator_dist = operator_counts
    for op in operator_dist:
        operator_dist[op] = float(operator_dist[op]) / num_examples
    results = {}
    results["programs"] = dataset_programs
    results["operators"] = dataset_operators
    results["distribution"] = operator_dist
    return results


def update_operator_counts(counts_dict, program):
    new_dict = counts_dict
    unique_ops = list(dict.fromkeys(program))
    for op in unique_ops:
        if op not in new_dict:
            new_dict[op] = 1
        else:
            new_dict[op] += 1
    return new_dict


def qdmr_program_encoding(qdmr_program):
    """return list of encoded operators + relevant arguments"""
    return [qdmr_step_encoding(step) for step in qdmr_program]


def qdmr_step_encoding(qdmr_step):
    op = qdmr_step.operator
    encoding = op
    if op in OP_REPLACE_STEPS:
        args = get_step_op_arguments(qdmr_step)
        encoding = f"{encoding}_{'_'.join(args)}" if args else encoding
    return encoding


def get_step_op_arguments(qdmr_step):
    op = qdmr_step.operator
    if op in ["aggregate", "comparison", "superlative"]:
        return [qdmr_step.arguments[0]]
    if op == "comparative":
        condition = qdmr_step.arguments[2]
        comp, value_phrase = extract_comparator(condition)
        return [comp]
    if op == "boolean":
        op_arg = qdmr_step.arguments[0]
        if op_arg in ["logical_and", "logical_or"]:
            return [op_arg]
    if op == "arithmetic":
        # extract arithmetic op and argument type
        arith = qdmr_step.arguments[0]
        args = qdmr_step.arguments[1:]
        anonymized_args = [REF if arg.startswith(REF) else CONST for arg in args]
        return [arith] + anonymized_args
    return None
