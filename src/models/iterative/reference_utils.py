import numpy as np
import string
import re

MAX_STEPS = 10


def _index_to_reference(i):
    return f"#{i + 1}"


def get_ref_pos_in_decomposition_step(ref, decomposition_step):
    pos = []
    for j in [m.start() for m in re.finditer(ref, decomposition_step)]:
        if j + len(ref) == len(decomposition_step) or \
                decomposition_step[j + len(ref)] in string.punctuation or \
                decomposition_step[j+len(ref)] == ' ':
            pos.append(j)

    return pos


def is_ref_in_decomposition(ref, decomposition_step):
    return len(get_ref_pos_in_decomposition_step(ref, decomposition_step)) > 0


def fill_in_references(decomposition_step, step_answers):
    for i in range(MAX_STEPS - 1, -1, -1):
        ref = _index_to_reference(i)
        if i < len(step_answers) and step_answers[i] is not None:
            if type(step_answers[i]) == str:
                step_answer = step_answers[i]
            elif type(step_answers[i]) == list:
                # this happens when the answer is multi-span
                step_answer = ';'.join(step_answers[i])
            else:
                raise RuntimeError
            pos = get_ref_pos_in_decomposition_step(ref, decomposition_step)
            for j in pos[::-1]:
                decomposition_step = decomposition_step[:j] + step_answer + decomposition_step[j+len(ref):]
    return decomposition_step


def has_reference(decomposition_step):
    for i in range(MAX_STEPS - 1, -1, -1):
        ref = _index_to_reference(i)
        if is_ref_in_decomposition(ref, decomposition_step):
            return True
    return False


def get_references(decomposition_step):
    refs = []
    for i in range(MAX_STEPS - 1, -1, -1):
        ref = _index_to_reference(i)
        if is_ref_in_decomposition(ref, decomposition_step):
            refs.append(i)
    return refs


def get_reachability(decomposition):
    ref_graph = np.zeros((len(decomposition), len(decomposition)))
    for i, step in enumerate(decomposition):
        refs = get_references(step)
        if i in refs:
            return None
        ref_graph[i][refs] = 1

    length = 1
    reachability = ref_graph
    while True:
        length += 1
        step_reachability = np.linalg.matrix_power(reachability, length)
        if np.sum(step_reachability) == 0:
            break
        if length == MAX_STEPS:
            return None
        reachability += step_reachability
    return reachability
