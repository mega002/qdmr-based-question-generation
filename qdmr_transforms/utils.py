import re
import json

DELIMITER = ';'
REF = '#'


def parse_decomposition(qdmr):
    """Parses the decomposition into an ordered list of steps

    Parameters
    ----------
    qdmr : str
        String representation of the QDMR

    Returns
    -------
    list
        returns ordered list of qdmr steps
    """
    # remove digit commas 1,000 --> 1000
    matches = re.findall(r"[\d,]+[,\d]", qdmr)
    for m in matches:
        no_comma = m.replace(",", "")
        qdmr = qdmr.replace(m, no_comma)
    # parse commas as separate tokens
    qdmr = qdmr.replace(",", " , ")
    crude_steps = qdmr.split(DELIMITER)
    steps = []
    for i in range(len(crude_steps)):
        step = crude_steps[i]
        tokens = step.split()
        step = ""
        # remove 'return' prefix
        for tok in tokens[1:]:
            step += tok.strip() + " "
        step = step.strip()
        steps += [step]
    return steps


def extract_ref_idx(ref_argument):
    return int(ref_argument.replace(REF, "").strip())


def load_json(filepath):
    with open(filepath, "r") as reader:
        text = reader.read()
    return json.loads(text)
