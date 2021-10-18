import json
from tqdm import tqdm


def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False
    return True


def text2int(textnum, numwords={}):
    textnum = textnum.replace(",", "").replace("%", "").strip()
    if textnum.isdigit():
        return int(textnum)
    if is_float(textnum):
        return float(textnum)

    if not numwords:
        units = [
            "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
            "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
            "sixteen", "seventeen", "eighteen", "nineteen",
        ]
        tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
        scales = ["hundred", "thousand", "million", "billion", "trillion"]
        numwords["and"] = (1, 0)
        for idx, word in enumerate(units):    numwords[word] = (1, idx)
        for idx, word in enumerate(tens):     numwords[word] = (1, idx * 10)
        for idx, word in enumerate(scales):   numwords[word] = (10 ** (idx * 3 or 2), 0)

    current = result = 0
    for word in textnum.split():
        if word not in numwords:
            raise Exception(f"Illegal word: {word}, textnum: {textnum}")

        scale, increment = numwords[word]
        current = current * scale + increment
        if scale > 100:
            result += current
            current = 0

    return result + current


def is_number_phrase(text):
    try:
        x = text2int(text)
    except:
        return False
    return True


def is_numeric_answer(answer):
    if answer.isdigit() or answer.isnumeric() or is_number_phrase(answer):
        return True
    answer_processed = answer.replace(",", "").replace("%", "").strip()
    if answer_processed.isdigit() or answer_processed.isnumeric():
        # catch strings that have commas 20,000
        return True
    return False


# HOTPOT_PATH_TRAIN = "../qa_datasets/hotpot/hotpot_train_v1.1.json"
# HOTPOT_PATH_DEV = "../qa_datasets/hotpot/hotpot_dev_distractor_v1.json"
# DROP_PATH_TRAIN = "../qa_datasets/drop/drop_dataset_train.json"
# DROP_PATH_DEV = "../qa_datasets/drop/drop_dataset_dev.json"
# CWQ_PATH_TRAIN = "../qa_datasets/cwq/ComplexWebQuestions_train.json"
# CWQ_PATH_DEV = "../qa_datasets/cwq/ComplexWebQuestions_dev.json"

HOTPOT_PATH_TRAIN = "hotpot_train_v1.1.json"
HOTPOT_PATH_DEV = "hotpot_dev_distractor_v1.json"
DROP_PATH_TRAIN = "drop_dataset_train.json"
DROP_PATH_DEV = "drop_dataset_dev.json"
CWQ_PATH_TRAIN = "ComplexWebQuestions_train.json"
CWQ_PATH_DEV = "ComplexWebQuestions_dev.json"
IIRC_PATH_TRAIN = "iirc/train.json"
IIRC_PATH_DEV = "iirc/dev.json"


def read_hotpot_number_answers(dataset_path, dev=None):
    question_answer_map = {}
    dataset_name = "HOTPOT_train" if dev is None else "HOTPOT_dev"
    with open(dataset_path) as f:
        data = json.load(f)
    for i in tqdm(range(len(data)), desc="Loading…", ascii=False, ncols=75):
        answer = data[i]["answer"]
        if is_numeric_answer(answer):
            question_id = dataset_name + "_" + data[i]["_id"]
            try:
                question_answer_map[question_id] = text2int(answer)
            except:
                continue
    print("Went over %s questions from %s. Wrote down %s number answers." % (len(data), dataset_name, len(question_answer_map)))
    return question_answer_map


def read_drop_number_answers(dataset_path, dev=None):
    question_answer_map = {}
    dataset_name = "DROP_train" if dev is None else "DROP_dev"
    with open(dataset_path) as f:
        data = json.load(f)
    for passage in data:
        questions = data[passage]["qa_pairs"]
        for ex in questions:
            question_id = dataset_name + "_" + passage + "_" + ex["query_id"]
            number_answer = ex["answer"]["number"]
            if number_answer != "":
                question_answer_map[question_id] = text2int(number_answer)
    print("Went over %s questions from %s. Wrote down %s number answers." % (len(data), dataset_name, len(question_answer_map)))
    return question_answer_map


def read_cwq_number_answers(dataset_path, dev=None):
    question_answer_map = {}
    dataset_name = "CWQ_train" if dev is None else "CWQ_dev"
    with open(dataset_path) as f:
        data = json.load(f)
    for i in range(len(data)):
        question_id = dataset_name + "_" + data[i]["ID"]
        answer = data[i]["answers"][0]["answer"]
        if answer is not None:
            if is_numeric_answer(answer):
                question_answer_map[question_id] = text2int(answer)
    print("Went over %s questions from %s. Wrote down %s number answers." % (len(data), dataset_name, len(question_answer_map)))
    return question_answer_map


def read_iirc_number_answers(dataset_path, dev=None):
    question_answer_map = {}
    dataset_name = "IIRC_train" if dev is None else "IIRC_dev"
    with open(dataset_path) as f:
        data = json.load(f)
    for passage in data:
        passage_id = passage["pid"]
        questions = passage["questions"]
        for ex in questions:
            question_id = dataset_name + "_" + passage_id + "_" + ex["qid"]
            answer_data = ex["answer"]
            answer_val = answer_data["answer_value"] if answer_data["type"] == "value" else None
            if answer_data["type"] == "span":
                answer_val = answer_data["answer_spans"][0]["text"]
            if answer_val is not None and is_numeric_answer(answer_val):
                try:
                    question_answer_map[question_id] = text2int(answer_val)
                except:
                    continue
    print("Went over %s questions from %s. Wrote down %s number answers." % (len(data), dataset_name, len(question_answer_map)))
    return question_answer_map


def merge_two_dicts(x, y):
    z = x.copy()  # start with x's keys and values
    z.update(y)  # modifies z with y's keys and values & returns None
    return z


def get_all_numeric_answers(split):
    assert split in ["train", "dev"]
    dev = True if split == "dev" else None
    hotpot_path = HOTPOT_PATH_TRAIN if split == "train" else HOTPOT_PATH_DEV
    drop_path = DROP_PATH_TRAIN if split == "train" else DROP_PATH_DEV
    cwq_path = CWQ_PATH_TRAIN if split == "train" else CWQ_PATH_DEV
    iirc_path = IIRC_PATH_TRAIN if split == "train" else IIRC_PATH_DEV
    hotpot = read_hotpot_number_answers(hotpot_path, dev=dev)
    drop = read_drop_number_answers(drop_path, dev=dev)
    cwq = read_cwq_number_answers(cwq_path, dev=dev)
    iirc = read_iirc_number_answers(iirc_path, dev=dev)
    full_qa = merge_two_dicts(hotpot, drop)
    full_qa = merge_two_dicts(full_qa, cwq)
    full_qa = merge_two_dicts(full_qa, iirc)
    return full_qa


def write_numeric_answers(out_file, split=None):
    assert split in [None, "train", "dev", "all"]
    if split == "train":
        full_qa = get_all_numeric_answers("train")
    elif split == "dev":
        full_qa = get_all_numeric_answers("dev")
    else:
        full_qa = merge_two_dicts(get_all_numeric_answers("train"), get_all_numeric_answers("dev"))
    with open(out_file, 'w', encoding='utf8') as fp:
        json.dump(full_qa, fp, indent=4)
    return True


write_numeric_answers("numeric_qa_examples_test_NEW.json", split="all")
