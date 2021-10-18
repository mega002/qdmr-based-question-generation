
import argparse
import json


def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument("data_path", type=str, help="path to an IIRC dataset file")
    # parse.add_argument("articles-path", type=str, help="path to an IIRC articles file")
    parse.add_argument("--output_path", type=str, default="")
    parse.add_argument("--append_yes_no", action="store_true")
    return parse.parse_args()


# def clean_context_paragraph(para):
    # remove hyperlinks
    # raise NotImplementedError


# this code snippet was taken from the original IIRC repository:
# https://github.com/jferguson144/IIRC-baseline/blob/main/make_drop_style.py
def get_drop_format_answer(answer_info):
    a_type = answer_info["type"]
    assert a_type not in ["none", "bad"]

    if a_type == "span":
        answer_spans = [a["text"] for a in answer_info["answer_spans"]]
        answer_num = ""
    elif a_type == "value":
        answer_spans = []
        answer_num = answer_info["answer_value"]
    elif a_type == "binary":
        answer_spans = [answer_info["answer_value"]]
        answer_num = ""

    answer_dict = {"date": {"day": "", "month": "", "year": ""},
                   "number": answer_num,
                   "spans": answer_spans}

    return answer_dict


def main():
    args = get_args()

    # with open(args.articles_path, "r") as fd:
    #     articles = json.load(fd)

    drop_data = {}
    with open(args.data_path, "r") as fd:
        data = json.load(fd)
    for para in data:
        for qa in para["questions"]:
            if qa["answer"]["type"] in ["none", "bad"]:
                continue
            context_paras = [para["text"]]
            para_ids = [para["pid"]]
            for context_para in qa["context"]:
                context_para_id = context_para["passage"]
                if context_para_id != "main":
                    # context_para_text = clean_context_paragraph(articles[context_para_id])
                    context_para_text = f"[{context_para_id}] {context_para['text']}"
                    context_paras.append(context_para_text)
                    para_ids.append(f"{context_para_id}-{context_para['indices'][0]}-{context_para['indices'][1]}")
            pid = '+'.join(para_ids)
            context = ' '.join(context_paras)
            if pid not in drop_data:
                drop_data[pid] = {
                    "passage": context,
                    "qa_pairs": [],
                    "wiki_url": ""
                }
            question = qa["question"]
            if args.append_yes_no is True:
                question += " yes no"
            drop_data[pid]["qa_pairs"].append({
                "query_id": qa["qid"],
                "question": question,
                "answer": get_drop_format_answer(qa["answer"])
            })

    if args.output_path != "":
        output_path = args.output_path
    else:
        output_path = args.data_path.replace(".json", "_drop_format.json")
    with open(output_path, 'w') as fd:
        json.dump(drop_data, fd, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
