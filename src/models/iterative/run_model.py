import torch
import json
import logging
import requests
import time
from copy import deepcopy
from typing import Optional

from allennlp.common.util import import_module_and_submodules
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from allennlp.training.metrics import BooleanAccuracy

from tqdm import tqdm

from src.data.dataset_readers.drop import DropReader
from src.data.dataset_readers.hotpotqa import HotpotQASQuADReader
from src.data.dataset_readers.transformed_qdmrs import read_qdmrs
from src.models.iterative.reference_utils import (
    fill_in_references,
    get_reachability,
    has_reference,
)

logger = logging.getLogger(__name__)

wh_words = ["what", "which", "who", "where", "when"]


def run_model_on_qdmr(qdmr, qa_pairs, predictor, q_gen_predictor, predictor_port, max_loops=10):
    instance = qa_pairs[qdmr["orig_qid"]]
    used_decomposition = deepcopy(qdmr["transformed"])
    if q_gen_predictor is not None:
        # Use the question generation model to transform decomposition steps without references into questions.
        # For steps that include references, use simple heuristics.
        for i in range(len(used_decomposition)):
            used_decomposition[i] = get_step_question(
                q_gen_predictor=q_gen_predictor,
                decomposition_step=used_decomposition[i]
            )

    # Per instance:
    # Until the final step has an answer, find in each iteration
    # all of the steps that are required to answer the last step (including by proxy)
    # and don't have references in them.
    # If it is not possible, return a score of zero for the instance.
    # If it is possible, retrieve paragraphs for these steps,
    # and then pass the step and the paragraphs for it to be answered by the model.
    # Replace the answer in all of the steps that has a reference for it.

    step_answers = [None for i in range(len(used_decomposition))]
    loop_count = 0
    while True:
        # safety check that we don't run into an infinite loop.
        loop_count += 1
        if loop_count >= max_loops:
            print(f"[-] reached maximum number of loop iterations: {qdmr['qid']}")
            break

        reachability = get_reachability([step for step in used_decomposition])
        if reachability is None:
            break

        if step_answers[-1] is not None:
            break

        indices_of_interest = []
        if (sum(reachability[-1])) != 0:
            for i, reachable in enumerate(reachability[-1]):
                if reachable > 0 and sum(reachability[i]) == 0:
                    indices_of_interest.append(i)
        else:
            indices_of_interest.append(len(step_answers) - 1)

        for i in indices_of_interest:
            step_answers[i] = get_answer(
                predictor=predictor,
                question=used_decomposition[i],
                paragraphs=[instance['metadata']["original_passage"]],
                force_yes_no=False,
                predictor_port=predictor_port,
            )  # Return the best non-empty answer

        for i, step in enumerate(used_decomposition):
            used_decomposition[i] = fill_in_references(
                step, step_answers
            )

    output_json_obj = {
        "qid": qdmr["qid"],
        "orig_qid": qdmr["orig_qid"],
        "passage": instance['metadata']["original_passage"],
        "question": instance['metadata']["original_question"],
        "transformed_decomposition": qdmr["transformed"],
        "transformed_evaluated": [step for step in used_decomposition],
        "step_answers": step_answers,
    }

    return output_json_obj


def get_step_question(q_gen_predictor, decomposition_step):
    # if the decomposition step includes references - use heuristics.
    if has_reference(decomposition_step):
        question = None
        for wh_word in wh_words:
            if decomposition_step.startswith(wh_word) or decomposition_step.startswith("if"):
                question = decomposition_step
                break
        if question is None:
            if decomposition_step.startswith("the"):
                question = "what is " + decomposition_step
            elif decomposition_step.startswith("#"):
                question = "what is " + decomposition_step
            else:
                question = "what is the " + decomposition_step

    # if the decomposition step do not include references - use the question generation model.
    else:
        result = q_gen_predictor.predict(decomposition_step)
        question = result["questions"][0][0]

    # make sure the question ends with a question mark.
    if not question.endswith("?"):
        question = question + "?"

    return question


def get_answer(predictor, question, paragraphs, force_yes_no, predictor_port):
    max_score = float("-inf")
    answer = None
    for paragraph in paragraphs:
        if predictor is not None:
            instances = predictor._batch_json_to_instances(
                [
                    {
                        "context": paragraph,
                        "question": question,
                    }
                ]
            )
            result = predictor.predict_batch_instance(
                instances, allow_null=False, force_yes_no=force_yes_no
            )[0]
            if max_score < result["best_span_scores"]:
                max_score = result["best_span_scores"]
                answer = result["best_span_str"]

        else:
            headers = {'content-type': 'application/json'}
            payload = {
                "passage": paragraph,
                "question": question + " yes no"
            }
            response_raw = requests.post(
                f'http://localhost:{predictor_port}/predict',
                data=json.dumps(payload),
                headers=headers
            )
            response = response_raw.json()
            answer = response["answer"]["value"]

    return answer


def main(
    gpu: int,
    qa_model_path: str,
    qa_model_server_port: int,
    q_gen_model_path: str,
    qdmrs_path: str,
    orig_data_path: str,
    dataset_name: str,   # drop / hotpotqa-squad / iirc
    output_predictions_file: str,
    overrides="{}",
):
    import_module_and_submodules("src")

    # if no server port was provided, load model from archive.
    predictor = None
    if qa_model_server_port == 0:
        overrides_dict = {}
        overrides_dict.update(json.loads(overrides))
        archive = load_archive(qa_model_path, cuda_device=gpu, overrides=json.dumps(overrides_dict))
        predictor = Predictor.from_archive(archive)
    assert predictor is not None or qa_model_server_port > 0

    q_gen_predictor = None
    if q_gen_model_path != "":
        q_gen_archive = load_archive(q_gen_model_path, cuda_device=gpu)
        q_gen_predictor = Predictor.from_archive(q_gen_archive, predictor_name="q_gen")

    logger.info("Reading QDMRs file at %s", qdmrs_path)
    qdmrs = read_qdmrs(qdmrs_path, dataset_name)

    dataset_reader = None
    just_qids = True
    if dataset_name in ["drop", "iirc"]:
        dataset_reader = DropReader()
        if dataset_name == "drop":
            just_qids = False
    elif dataset_name == "hotpot-squad":
        dataset_reader = HotpotQASQuADReader()
    logger.info("Reading the dataset:")
    logger.info("Reading file at %s", orig_data_path)
    qa_pairs = dataset_reader.get_qa_pairs_dict(args.orig_data_path, just_qids=just_qids)

    output_dataset = []
    for qdmr_i, qdmr in tqdm(enumerate(qdmrs)):
        # this can happen in IIRC when we could generate QDMR transformations,
        # but we dropped the question when converting into DROP format (which does not support no-answer questions).
        # according to the IIRC paper, there are 30% of the questions cannot be answered given the provided context.
        if qdmr["orig_qid"] not in qa_pairs:
            continue

        try:
            output_json_obj = run_model_on_qdmr(qdmr, qa_pairs, predictor, q_gen_predictor, qa_model_server_port)
            output_dataset.append(output_json_obj)
        except Exception as e:
            print(f"error for {qdmr['qid']}: {e}")
            continue

        if qdmr_i < 5:
            logger.info(output_json_obj)

    if output_predictions_file is not None:
        with open(output_predictions_file, "w", encoding="utf-8") as f:
            json.dump(output_dataset, f, ensure_ascii=False, indent=4)
            logger.info(f"Evaluated {len(output_dataset)} QDMRs.")
            logger.info(f"Output predictions are at: {output_predictions_file}")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parse = argparse.ArgumentParser()
    parse.add_argument("-g", "--gpu", type=int, default="", help="CUDA device")
    parse.add_argument("--qa-model-path", type=str)
    parse.add_argument("--qa-model-server-port", type=int, default=0)
    parse.add_argument("--q-gen-model-path", type=str)
    parse.add_argument("--qdmrs-path", type=str)
    parse.add_argument("--orig-data-path", type=str)
    parse.add_argument("--dataset-name", choices=['drop', 'hotpot-squad', 'iirc'], required=True)
    parse.add_argument("--output-predictions-file", type=str)
    parse.add_argument("-o", "--overrides", type=str, default="{}", help="Overrides")
    args = parse.parse_args()

    main(**vars(args))
