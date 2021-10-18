import csv
import logging
import os
import re
from typing import Optional, List

from allennlp.data.fields import MetadataField, LabelField
from overrides import overrides

from allennlp.common.file_utils import cached_path, open_compressed
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance

from transformers import AutoConfig

from src.data.dataset_readers.base_dataset_reader import BaseDatasetReader
from src.data.fields.dictionary_field import DictionaryField
from src.data.fields.labels_field import LabelsField

logger = logging.getLogger(__name__)


@DatasetReader.register("break_reader_reverse")
class BreakReader(BaseDatasetReader):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    @overrides
    def _direct_read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        logger.info("Reading the dataset:")
        logger.info("Reading file at %s", file_path)
        with open_compressed(file_path) as dataset_file:
            lines = csv.reader(dataset_file)
            header = next(lines, None)
            num_fields = len(header)
            assert num_fields == 5
            if str(header) == "['question_id', 'question_text', 'decomposition', 'operators', 'split']":
                mode = "break"
            elif str(header) == "['id', 'question', 'decomposition', 'transformation', 'type']":
                mode = "transformed"
            else:
                raise NotImplementedError

            for i, line in enumerate(lines):
                assert len(line) == num_fields, "read {} fields, and not {}".format(
                    len(line), num_fields
                )
                if mode == "break":
                    question_id, source, target, _, _ = line
                elif mode == "transformed":
                    question_id, source, _, target, _ = line
                else:
                    raise NotImplementedError

                item = {
                    "qid": question_id,
                    "question": source,
                    "decomposition_obj": {"decomposition": target},
                }

                instance = self._item_to_instance(item)
                if instance is not None:
                    yield instance

    def _item_to_instance(self, item):
        question: Optional[List] = item[
            "question"
        ] if "question" in item else None

        decomposition_obj: List[str] = item["decomposition_obj"]

        if not self._is_training or decomposition_obj is not None:
            instance = self.text_to_instance(
                decomposition_obj=decomposition_obj,
                question=question,
                qid=item["qid"]
            )
            return instance
        return None

    @overrides
    def text_to_instance(
        self,  # type: ignore
        decomposition_obj: str,
        question: Optional[str] = None,
        qid: Optional[str] = "0",
    ) -> Instance:
        tokenizer_wrapper = self._tokenizer_wrapper
        fields = {}
        pad_token_id = tokenizer_wrapper.tokenizer.pad_token_id

        decomposition = process_target(decomposition_obj["decomposition"])
        decomposition_str = " ; ".join(decomposition)

        encoded_input = tokenizer_wrapper.encode(decomposition_str)
        fields["source"] = DictionaryField(
            {
                key: LabelsField(value, padding_value=pad_token_id)
                if key == "input_ids"
                else LabelsField(value)
                for key, value in encoded_input.items()
            }
        )

        if question is not None:
            encoded_target = tokenizer_wrapper.encode(question)
            fields["target_ids"] = LabelsField(
                encoded_target["input_ids"], padding_value=pad_token_id
            )
        else:
            fields["target_ids"] = LabelsField([], padding_value=pad_token_id,)

        if tokenizer_wrapper.tokenizer.bos_token_id is not None:
            start_token = tokenizer_wrapper.tokenizer.bos_token_id
        else:
            start_token = tokenizer_wrapper.tokenizer.pad_token_id
        fields["decoder_start_token_id"] = LabelField(
            start_token, skip_indexing=True
        )

        # make the metadata
        metadata = {
            "decomposition": decomposition_str,
            "gold_question": question
            if question is not None
            else None,
        }

        # for predictor
        metadata["qid"] = qid

        fields["metadata"] = MetadataField(metadata)

        return Instance(fields)


def process_target(target):
    # replace multiple whitespaces with a single whitespace.
    target_new = " ".join(target.split())

    # replace semi-colons with @@SEP@@ token, remove 'return' statements.
    parts = target_new.split(";")
    new_parts = [re.sub(r"return", "", part.strip()).strip() for part in parts]
    return new_parts
