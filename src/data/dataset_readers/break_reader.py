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
from src.data.dataset_readers.standardization_utils import fix_references
from src.data.fields.dictionary_field import DictionaryField
from src.data.fields.labels_field import LabelsField

logger = logging.getLogger(__name__)


@DatasetReader.register("break_reader")
class BreakReader(BaseDatasetReader):
    def __init__(self, add_special_tokens: bool = True, add_prefix: str = "", **kwargs) -> None:
        self._add_special_tokens = add_special_tokens
        super().__init__(**kwargs)

        if self._add_special_tokens:
            self._sep_token = "@@SEP@@"
        else:
            self._sep_token = ";"

        self._add_prefix = add_prefix

    @overrides
    def _reader_specific_init(self):
        if self._add_special_tokens:
            self.additional_special_tokens.add("@@SEP@@")
            for i in range(1, 31):
                self.additional_special_tokens.add(f"@@{i}@@")

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
            assert num_fields in [5, 2]

            for i, line in enumerate(lines):
                assert len(line) == num_fields, "read {} fields, and not {}".format(
                    len(line), num_fields
                )
                if num_fields == 5:
                    question_id, source, target, _, split = line
                    target = process_target(target, fix_refs=self._add_special_tokens)
                    item = {
                        "qid": question_id,
                        "question": self._add_prefix + source,
                        "decomposition_obj": {"decomposition": target},
                    }
                else:   # num_fields == 2
                    question_id, source = line
                    item = {
                        "qid": question_id,
                        "question": self._add_prefix + source,
                    }

                instance = self._item_to_instance(item)
                if instance is not None:
                    yield instance

    def _item_to_instance(self, item):
        question: str = item["question"]
        decomposition_obj: Optional[List] = item[
            "decomposition_obj"
        ] if "decomposition_obj" in item else None

        if not self._is_training or decomposition_obj is not None:
            instance = self.text_to_instance(question, decomposition_obj)
            if instance is not None:
                instance["metadata"].metadata["qid"] = item["qid"]
            return instance
        return None

    @overrides
    def text_to_instance(
        self,  # type: ignore
        question: str,
        decomposition_obj: Optional[List[str]] = None,
    ) -> Instance:
        tokenizer_wrapper = self._tokenizer_wrapper
        fields = {}
        pad_token_id = tokenizer_wrapper.tokenizer.pad_token_id

        encoded_input = tokenizer_wrapper.encode(question)
        fields["source"] = DictionaryField(
            {
                key: LabelsField(value, padding_value=pad_token_id)
                if key == "input_ids"
                else LabelsField(value)
                for key, value in encoded_input.items()
            }
        )

        if decomposition_obj is not None:
            decomposition_str = f" {self._sep_token} ".join(
                decomposition_obj["decomposition"]
            )
            encoded_target = tokenizer_wrapper.encode(decomposition_str)
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
            "question": question,
            "gold_decomposition": decomposition_obj["decomposition"]
            if decomposition_obj is not None
            else None,
        }
        fields["metadata"] = MetadataField(metadata)

        return Instance(fields)


def process_target(target, fix_refs=True):
    # replace multiple whitespaces with a single whitespace.
    target_new = " ".join(target.split())

    if fix_refs:
        # replacing references with special tokens, for example replacing #2 with @@2@@.
        target_new = fix_references(target_new)

    # replace semi-colons with @@SEP@@ token, remove 'return' statements.
    parts = target_new.split(";")
    new_parts = [re.sub(r"return", "", part.strip()).strip() for part in parts]
    return new_parts
