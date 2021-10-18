import logging
import os
from typing import Any, Dict, List, Optional

import torch
from overrides import overrides

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.training.metrics.metric import Metric
from allennlp.nn import InitializerApplicator

from transformers import AutoModelForSeq2SeqLM
from transformers.modeling_bart import shift_tokens_right

from src.data.tokenizers.hf_tokenizer_wrapper import HFTokenizerWrapper
from src.generation.tokens_interpreter import TokensInterpreter

logger = logging.getLogger(__name__)


@Model.register("q_gen")
class QuestionGenModel(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        serialization_dir: str,
        pretrained_model: str,
        tokenizer_wrapper: HFTokenizerWrapper,
        generate_while_training: bool = False,
        repetition_penalty: Optional[float] = 2.5,
        metrics: Dict[str, Metric] = {},
        is_dummy: bool = False,
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs,
    ) -> None:

        super().__init__(vocab, **kwargs, serialization_dir=serialization_dir)

        self._tokenizer_wrapper = tokenizer_wrapper
        self._generate_while_training = generate_while_training
        self._repetition_penalty = repetition_penalty

        pre_serialization_dir = os.environ.get("pre_serialization_dir", None)
        if pre_serialization_dir is not None:
            tokenizer_wrapper.tokenizer = tokenizer_wrapper.load(pre_serialization_dir)

        self._tokens_interpreter = TokensInterpreter(
            tokenizer_wrapper,
            multi_span_sep_token=";",
        )

        self._seq2seq = None
        if not is_dummy:
            model_kwargs = {"return_dict": True}
            self._seq2seq = AutoModelForSeq2SeqLM.from_pretrained(
                pretrained_model, **model_kwargs
            )
            self._seq2seq.resize_token_embeddings(len(tokenizer_wrapper.tokenizer))

        self._metrics = metrics

        initializer(self)
        self._tokenizer_wrapper.tokenizer = self._tokenizer_wrapper.load(
            serialization_dir, pending=True
        )
        if not is_dummy:
            self._tokenizer_wrapper.save(serialization_dir)
            self._seq2seq.resize_token_embeddings(len(tokenizer_wrapper.tokenizer))

    def forward(  # type: ignore
        self,
        source: Dict[str, Dict[str, torch.LongTensor]] = None,
        target_ids: torch.Tensor = None,
        decoder_start_token_id: Optional[torch.LongTensor] = None,
        metadata: List[Dict[str, Any]] = None,
        keys_mapping: Dict[str, str] = {},
        **kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        # Handle kwargs by key_mappings, assume there are no identity mappings
        kwargs = kwargs.copy()
        source = kwargs.get(keys_mapping.get("source", None), source)
        target_ids = kwargs.get(keys_mapping.get("target_ids", None), target_ids)
        metadata = kwargs.get(keys_mapping.get("metadata", None), metadata)
        if "source" not in kwargs:
            kwargs["source"] = source
        if "target_ids" not in kwargs:
            kwargs["target_ids"] = target_ids
        if "metadata" not in kwargs:
            kwargs["metadata"] = metadata

        output_dict = {}

        has_target = target_ids is not None and target_ids.shape[-1] > 0
        if has_target:
            output_dict["loss"] = self.loss(source, target_ids)

        if (not self.training) or self._generate_while_training:
            with torch.no_grad():
                batch_generated_ids, scores, probs = self._seq2seq.generate(
                    **source,
                    max_length=100,
                    num_beams=3,
                    num_return_sequences=3,
                    repetition_penalty=self._repetition_penalty,
                    use_cache=True,
                    return_probs=True,
                    decoder_start_token_id=decoder_start_token_id,
                )

                generated_count = batch_generated_ids.shape[0]
                batch_size = source["input_ids"].shape[0]
                num_return_sequences = generated_count // batch_size
                batch_generated_ids = batch_generated_ids.view(
                    batch_size, num_return_sequences, -1
                )

                scores = (
                    scores.clone()
                    .detach()
                    .view(batch_size, num_return_sequences)
                    .tolist()
                )

            output_dict["generated_ids"] = batch_generated_ids

            output_dict["generated_tokens"] = []
            for generated_ids_seqs in batch_generated_ids:
                output_dict["generated_tokens"].append([])
                for generated_ids in generated_ids_seqs:
                    tokens = self._tokenizer_wrapper.convert_ids_to_tokens(
                        generated_ids
                    )
                    output_dict["generated_tokens"][-1].append(tokens)

            output_dict[
                "best_gen_scores"
            ] = scores  # unreliable until an official huggingface feature

            output_dict.update(
                self._task_specific_output_and_evaluation(
                    original_output_dict=output_dict, **kwargs,
                )
            )
        return output_dict

    def loss(
        self, source: Dict[str, torch.Tensor], target_ids: torch.Tensor,
    ):
        if target_ids.dim() == 3:
            target_ids = target_ids[:, 0, :]
        pad_token_id = self._tokenizer_wrapper.tokenizer.pad_token_id

        decoder_input_ids = target_ids[:, :-1].contiguous()  # Why this line?
        lm_labels = target_ids[:, 1:].clone()  # why clone?

        """
        This is what is currently done in huggingface.
        For the multi-task setup, it requires moving the task_token to after `eos`
        instead of replacing `bos` with it, AND setting `decoder_start_token_id`
        to target_ids[-1] instead target_ids[0].
        Requires changing BaseDatasetReader to set `decoder_start_token_id`
        to target_ids[-1] instead target_ids[0], and changing MultiDatasetReader
        to append `task_token` at the end of target_ids.
        decoder_input_ids = (
            target_ids.ne(pad_token_id).sum(dim=1) > 0
        ).long().unsqueeze(
            -1
        ) * target_ids  # shift_tokens_right can't handle an all-pad sequence, so this is a hack
        decoder_input_ids = shift_tokens_right(decoder_input_ids, pad_token_id)
        lm_labels = target_ids"""

        outputs = self._seq2seq(
            **source, decoder_input_ids=decoder_input_ids, use_cache=False,
        )

        lm_logits = outputs["logits"]

        # Same behavior as modeling_bart.py
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)
        assert lm_logits.shape[-1] == self._seq2seq.config.vocab_size
        loss = loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), lm_labels.view(-1))

        """lprobs = torch.nn.functional.log_softmax(lm_logits, dim=-1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, lm_labels, 0.1, ignore_index=pad_token_id
        )"""

        return loss

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {}
        if not self.training or self._generate_while_training:
            for key, metric in self._metrics.items():
                metric_value = metric.get_metric(reset)
                if isinstance(metric_value, dict):
                    if hasattr(metric, "is_main") and metric.is_main:
                        metrics.update(metric_value)
                    else:
                        metrics.update(
                            {
                                f"{key}_{sub_key}": value
                                for sub_key, value in metric_value.items()
                            }
                        )
                else:
                    metrics[key] = metric_value
        return metrics

    def _task_specific_output_and_evaluation(
        self, target_ids, original_output_dict, metadata, **kwargs
    ):
        output_dict: Dict[str, Any] = {}

        generated_tokens = []
        for i in range(len(original_output_dict["generated_tokens"])):
            batch_input_generated_tokens = [
                original_output_dict["generated_tokens"][i][j]
                for j in range(len(original_output_dict["generated_tokens"][i]))
            ]
            generated_tokens.append(batch_input_generated_tokens)

        questions: List[List[str]] = []
        for i in range(len(generated_tokens)):
            question: List[str] = [
                self._tokens_interpreter(
                    tokens=generated_tokens[i][j], explicit_translation_request=True
                ).translation
                for j in range(len(generated_tokens[i]))
            ]
            questions.append(question)

        output_dict["questions"] = questions
        output_dict["qid"] = [metadata_entry["qid"] for metadata_entry in metadata]
        output_dict["decomposition"] = [
            metadata_entry["decomposition"] for metadata_entry in metadata
        ]
        output_dict["gold_question"] = [
            metadata_entry["gold_question"] for metadata_entry in metadata
        ]

        has_target = target_ids is not None and target_ids.shape[-1] > 0
        if has_target:
            generated_ids = original_output_dict["generated_ids"][:, 0, :]
            for metric_key, metric in self._metrics.items():
                if metric.__class__.__name__ == "SARI":
                    # Specific to BART and the way the StrategyQA and Break dataset readers work
                    sources = [" ".join(output_dict["decomposition"]).split(" ")]
                    predictions = [q[0].split(" ") for q in output_dict["questions"][0]]
                    targets = [
                        [g.split(" ")]
                        for g in output_dict["gold_question"]
                    ]
                    metric(sources, predictions, targets)
                else:
                    metric(generated_ids, target_ids)

        return output_dict


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=-100):
    """From fairseq"""
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)

    nll_loss = nll_loss.sum()  # mean()? Scared to break other math.
    smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss
