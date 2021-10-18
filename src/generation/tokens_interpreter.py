from typing import List, Dict, Union, Tuple, Any, Optional

from collections import OrderedDict

from allennlp.common.registrable import Registrable

from src.data.tokenizers.hf_tokenizer_wrapper import HFTokenizerWrapper
from src.data.tokenizers.offset_mapping_utils import get_sequence_boundaries
from src.generation.position_tokens_handlers import PositionTokensHandler
from src.generation import Span, Position


class Interpretation(OrderedDict):
    __slots__ = ["output", "stats", "translation"]

    def __init__(self, **kwargs):
        super().__init__()
        self.output: List[Union[Span, Position, str]] = kwargs["output"]
        self.stats: Dict[str, Any] = kwargs["stats"]
        self.translation: Optional[List[str]] = kwargs["translation"]


class TokensInterpreter(Registrable):
    """
    Note: This class is note best to be initialized with FromParams, as we'd like to use the same
          PositionTokensHandler and HFTokenizerWrapper instances that are used by the model
    """

    default_implementation = "default"

    def __init__(
        self,
        tokenizer_wrapper: HFTokenizerWrapper,
        position_tokens_handler: Optional[PositionTokensHandler] = None,
        multi_span_sep_token: Optional[str] = None,
        position_only: bool = False,
    ) -> None:
        super().__init__()

        self._tokenizer_wrapper = tokenizer_wrapper
        self._position_tokens_handler = position_tokens_handler
        self._multi_span_sep_token = multi_span_sep_token
        self._position_only = position_only

    def __call__(
        self, **kwargs,
    ):
        return self.interpret(**kwargs)

    def interpret(
        self,
        tokens: List[str],
        valid_span_ranges: List[Span] = [],
        origin_texts: Optional[List[str]] = None,
        offset_mapping: Optional[List[Tuple[int, int]]] = None,
        special_tokens_mask: Optional[List[int]] = None,
        explicit_translation_request=False,
    ) -> Interpretation:
        """
        Given a sequence of tokens and position tokens,
        return a sequence composed of tokens and span tuples
        which can naturally be translated into text.
        """
        position_tokens_handler = self._position_tokens_handler

        output = []
        stats = {
            "no_positional_token": 0,
            "invalid_format": 0,
            "decreasing_spans": 0,
            "out_of_valid_range": 0,
        }
        translation = [""]

        has_positional_token = False
        i = 0
        while i < len(tokens):
            token = tokens[i]

            if (
                position_tokens_handler is not None
                and position_tokens_handler.is_position_signal(token)
            ):
                has_positional_token = True

                # TODO: Allow both span and position, but it must be done with separator tokens
                if self._position_only:
                    (
                        position,
                        last_consumed_i,
                    ) = position_tokens_handler.consume_position(
                        tokens=tokens, from_index=i
                    )
                    i = last_consumed_i

                    if position is None:
                        stats["invalid_format"] += 1
                    else:
                        is_valid = False or len(valid_span_ranges) == 0
                        for valid_span_range in valid_span_ranges:
                            if position.is_in(valid_span_range):
                                is_valid = True
                                break
                        if not is_valid:
                            stats["out_of_valid_range"] += 1

                        output.append(position)
                else:
                    span, last_consumed_i = position_tokens_handler.consume_span(
                        tokens=tokens, from_index=i
                    )
                    i = last_consumed_i

                    if span is None:
                        stats["invalid_format"] += 1
                    else:
                        is_decreasing = False
                        is_decreasing = span.start > span.end
                        if is_decreasing:
                            stats["decreasing_spans"] += 1

                        is_valid = False or len(valid_span_ranges) == 0
                        for valid_span_range in valid_span_ranges:
                            if span.is_subspan_of(valid_span_range):
                                is_valid = True
                                break
                        if not is_valid:
                            stats["out_of_valid_range"] += 1

                        if not is_decreasing:
                            output.append(span)
            else:
                output.append(token)
            i += 1

        if not has_positional_token:
            stats["no_positional_token"] += 1

        if (
            origin_texts is not None and offset_mapping is not None
        ) or explicit_translation_request:
            translation = self.translate(
                output, origin_texts, offset_mapping, special_tokens_mask
            )

        return Interpretation(output=output, stats=stats, translation=translation)

    def translate(
        self,
        interpretation_output: List[Union[Span, str]],
        origin_texts: List[str] = None,
        offset_mapping: List[Tuple[int, int]] = None,
        special_tokens_mask: List[int] = None,
    ) -> List[str]:
        tokenizer_wrapper = self._tokenizer_wrapper
        multi_span_sep_token = self._multi_span_sep_token

        output = []

        raw_tokens = []
        for item in interpretation_output + [None]:
            is_none = item is None
            is_span = isinstance(item, Span)
            is_multi_span_sep = (
                (not is_none) and (not is_span) and item == multi_span_sep_token
            )

            if is_none or is_span or is_multi_span_sep:
                if len(raw_tokens) > 0:
                    partial_output = self._ids_to_clean_text(
                        tokenizer_wrapper.convert_tokens_to_ids(raw_tokens)
                    )
                    raw_tokens = []
                    if len(partial_output) > 0:
                        output.append(partial_output)

                if is_none or is_multi_span_sep:
                    continue

            if is_span:
                span = item
                if (
                    origin_texts is not None
                    and offset_mapping is not None
                    and special_tokens_mask is not None
                ):
                    partial_output = TokensInterpreter.extract_span_string_from_origin_texts(
                        span, origin_texts, offset_mapping, special_tokens_mask
                    )
                    output.append(partial_output)
                else:
                    output.append(str(span))
            else:
                raw_tokens.append(item)

        if len(output) == 0:
            output.append("")

        return output

    @staticmethod
    def extract_span_string_from_origin_texts(
        span: Span,
        origin_texts: List[str],
        offset_mapping: List[Tuple[int, int]],
        special_tokens_mask: List[int],
    ):
        if span.start == -1 or span.end == -1:
            return ""
        if span.start >= len(offset_mapping):
            return ""
        if span.end >= len(offset_mapping):
            span = Span(span.start, len(offset_mapping) - 1)

        sequence_ranges = get_sequence_boundaries(special_tokens_mask)
        start_span_origin_index = None
        end_span_origin_index = None
        for i in range(len(origin_texts)):
            sequence_range = sequence_ranges[i]

            if sequence_range.start <= span.start <= sequence_range.end:
                start_span_origin_index = i
                if end_span_origin_index is not None:
                    break
            if sequence_range.start <= span.end <= sequence_range.end:
                end_span_origin_index = i
                if start_span_origin_index is not None:
                    break

        if start_span_origin_index is None:
            if span.start > sequence_ranges[-1].end:
                return ""
            for i in range(len(sequence_ranges)):
                if span.start < sequence_ranges[i].start:
                    start_span_origin_index = i
                    break

        if end_span_origin_index is None:
            if span.end < sequence_ranges[0].start:
                return ""
            for i in range(len(sequence_ranges) - 1, -1, -1):
                if span.end > sequence_ranges[i].end:
                    end_span_origin_index = i
                    break

        span_string_parts = []
        for i in range(start_span_origin_index, end_span_origin_index + 1):
            character_start = offset_mapping[max(sequence_ranges[i][0], span.start)][0]
            character_end = offset_mapping[min(sequence_ranges[i][1], span.end)][1]
            span_string_parts.append(origin_texts[i][character_start:character_end])
        span_string = " ".join(span_string_parts)

        return span_string

    def _ids_to_clean_text(self, generated_ids: List[int]):
        # clear original special ids here, and avoid internal cleaning of all special ids that
        # include the reference tokens, e.g. @@1@@.
        # Originally, there are 4 special ids in BART: <s>, <pad>, </s>, <unk>.
        original_special_ids = self._tokenizer_wrapper.tokenizer.all_special_ids[:4]
        generated_ids_filtered = [
            gen_id
            for gen_id in generated_ids
            if gen_id not in original_special_ids
        ]
        gen_text = self._tokenizer_wrapper.tokenizer.decode(
            generated_ids_filtered, skip_special_tokens=False, clean_up_tokenization_spaces=True
        )
        return gen_text.strip()


TokensInterpreter.register("default")(TokensInterpreter)
