from typing import List, Tuple

from allennlp.common.registrable import Registrable

from src.generation import Span, Position
from src.data.tokenizers.hf_tokenizer_wrapper import HFTokenizerWrapper


class PositionTokensHandler(Registrable):
    """
    Note: This class is not best to be initialized with FromParams, as we'd like to use the same
          HFTokenizerWrapper instance that is used by the model/dataset reader
    """

    default_implementation = "dedicated"

    def __init__(self, tokenizer_wrapper: HFTokenizerWrapper):
        super().__init__()

        self._tokenizer_wrapper = tokenizer_wrapper

    def convert_position_to_tokens(self, position: int) -> List[str]:
        raise NotImplementedError

    def convert_span_to_tokens(self, span: Span) -> List[str]:
        if span is None:
            return []

        if span.start == -1 or span.end == -1:
            return []

        return None  # Indication to continue handling by the overriding method

    def is_position_signal(self, token: str) -> bool:
        """
        Can be either a positional token (unified or start/end),
        or another dedicated token that indicates the start of a span.

        * If start-end positional tokens are used, then `True` will be returned for an end positional token,
        but consumption will result in an invalid span.
        """
        raise NotImplementedError

    def consume_span(self, tokens: List[str], from_index: int) -> Tuple[Span, int]:
        raise NotImplementedError

    def consume_position(
        self, tokens: List[str], from_index: int
    ) -> Tuple[Position, int]:
        raise NotImplementedError

    def convert_tokens_to_position(self, tokens: List[str]) -> int:
        raise NotImplementedError
