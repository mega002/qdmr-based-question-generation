from typing import NamedTuple


class Position(NamedTuple):
    index: int

    def is_in(self, span):
        return span.start <= self.index <= span.end


class Span(NamedTuple):
    start: int
    end: int

    def is_subspan_of(self, other_span):
        return (
            other_span.start <= self.start <= other_span.end
            and other_span.start <= self.end <= other_span.end
        )
