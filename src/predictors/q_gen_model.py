from typing import List, Dict, Any

from allennlp.models import Model
from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor


@Predictor.register("q_gen")
class QuestionGenPredictor(Predictor):
    """
    Predictor for question generator model (class q_gen)
    """

    def predict(self, decomposition_str: str) -> JsonDict:
        return self.predict_json({"decomposition_str": decomposition_str})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"source": "..."}``.
        """
        return self._dataset_reader.text_to_instance({"decomposition": json_dict["decomposition_str"]})
