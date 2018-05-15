from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader
from allennlp.data.instance import Instance
from allennlp.models import Model
from allennlp.service.predictors.predictor import Predictor


@Predictor.register("WebQABasePredictor")
class WebQABaselinesPredictor(Predictor):
    """
    Wrapper for the WebQA architecture
    """

    def __init__(self, model: Model, dataset_reader: DatasetReader):
        super().__init__(model, dataset_reader)

    @overrides
    def _json_to_instance(self, json: JsonDict):
        question = json["question"]
        instance = self._dataset_reader.text_to_instance(question)
        return instance, {}