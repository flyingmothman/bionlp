from utils.structs import Annotation, ModelConfig, DatasetConfig, Sample
from abc import ABC, abstractmethod
import torch

SeqLabelPredictions = list[list[Annotation]]
ClassificationPredictions = list[str]

class ModelBase(ABC, torch.nn.Module):
    """
    The model abstraction used by the framework. Every model
    should inherit this abstraction.
    """

    def __init__(
            self,
            model_config: ModelConfig,
            dataset_config: DatasetConfig
    ):
        super(ModelBase, self).__init__()
        self.model_config = model_config
        self.dataset_config = dataset_config

    @abstractmethod
    def forward(
            self,
            samples: list[Sample]
    ) -> tuple[torch.Tensor, SeqLabelPredictions|ClassificationPredictions]:
        """
        Forward pass.
        :param samples: batch of samples
        :return: a tuple with loss(a tensor) and the batch of predictions made by the model
        """
