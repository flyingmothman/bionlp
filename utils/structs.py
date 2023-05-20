from dataclasses import dataclass, field
from typing import NamedTuple
from enum import Enum
import torch
from typing import Optional


@dataclass
class Annotation:
    begin_offset: int
    end_offset: int
    label_type: str
    extraction: str
    features: dict = field(default_factory=dict)


class SampleAnnotation(NamedTuple):
    sample_id: str
    type_string: str
    begin_offset: int
    end_offset: int


@dataclass
class AnnotationCollection:
    gold: list[Annotation]
    external: list[Annotation]


@dataclass
class Sample:
    text: str
    id: str
    annos: AnnotationCollection
    features: dict = field(default_factory=dict)


@dataclass
class DatasetConfig:
    train_samples_file_path: str
    valid_samples_file_path: str
    test_samples_file_path: str
    types_file_path: str
    num_types: int
    dataset_name: str
    dataset_config_name: str
    expected_number_of_train_samples: int
    expected_number_of_valid_samples: int
    expected_number_of_test_samples: int


@dataclass
class ModelConfig:
    model_config_name: str
    pretrained_model_name: str
    pretrained_model_output_dim: int
    model_name: str
    learning_rate: float
    batch_size: int

    # Span Rep Model specific
    max_span_length: Optional[int] = None

    # Tokenization specific options
    use_special_bert_tokens: Optional[bool] = True

    # External Anno type 
    external_feature_type: Optional[str] = None


@dataclass
class PreprocessorConfig:
    preprocessor_config_name: str
    preprocessor_class_path: str
    preprocessor_class_init_params: dict


class EvaluationType(Enum):
    """
    Different tasks can have different metrics for
    evaluating performance. This enum attemps to
    capture these differences.
    """
    f1 = 0
    accuracy = 1


@dataclass
class ExperimentConfig:
    dataset_config: DatasetConfig
    model_config: ModelConfig
    testing_frequency: int
    optimizer: str = 'Adam'
    num_epochs: int = 20
    evaluation_type: EvaluationType = EvaluationType.f1


class BioTag(Enum):
    out = 0
    begin = 1
    inside = 2


OUTSIDE_LABEL_STRING = 'o'

class Label:
    def __init__(self, label_type, bio_tag):
        self.label_type = label_type
        self.bio_tag = bio_tag

    def __key(self):
        return self.label_type, self.bio_tag

    def __str__(self):
        if self.bio_tag == BioTag.begin:
            return self.label_type + '-BEGIN'
        elif self.bio_tag == BioTag.inside:
            return self.label_type + '-INSIDE'
        else:
            return OUTSIDE_LABEL_STRING

    def __repr__(self) -> str:
        if self.bio_tag == BioTag.begin:
            return self.label_type + '-BEGIN'
        elif self.bio_tag == BioTag.inside:
            return self.label_type + '-INSIDE'
        else:
            return OUTSIDE_LABEL_STRING

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, Label):
            return (other.label_type == self.label_type) and (other.bio_tag == self.bio_tag)
        raise NotImplementedError()

    @staticmethod
    def get_outside_label():
        return Label(OUTSIDE_LABEL_STRING, BioTag.out)


class DatasetSplit(Enum):
    """
    Represent the variety in splitting data.
    """
    train = 0
    valid = 1
    test = 2



@dataclass
class TrainingArgs:
    device: torch.device
    is_dry_run_mode: bool
    experiment_name: str
    is_testing: bool


SampleAnnotations = dict[str, list[Annotation]]
SampleId = str
