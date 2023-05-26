from dataclasses import dataclass, field
from typing import NamedTuple
from enum import Enum
import torch
from typing import Optional


@dataclass
class Annotation:
    """
    Represents a segment of text which has been labeled with `label_type`. 
    The text itself is stored in `extraction`. Additional information attached 
    to the annotation is stored in `features`.
    """
    begin_offset: int
    end_offset: int
    label_type: str
    extraction: str
    features: dict = field(default_factory=dict)


class SampleAnnotation(NamedTuple):
    """
    Represents a segment of text in sample `sample_id` that has been
    labeled with `type_string`.
    """
    sample_id: str
    type_string: str
    begin_offset: int
    end_offset: int


@dataclass
class AnnotationCollection:
    """
    Container with two sets of annotations: `gold` and `external`.
    `gold` annotations represent the true/gold labels on a sample.
    `external` annotations represent labels that have been generated
    using an external resource like ChatGPT.
    """
    gold: list[Annotation]
    external: list[Annotation]


@dataclass
class Sample:
    """
    Represents one sample of the training data.
    """
    text: str
    id: str
    annos: AnnotationCollection
    features: dict = field(default_factory=dict)


@dataclass
class DatasetConfig:
    """
    Describes a preprocessed dataset.
    """
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
    """
    Describes a machine learning model that uses
    a pretrained transformer.
    """
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
    """
    Describes a dataset preprocessor.
    """
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
    """
    Describes an experiment to be run.
    """
    dataset_config: DatasetConfig
    model_config: ModelConfig
    testing_frequency: int
    optimizer: str = 'Adam'
    num_epochs: int = 20
    evaluation_type: EvaluationType = EvaluationType.f1


class BioTag(Enum):
    """
    The tags in the BIO(beginning, inside, outside) 
    sequence-labeling scheme.
    """
    out = 0
    begin = 1
    inside = 2


OUTSIDE_LABEL_STRING = 'o'

class BioLabel:
    """
    Represents a BIO label.
    """
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
        if isinstance(other, BioLabel):
            return (other.label_type == self.label_type) and (other.bio_tag == self.bio_tag)
        raise NotImplementedError()

    @staticmethod
    def get_outside_label():
        return BioLabel(OUTSIDE_LABEL_STRING, BioTag.out)


class DatasetSplit(Enum):
    """
    Represent the variety in splitting data.
    """
    train = 0
    valid = 1
    test = 2



@dataclass
class TrainingArgs:
    """
    Describing a training process.
    """
    device: torch.device
    is_dry_run_mode: bool
    experiment_name: str
    is_testing: bool


SampleAnnotations = dict[str, list[Annotation]]
SampleId = str


class Dataset(Enum):
    """
    Enumerating all datasets that can currently
    be preprocessed by the framework.
    """
    social_dis_ner = 1
    few_nerd = 2
    genia = 3
    living_ner = 4
    multiconer_coarse = 5
    multiconer_fine = 6
    legaleval_judgement = 7
    legaleval_preamble = 8
    cdr = 9
    chem_drug_ner = 10
    ncbi_disease = 11


class PreprocessorRunType(Enum):
    """
    While preprocessing, it is useful to be able to
    specify whether we want to do a dry-run or
    not -- preprocessing can be expensive in
    terms of compute. 
    """
    production = 0
    dry_run = 1
