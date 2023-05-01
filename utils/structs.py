from dataclasses import dataclass, field

@dataclass
class Annotation:
    begin_offset: int
    end_offset: int
    label_type: str
    extraction: str
    features: dict = field(default_factory=dict)


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
