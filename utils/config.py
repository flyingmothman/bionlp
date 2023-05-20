from utils.structs import ExperimentConfig, EvaluationType, DatasetConfig, ModelConfig
import glob
import yaml
import importlib
from overrides import overrides


class ExperimentModifier:
    def modify(self, experiment_config: ExperimentConfig) -> ExperimentConfig:
        # modify the experiment config and return in
        raise NotImplementedError("Need to implement the modify method")


class BiggerBatchModifier(ExperimentModifier):
    @overrides
    def modify(self, experiment_config: ExperimentConfig) -> ExperimentConfig:
        experiment_config.model_config.batch_size = 8
        return experiment_config


class EvenBiggerBatchModifier(ExperimentModifier):
    @overrides
    def modify(self, experiment_config: ExperimentConfig) -> ExperimentConfig:
        experiment_config.model_config.batch_size = 16
        return experiment_config


class SmallerBatchModifier(ExperimentModifier):
    @overrides
    def modify(self, experiment_config: ExperimentConfig) -> ExperimentConfig:
        experiment_config.model_config.batch_size = 1
        return experiment_config


class SmallerSpanWidthModifier(ExperimentModifier):
    @overrides
    def modify(self, experiment_config: ExperimentConfig) -> ExperimentConfig:
        experiment_config.model_config.max_span_length = 32
        return experiment_config


class TinySpanWidthModifier(ExperimentModifier):
    @overrides
    def modify(self, experiment_config: ExperimentConfig) -> ExperimentConfig:
        experiment_config.model_config.max_span_length = 16
        return experiment_config


class TestEveryEpochModifier(ExperimentModifier):
    @overrides
    def modify(self, experiment_config: ExperimentConfig) -> ExperimentConfig:
        experiment_config.testing_frequency = 1
        return experiment_config


class TestFrequencyModifier(ExperimentModifier):
    def __init__(self, frequency: int):
        super().__init__()
        self.frequency = frequency

    @overrides
    def modify(self, experiment_config: ExperimentConfig) -> ExperimentConfig:
        experiment_config.testing_frequency = self.frequency
        return experiment_config


class Epochs20Modifier(ExperimentModifier):
    @overrides
    def modify(self, experiment_config: ExperimentConfig) -> ExperimentConfig:
        experiment_config.num_epochs = 20
        return experiment_config

class EpochsCustomModifier(ExperimentModifier):
    def __init__(self, num_epochs: int):
        super().__init__()
        self.num_epochs = num_epochs

    @overrides
    def modify(self, experiment_config: ExperimentConfig) -> ExperimentConfig:
        experiment_config.num_epochs = self.num_epochs
        return experiment_config

class Epochs30Modifier(ExperimentModifier):
    @overrides
    def modify(self, experiment_config: ExperimentConfig) -> ExperimentConfig:
        experiment_config.num_epochs = 30
        return experiment_config

class AccuracyEvaluationModifier(ExperimentModifier):
    @overrides
    def modify(self, experiment_config: ExperimentConfig) -> ExperimentConfig:
        experiment_config.evaluation_type = EvaluationType.accuracy
        return experiment_config

class AdamModifier(ExperimentModifier):
    @overrides
    def modify(self, experiment_config: ExperimentConfig) -> ExperimentConfig:  
        experiment_config.optimizer = 'Adam'
        return experiment_config

class AdafactorModifier(ExperimentModifier):
    @overrides
    def modify(self, experiment_config: ExperimentConfig) -> ExperimentConfig:  
        experiment_config.optimizer = 'Adafactor'
        return experiment_config


def read_dataset_config(config_file_path: str) -> DatasetConfig:
    with open(config_file_path, 'r') as yaml_file:
        dataset_config_raw = yaml.safe_load(yaml_file)
        dataset_config = DatasetConfig(
            train_samples_file_path=dataset_config_raw['train_samples_file_path'],
            valid_samples_file_path=dataset_config_raw['valid_samples_file_path'],
            test_samples_file_path=dataset_config_raw['test_samples_file_path'],
            types_file_path=dataset_config_raw['types_file_path'],
            num_types=int(dataset_config_raw['num_types']),
            dataset_name=dataset_config_raw['dataset_name'],
            dataset_config_name=dataset_config_raw['dataset_config_name'],
            expected_number_of_train_samples=dataset_config_raw['expected_number_of_train_samples'],
            expected_number_of_valid_samples=dataset_config_raw['expected_number_of_valid_samples'],
            expected_number_of_test_samples=dataset_config_raw['expected_number_of_test_samples']
        )
        assert isinstance(dataset_config.num_types, int)
        assert ('test' in dataset_config.test_samples_file_path) \
                or ('change_this' == dataset_config.test_samples_file_path)
        assert 'train' in dataset_config.train_samples_file_path
        assert 'valid' in dataset_config.valid_samples_file_path
        return dataset_config


def get_dataset_config_by_name(dataset_config_name: str) -> DatasetConfig:
    all_config_file_paths = glob.glob('configs/dataset_configs/*.yaml')
    found_dataset_config = None
    for config_file_path in all_config_file_paths:
        dataset_config = read_dataset_config(config_file_path)
        if dataset_config.dataset_config_name == dataset_config_name:
            assert found_dataset_config is None, f"Duplicate dataset config {dataset_config}"
            found_dataset_config = dataset_config
    assert found_dataset_config is not None, f"Should have been able to find dataset config with name {dataset_config_name}"
    assert 'production' in found_dataset_config.train_samples_file_path
    assert 'production' in found_dataset_config.test_samples_file_path
    assert 'production' in found_dataset_config.valid_samples_file_path
    assert 'json' in found_dataset_config.train_samples_file_path
    assert 'json' in found_dataset_config.test_samples_file_path
    assert 'json' in found_dataset_config.valid_samples_file_path
    return found_dataset_config


def get_model_config_from_module(model_config_module_name: str) -> ModelConfig:
    """
    param:
        model_config_module_path(str): the path to the module in which the model config is defined
    """
    model_config_module = importlib.import_module(f'configs.model_configs.{model_config_module_name}')
    return model_config_module.create_model_config(model_config_module_name)


def get_experiment_config(
        model_config_module_name: str,
        dataset_config_name: str,
        modifiers: list[ExperimentModifier] = []
    ) -> ExperimentConfig:
    experiment_config = ExperimentConfig(
        get_dataset_config_by_name(dataset_config_name),
        get_model_config_from_module(model_config_module_name),
        testing_frequency=4
    )

    if len(modifiers):
        for modifier in modifiers:
            experiment_config = modifier.modify(experiment_config)

    return experiment_config
