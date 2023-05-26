import torch
import importlib
import time
import pandas as pd
from gatenlp import Document
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.optimization import Adafactor
from transformers.tokenization_utils_base import BatchEncoding
from utils.structs import ExperimentConfig, Annotation, BioLabel, BioTag,\
        DatasetConfig, Sample, ModelConfig, TrainingArgs, AnnotationCollection,\
        DatasetSplit, EvaluationType, SampleAnnotations, SampleId, SampleAnnotation
from utils.universal import Option, device, blue, green, red, f1, get_f1_score_from_sets
from utils.config import get_dataset_config_by_name
from pathlib import Path
import glob
import argparse
from pyfzf.pyfzf import FzfPrompt
import csv
import json
from pydoc import locate


def get_optimizer(model, experiment_config: ExperimentConfig):
    if experiment_config.optimizer == 'Ranger':
        # return torch_optimizer.Ranger(model.parameters(), dataset_config['learning_rate'])
        raise Exception("no ranger optimizer")
    elif experiment_config.optimizer == 'Adam':
        return torch.optim.Adam(model.parameters(), experiment_config.model_config.learning_rate)
    elif experiment_config.optimizer == 'AdamW':
        return torch.optim.AdamW(model.parameters(), experiment_config.model_config.learning_rate)
    elif experiment_config.optimizer == 'Adafactor':
        return Adafactor(
            model.parameters(),
            lr=experiment_config.model_config.learning_rate,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )
    else:
        raise Exception(f"optimizer not found: {experiment_config.optimizer}")


def get_bio_labels_for_bert_tokens_batch(
        token_annos_batch: list[list[Option[Annotation]]],
        gold_annos_batch: list[list[Annotation]]
):
    labels_batch = [get_labels_bio_new(token_annos, gold_annos)
                    for token_annos, gold_annos in zip(token_annos_batch, gold_annos_batch)]
    return labels_batch


def get_labels_bio_new(token_anno_list: list[Option[Annotation]], gold_annos: list[Annotation]) -> list[BioLabel]:
    """
    Takes all tokens and gold annotations for a sample
    and outputs a labels(one for each token) representing 
    whether a token is at the beginning(B), inside(I), or outside(O) of an entity.
    """
    new_labels = [BioLabel.get_outside_label() for _ in token_anno_list]
    for gold_anno in gold_annos:
        found_start = False
        start_idx = None
        for i, token_anno in enumerate(token_anno_list):
            if token_anno.is_something() and (token_anno.get_value().begin_offset == gold_anno.begin_offset):
                found_start = True
                start_idx = i
                new_labels[i] = BioLabel(gold_anno.label_type, BioTag.begin)
                break
        if not found_start:
            print(f"WARN: could not generate BIO tags for gold anno: {gold_anno}")
        else:
            assert start_idx is not None
            for i in range(start_idx + 1, len(token_anno_list)):
                curr_token_anno = token_anno_list[i]
                if curr_token_anno.is_nothing() or (curr_token_anno.get_value().begin_offset >= gold_anno.end_offset):
                    break
                else:
                    assert gold_anno.begin_offset <= curr_token_anno.get_value().begin_offset < gold_anno.end_offset
                    new_labels[i] = BioLabel(gold_anno.label_type, BioTag.inside)
    return new_labels



def get_annos_from_bio_labels(
        prediction_labels: list[BioLabel],
        batch_encoding,
        batch_idx: int,
        sample_text: str,
) -> list[Annotation]:
    spans_token_idx = get_spans_from_bio_labels_token_indices(prediction_labels)
    ret = []
    for span in spans_token_idx:
        start_char_span = batch_encoding.token_to_chars(batch_or_token_index=batch_idx, token_index=span[0])
        end_char_span = batch_encoding.token_to_chars(batch_or_token_index=batch_idx, token_index=span[1])
        if (start_char_span is not None) and (end_char_span is not None):
            ret.append(
                Annotation(
                    begin_offset=start_char_span.start,
                    end_offset=end_char_span.end,
                    label_type=span[2],
                    extraction=sample_text[start_char_span.start: end_char_span.end]
                )
            )
    return ret


def get_spans_from_bio_labels_token_indices(predictions_sub: list[BioLabel]) -> list[tuple]:
    span_list = []
    start = None
    start_label = None
    for i, label in enumerate(predictions_sub):
        if label.bio_tag == BioTag.out:
            if start is not None:
                span_list.append((start, i - 1, start_label))
                start = None
                start_label = None
        elif label.bio_tag == BioTag.begin:
            if start is not None:
                span_list.append((start, i - 1, start_label))
            start = i
            start_label = label.label_type
        elif label.bio_tag == BioTag.inside:
            if (start is not None) and (start_label != label.label_type):
                span_list.append((start, i - 1, start_label))
                start = None
                start_label = None
        else:
            raise Exception(f'Illegal label {label}')
    if start is not None:
        span_list.append((start, len(predictions_sub) - 1, start_label))
    return span_list


def enumerate_spans(sentence: list,
                    offset: int = 0,
                    max_span_width: int | None = None,
                    min_span_width: int = 1,
                    ) -> list[tuple[int, int]]:
    """
    Given a sentence, return all token spans within the sentence. 
    Spans are `inclusive`.
    Additionally, you can provide a maximum and minimum span width,
    which will be used to exclude spans outside of this range.

    Finally, you can provide a function mapping `List[T] -> bool`, which will
    be applied to every span to decide whether that span should be included. 
    This allows filtering by length, regex matches, pos tags or any Spacy 
    `Token` attributes, for example.

    # Parameters

    sentence : `List[T]`, required.
        The sentence to generate spans for. The type is generic, as this 
        function can be used with strings, or Spacy `Tokens` or other 
        sequences.
    offset : `int`, optional (default = `0`)
        A numeric offset to add to all span start and end indices. This is 
        helpful if the sentence is part of a larger structure, such as a 
        document, which the indices need to respect.
    max_span_width : `int`, optional (default = `None`)
        The maximum length of spans which should be included. 
        Defaults to len(sentence).
    min_span_width : `int`, optional (default = `1`)
        The minimum length of spans which should be included. Defaults to 1.
    filter_function : `Callable[[List[T]], bool]`, optional (default = `None`)
        A function mapping sequences of the passed type T to a boolean value.
        If `True`, the span is included in the returned spans from the
        sentence, otherwise it is excluded..
    """
    max_span_width = max_span_width or len(sentence)
    spans: list[tuple[int, int]] = []

    for start_index in range(len(sentence)):
        last_end_index = min(start_index + max_span_width, len(sentence))
        first_end_index = min(start_index + min_span_width - 1, len(sentence))
        for end_index in range(first_end_index, last_end_index):
            start = offset + start_index
            end = offset + end_index
            spans.append((start, end))
    return spans


def get_bio_label_idx_dicts(all_types: list[str], dataset_config: DatasetConfig) -> tuple[dict[BioLabel, int], dict[int, BioLabel]]:
    """
    get dictionaries mapping from BIO labels to their corresponding indices.
    """
    label_to_idx_dict = {}
    for type_string in all_types:
        assert len(type_string), "Type cannot be an empty string"
        label_to_idx_dict[BioLabel(type_string, BioTag.begin)] = len(
            label_to_idx_dict)
        label_to_idx_dict[BioLabel(type_string, BioTag.inside)] = len(
            label_to_idx_dict)
    label_to_idx_dict[BioLabel.get_outside_label()] = len(label_to_idx_dict)
    idx_to_label_dict = {}
    for label in label_to_idx_dict:
        idx_to_label_dict[label_to_idx_dict[label]] = label
    assert len(label_to_idx_dict) == len(idx_to_label_dict)
    assert len(label_to_idx_dict) == dataset_config.num_types * 2 + 1
    return label_to_idx_dict, idx_to_label_dict



def get_bert_encoding_for_batch(samples: list[Sample],
                                model_config: ModelConfig,
                                bert_tokenizer) -> BatchEncoding:
    batch_of_sample_texts = [sample.text for sample in samples]
    bert_encoding_for_batch = bert_tokenizer(batch_of_sample_texts,
                                             return_tensors="pt",
                                             is_split_into_words=False,
                                             add_special_tokens=model_config.use_special_bert_tokens,
                                             truncation=True,
                                             padding=True,
                                             max_length=512).to(device)
    return bert_encoding_for_batch


def check_config_integrity():
    """
    Enforce config file name conventions so that 
    it is easier to search for them.
    """
    all_experiment_file_paths = glob.glob('./configs/experiment_configs/*.py')
    all_experiment_names = [Path(file_path).stem for file_path in all_experiment_file_paths]
    # ignore the init file
    assert '__init__' in all_experiment_names, "__init__ file not found in experiment_config package"
    print(all_experiment_names)
    all_experiment_names.remove('__init__')

    all_model_config_file_paths = glob.glob('./configs/model_configs/*.py')
    all_model_config_names = [Path(file_path).stem for file_path in all_model_config_file_paths]
    # ignore the init file
    assert '__init__' in all_model_config_names, "__init__ file not found in model_config package"
    all_model_config_names.remove('__init__')


def parse_training_args() -> TrainingArgs:
    parser = argparse.ArgumentParser(description='Train models and store their output for inspection.')
    parser.add_argument('--production', action='store_true',
                        help='start training on ALL data (10 samples only by default)')
    parser.add_argument('--test', action='store_true', help="Evaluate on the test dataset.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device:", device)

    experiment_name = get_experiment_name_from_user()

    return TrainingArgs(device, not args.production, experiment_name, args.test)


def get_all_experiment_names():
    """
    Get names of all available experiments.
    """
    all_experiment_file_paths = glob.glob('./configs/experiment_configs/*.py')
    all_experiment_names = [Path(file_path).stem for file_path in all_experiment_file_paths]
    # ignore the init file
    assert '__init__' in all_experiment_names, "__init__ file not found in experiment_config package"
    print(all_experiment_names)
    all_experiment_names.remove('__init__')
    return all_experiment_names


def get_experiments(experiment_name: str) -> list[ExperimentConfig]:
    experiments_module = importlib.import_module(f"configs.experiment_configs.{experiment_name}")
    experiments: list[ExperimentConfig] = experiments_module.experiments
    return experiments


def get_experiment_name_from_user() -> str:
    all_experiment_names = get_all_experiment_names()
    # use fzf to select an experiment
    fzf = FzfPrompt()
    chosen_experiment = fzf.prompt(all_experiment_names)[0]
    return chosen_experiment


def create_directory_structure(folder_path: str):
    """
    Creates all the directories on the given `folder_path`.
    Doesn't throw an error if directories already exist.
    Args:
        folder_path: the directory path to create.
    """
    Path(folder_path).mkdir(parents=True, exist_ok=True)



def create_performance_file_header(performance_file_path):
    with open(performance_file_path, 'w') as performance_file:
        mistakes_file_writer = csv.writer(performance_file)
        mistakes_file_writer.writerow(['experiment_name', 'dataset_config_name', 'dataset_split', 'model_name', 'epoch', 'f1_score'])


def print_experiment_info(
    experiment_config: ExperimentConfig,
    dataset_config: DatasetConfig,
    model_config: ModelConfig,
    experiment_name: str,
    is_dry_run: bool,
    is_testing: bool,
    test_evaluation_frequency: int,
) -> None:
    """Print the configurations of the current run"""
    print("\n\n------ EXPERIMENT OVERVIEW --------")
    print(blue("Experiment:"), green(experiment_name))
    print(blue("DRY_RUN_MODE:"), green(is_dry_run))
    print(blue("Is Testing:"), green(is_testing))
    print(blue("Dataset Config Name:"), green(dataset_config.dataset_config_name))
    print(blue("Dataset:"), green(dataset_config.dataset_name))
    print(blue("Model Config Name:"), green(model_config.model_config_name))
    print(blue("Model Name:"), green(model_config.model_name))
    print(blue("Batch_size"), green(model_config.batch_size))
    print(blue("Testing Frequency"), green(test_evaluation_frequency))
    print("----------------------------\n\n")
    print("Experiment Config")
    print(experiment_config)
    print()
    print("Model Config:")
    print(model_config)
    print()
    print("Dataset Config:")
    print(dataset_config)
    print()



def get_train_samples(dataset_config: DatasetConfig) -> list[Sample]:
    return read_samples(dataset_config.train_samples_file_path)


def get_valid_samples(dataset_config: DatasetConfig) -> list[Sample]:
    return read_samples(dataset_config.valid_samples_file_path)


def get_test_samples(dataset_config: DatasetConfig) -> list[Sample]:
    return read_samples(dataset_config.test_samples_file_path)


def read_samples(input_json_file_path: str) -> list[Sample]:
    ret = []
    with open(input_json_file_path, 'r') as f:
        sample_list_raw = json.load(f)
        assert type(sample_list_raw) == list
        for sample_raw in sample_list_raw:
            sample = Sample(
                text=sample_raw['text'],
                id=sample_raw['id'],
                annos=AnnotationCollection(
                    gold=get_annotations_from_raw_list(
                        sample_raw["annos"]["gold"]),
                    external=get_annotations_from_raw_list(
                        sample_raw["annos"]["external"])
                )
            )
            ret.append(sample)
    return ret


def get_annotations_from_raw_list(annotation_raw_list) -> list[Annotation]:
    return [
        Annotation(
            begin_offset=annotation_raw['begin_offset'],
            end_offset=annotation_raw['end_offset'],
            label_type=annotation_raw['label_type'],
            extraction=annotation_raw['extraction'],
            features={}
        )
        for annotation_raw in annotation_raw_list
    ]


def has_external_features(samples: list[Sample]) -> bool:
    for sample in samples:
        if len(sample.annos.external):
            return True
    return False


def check_external_features(samples: list[Sample], external_feature_type: str):
    for sample in samples:
        for anno in sample.annos.external:
            if anno.label_type == external_feature_type:
                return True
    raise RuntimeError(f"External Feature {external_feature_type} not found in data")


def ensure_no_sample_gets_truncated_by_bert(samples: list[Sample], dataset_config: DatasetConfig):
    bert_tokenizer = get_bert_tokenizer()
    # num_truncated = 0
    for sample in samples:
        num_tokens = len(bert_tokenizer(sample.text, truncation=True)['input_ids'])
        if num_tokens == bert_tokenizer.model_max_length:
            raise RuntimeError(f"WARN: In dataset {dataset_config.dataset_name}, the sample {sample.id} is being {red('Truncated')}")
            # num_truncated += 1
    # if num_truncated > 0:
    #     print(blue(f"WARN: Total truncated samples : {num_truncated}"))


def get_bert_tokenizer():
    return AutoTokenizer.from_pretrained('bert-base-uncased')


def prepare_model(model_config: ModelConfig, dataset_config: DatasetConfig):
    all_types = get_all_types(dataset_config.types_file_path, dataset_config.num_types)
    model_class = get_model_class(model_config=model_config)
    return model_class(all_types, model_config=model_config, dataset_config=dataset_config).to(device)


def get_model_class(model_config: ModelConfig) -> type:
    model_class = None
    for model_module_name in get_all_model_modules():
        if model_class is None:
            model_class = locate(f"models.{model_module_name}.{model_config.model_name}")
        else:
            if locate(f"models.{model_module_name}.{model_config.model_name}") is not None:
                print(red(f"WARN: Found duplicate model class: {model_config.model_name}"))
    assert model_class is not None, f"model class name {model_config.model_name} could not be found"
    return model_class


def get_all_model_modules():
    all_module_paths = [file_path for file_path in glob.glob('./models/*.py') 
                        if '__init__.py' not in file_path]
    return [Path(module_path).stem for module_path in all_module_paths]


def get_all_types(types_file_path: str, num_expected_types: int) -> list[str]:
    ret = []
    with open(types_file_path, 'r') as types_file:
        for line in types_file:
            type_name = line.strip()
            if len(type_name):
                ret.append(type_name)
    assert len(ret) == num_expected_types, f"Expected {num_expected_types} num types, " \
                                           f"but found {len(ret)} in types file."
    return ret


def check_label_types(train_samples: list[Sample], valid_samples: list[Sample], all_types: list[str]):
    # verify that all label types in annotations are valid types
    for sample in train_samples:
        for anno in sample.annos.gold:
            assert anno.label_type in all_types, f"anno label type {anno.label_type} not expected"
    for sample in valid_samples:
        for anno in sample.annos.gold:
            assert anno.label_type in all_types, f"anno label type {anno.label_type} not expected"


def get_batches(samples: list[Sample], batch_size: int) -> list[list[Sample]]:
    return [
        samples[batch_start_idx: batch_start_idx + batch_size]
        for batch_start_idx in range(0, len(samples), batch_size)
    ]


def evaluate_validation_split(
        logger,
        model: torch.nn.Module,
        validation_samples: list[Sample],
        mistakes_folder_path: str,
        predictions_folder_path: str,
        error_visualization_folder_path: str,
        validation_performance_file_path: str,
        experiment_name: str,
        dataset_config_name: str,
        model_config_name: str,
        epoch: int,
        experiment_idx: int,
        evaluation_type: EvaluationType
):
    evaluate_dataset_split(
        logger=logger,
        model=model,
        samples=validation_samples,
        mistakes_folder_path=mistakes_folder_path,
        predictions_folder_path=predictions_folder_path,
        error_visualization_folder_path=error_visualization_folder_path,
        performance_file_path=validation_performance_file_path,
        experiment_name=experiment_name,
        dataset_config_name=dataset_config_name,
        model_config_name=model_config_name,
        epoch=epoch,
        dataset_split=DatasetSplit.valid,
        experiment_idx=experiment_idx,
        evaluation_type=evaluation_type
    )


def evaluate_test_split(
        logger,
        model: torch.nn.Module,
        test_samples: list[Sample],
        mistakes_folder_path: str,
        predictions_folder_path: str,
        error_visualization_folder_path: str,
        test_performance_file_path: str,
        experiment_name: str,
        dataset_config_name: str,
        model_config_name: str,
        epoch: int,
        experiment_idx: int,
        evaluation_type: EvaluationType
):
    evaluate_dataset_split(
        logger=logger,
        model=model,
        samples=test_samples,
        mistakes_folder_path=mistakes_folder_path,
        predictions_folder_path=predictions_folder_path,
        error_visualization_folder_path=error_visualization_folder_path,
        performance_file_path=test_performance_file_path,
        experiment_name=experiment_name,
        dataset_config_name=dataset_config_name,
        model_config_name=model_config_name,
        epoch=epoch,
        dataset_split=DatasetSplit.test,
        experiment_idx=experiment_idx,
        evaluation_type=evaluation_type
    )


def evaluate_dataset_split(
        logger,
        model: torch.nn.Module,
        samples: list[Sample],
        mistakes_folder_path: str,
        predictions_folder_path: str,
        error_visualization_folder_path: str,
        performance_file_path: str,
        experiment_name: str,
        dataset_config_name: str,
        model_config_name: str,
        epoch: int,
        dataset_split: DatasetSplit,
        experiment_idx: int,
        evaluation_type: EvaluationType
):
    logger.info(f"\n\nEvaluating {dataset_split.name} data")
    model.eval()
    output_file_prefix = f"{experiment_name}_{experiment_idx}_{dataset_config_name}_{model_config_name}_{dataset_split.name}" \
                         f"_epoch_{epoch}"
    mistakes_file_path = f"{mistakes_folder_path}/{output_file_prefix}_mistakes.tsv"
    predictions_file_path = f"{predictions_folder_path}/{output_file_prefix}_predictions.tsv"

    if evaluation_type == EvaluationType.f1:
        evaluate_with_f1(
                predictions_file_path=predictions_file_path,
                mistakes_file_path=mistakes_file_path,
                samples=samples,
                model=model,
                logger=logger,
                performance_file_path=performance_file_path,
                error_visualization_folder_path=error_visualization_folder_path,
                output_file_prefix=output_file_prefix,
                epoch=epoch,
                experiment_name=experiment_name,
                dataset_config_name=dataset_config_name,
                model_config_name=model_config_name,
                dataset_split=dataset_split
                )
    elif evaluation_type == EvaluationType.accuracy:
        evaluate_with_accuracy(
                predictions_file_path=predictions_file_path,
                mistakes_file_path=mistakes_file_path,
                samples=samples,
                model=model,
                logger=logger,
                performance_file_path=performance_file_path,
                error_visualization_folder_path=error_visualization_folder_path,
                output_file_prefix=output_file_prefix,
                epoch=epoch,
                experiment_name=experiment_name,
                dataset_config_name=dataset_config_name,
                model_config_name=model_config_name,
                dataset_split=dataset_split
                )



def evaluate_with_accuracy(
        predictions_file_path: str,
        mistakes_file_path: str,
        samples: list[Sample],
        model,
        logger,
        performance_file_path: str,
        error_visualization_folder_path: str,
        output_file_prefix: str,
        epoch: int,
        experiment_name: str,
        dataset_config_name: str,
        model_config_name: str,
        dataset_split: DatasetSplit):

    evaluation_start_time = time.time()

    total_num_samples = len(samples)
    num_correct_labels = 0

    with open(predictions_file_path, 'w') as predictions_file, \
            open(mistakes_file_path, 'w') as mistakes_file:
        #  --- GET FILES READY FOR WRITING ---
        predictions_file_writer = csv.writer(predictions_file, delimiter='\t')
        mistakes_file_writer = csv.writer(mistakes_file, delimiter='\t')
        prepare_file_headers_accuracy(mistakes_file_writer, predictions_file_writer)
        with torch.no_grad():
            # Eval Loop
            for sample in samples:
                loss, [predicted_anno] = model([sample])
                assert len(sample.annos.gold) == 1
                gold_anno = sample.annos.gold[0].label_type
                assert gold_anno in ['correct', 'incorrect']
                assert predicted_anno in ['correct', 'incorrect']
                if gold_anno == predicted_anno:
                    num_correct_labels += 1
                # write sample predictions
                store_prediction_accuracy(sample, predicted_anno, predictions_file_writer)
    accuracy = num_correct_labels/total_num_samples
    logger.info(blue(f"Accuracy: {accuracy}"))
    visualize_errors_file_path = f"{error_visualization_folder_path}/{output_file_prefix}_visualize_errors.bdocjs"
    store_performance_result_accuracy(
            performance_file_path=performance_file_path,
            epoch=epoch,
            experiment_name=experiment_name,
            dataset_config_name=dataset_config_name,
            model_config_name=model_config_name,
            dataset_split=dataset_split,
            accuracy=accuracy
    )

    # # upload files to dropbox
    # dropbox_util.upload_file(predictions_file_path)
    # # dropbox_util.upload_file(mistakes_file_path)
    # dropbox_util.upload_file(performance_file_path)

    logger.info(green(f"Done evaluating {dataset_split.name} data.\n"
                      f"Took {str(time.time() - evaluation_start_time)} secs."
                      f"\n\n"))


def prepare_file_headers_accuracy(mistakes_file_writer, predictions_file_writer):
    predictions_file_header = ['sample_id', 'label']
    predictions_file_writer.writerow(predictions_file_header)

    mistakes_file_header = ['sample_id', 'label']
    mistakes_file_writer.writerow(mistakes_file_header)



def store_prediction_accuracy(
        sample: Sample,
        predicted_anno: str,
        predictions_file_writer
):
    # write predictions
    predictions_file_writer.writerow(
        [sample.id, predicted_anno]
    )


def store_performance_result_accuracy(
    performance_file_path,
    accuracy,
    epoch: int,
    experiment_name: str,
    dataset_config_name: str,
    model_config_name: str,
    dataset_split: DatasetSplit
):
    with open(performance_file_path, 'a') as performance_file:
        mistakes_file_writer = csv.writer(performance_file)
        mistakes_file_writer.writerow([experiment_name, dataset_config_name, dataset_split.name,
                                       model_config_name, str(epoch),
                                       str(accuracy)])

def evaluate_with_f1(
        predictions_file_path: str,
        mistakes_file_path: str,
        samples: list[Sample],
        model,
        logger,
        performance_file_path: str,
        error_visualization_folder_path: str,
        output_file_prefix: str,
        epoch: int,
        experiment_name: str,
        dataset_config_name: str,
        model_config_name: str,
        dataset_split: DatasetSplit):

    evaluation_start_time = time.time()

    with open(predictions_file_path, 'w') as predictions_file, \
            open(mistakes_file_path, 'w') as mistakes_file:
        #  --- GET FILES READY FOR WRITING ---
        predictions_file_writer = csv.writer(predictions_file, delimiter='\t')
        mistakes_file_writer = csv.writer(mistakes_file, delimiter='\t')
        prepare_file_headers(mistakes_file_writer, predictions_file_writer)
        with torch.no_grad():
            num_TP_total = 0
            num_FP_total = 0
            num_FN_total = 0
            # Eval Loop
            for sample in samples:
                loss, [predicted_annos] = model([sample])
                gold_annos_set = set(
                    [
                        (gold_anno.begin_offset, gold_anno.end_offset, gold_anno.label_type)
                        for gold_anno in sample.annos.gold
                    ]
                )
                predicted_annos_set = set(
                    [
                        (predicted_anno.begin_offset, predicted_anno.end_offset, predicted_anno.label_type)
                        for predicted_anno in predicted_annos
                    ]
                )

                # calculate true positives, false positives, and false negatives
                true_positives_sample = gold_annos_set.intersection(predicted_annos_set)
                false_positives_sample = predicted_annos_set.difference(gold_annos_set)
                false_negatives_sample = gold_annos_set.difference(predicted_annos_set)
                num_TP = len(true_positives_sample)
                num_TP_total += num_TP
                num_FP = len(false_positives_sample)
                num_FP_total += num_FP
                num_FN = len(false_negatives_sample)
                num_FN_total += num_FN

                # write sample predictions
                store_predictions(sample, predicted_annos, predictions_file_writer)
                # write sample mistakes
                store_mistakes(sample, false_positives_sample, false_negatives_sample,
                               mistakes_file_writer)
    micro_f1, micro_precision, micro_recall = f1(num_TP_total, num_FP_total, num_FN_total)
    logger.info(blue(f"Micro f1 {micro_f1}, prec {micro_precision}, recall {micro_recall}"))
    visualize_errors_file_path = f"{error_visualization_folder_path}/{output_file_prefix}_visualize_errors.bdocjs"
    create_mistakes_visualization(mistakes_file_path, visualize_errors_file_path, samples)
    store_performance_result(
            performance_file_path=performance_file_path,
            f1_score=micro_f1,
            precision_score=micro_precision,
            recall_score=micro_recall,
            epoch=epoch,
            experiment_name=experiment_name,
            dataset_config_name=dataset_config_name,
            model_config_name=model_config_name,
            dataset_split=dataset_split
    )

    # upload files to dropbox
    # dropbox_util.upload_file(visualize_errors_file_path)
    # dropbox_util.upload_file(predictions_file_path)
    # # dropbox_util.upload_file(mistakes_file_path)
    # dropbox_util.upload_file(performance_file_path)

    logger.info(green(f"Done evaluating {dataset_split.name} data.\n"
                      f"Took {str(time.time() - evaluation_start_time)} secs."
                      f"\n\n"))


def prepare_file_headers(mistakes_file_writer, predictions_file_writer):
    prepare_predictions_file_header(predictions_file_writer)

    mistakes_file_header = ['sample_id', 'begin', 'end', 'type', 'extraction', 'mistake_type']
    mistakes_file_writer.writerow(mistakes_file_header)


def prepare_predictions_file_header(predictions_file_writer):
    predictions_file_header = ['sample_id', 'begin', 'end', 'type', 'extraction']
    predictions_file_writer.writerow(predictions_file_header)


def store_predictions(
    sample: Sample,
    predicted_annos_valid: list[Annotation],
    predictions_file_writer
):
    # write predictions
    for anno in predicted_annos_valid:
        predictions_file_writer.writerow(
            [sample.id, str(anno.begin_offset), str(anno.end_offset), anno.label_type, anno.extraction]
        )


def store_mistakes(
        sample: Sample,
        false_positives,
        false_negatives,
        mistakes_file_writer
):
    # write false positive errors
    for span in false_positives:
        start_offset = span[0]
        end_offset = span[1]
        extraction = sample.text[start_offset: end_offset]
        mistakes_file_writer.writerow(
            [sample.id, str(start_offset), str(end_offset), span[2], extraction, 'FP']
        )
    # write false negative errors
    for span in false_negatives:
        start_offset = span[0]
        end_offset = span[1]
        extraction = sample.text[start_offset: end_offset]
        mistakes_file_writer.writerow(
            [sample.id, str(start_offset), str(end_offset), span[2], extraction, 'FN']
        )


def store_performance_result(
    performance_file_path,
    f1_score,
    precision_score,
    recall_score,
    epoch: int,
    experiment_name: str,
    dataset_config_name: str,
    model_config_name: str,
    dataset_split: DatasetSplit
):
    with open(performance_file_path, 'a') as performance_file:
        mistakes_file_writer = csv.writer(performance_file)
        mistakes_file_writer.writerow([experiment_name, dataset_config_name, dataset_split.name,
                                       model_config_name, str(epoch),
                                       str((f1_score, precision_score, recall_score))])


def create_mistakes_visualization(
        mistakes_file_path: str,
        mistakes_visualization_file_path: str,
        validation_samples: list[Sample]
) -> None:
    """
    Create a gate-visualization-file(.bdocjs format) that contains the mistakes
    made by a trained model.

    Args:
        - mistakes_file_path: the file path containing the mistakes of the model
        - gate_visualization_file_path: the gate visualization file path to create 
    """
    mistake_annos_dict = get_mistakes_annos(mistakes_file_path)
    combined_annos_dict = {}
    for sample in validation_samples:
        gold_annos_list = sample.annos.gold
        mistake_annos_list = mistake_annos_dict.get(sample.id, [])
        combined_list = gold_annos_list + mistake_annos_list
        for anno in combined_list:
            anno.begin_offset = int(anno.begin_offset)
            anno.end_offset = int(anno.end_offset)
        combined_annos_dict[sample.id] = combined_list
    sample_to_text_valid = {sample.id: sample.text for sample in validation_samples}
    create_visualization_file(
        mistakes_visualization_file_path,
        combined_annos_dict,
        sample_to_text_valid
    )


def get_mistakes_annos(mistakes_file_path) -> SampleAnnotations:
    """
    Get the annotations that correspond to mistakes for each sample using
    the given mistakes file. 

    Args:
        mistakes_file_path: the file-path representing the file(a .tsv file)
        that contains the mistakes made by a model.
    """
    df = pd.read_csv(mistakes_file_path, sep='\t')
    sample_to_annos = {}
    for _, row in df.iterrows():
        annos_list = sample_to_annos.get(str(row['sample_id']), [])
        annos_list.append(Annotation(int(row['begin']), int(row['end']), 
                                     row['mistake_type'], row['extraction'],
                                     {"type": row['type']}))
        sample_to_annos[str(row['sample_id'])] = annos_list
    return sample_to_annos



def create_visualization_file(
        visualization_file_path: str,
        sample_to_annos: dict[SampleId, list[Annotation]],
        sample_to_text: dict[SampleId, str]
) -> None:
    """
    Create a .bdocjs formatted file which can me directly imported into gate developer.
    We create the file using the given text and annotations.

    Args:
        visualization_file_path: str
            the path of the visualization file we want to create
        annos_dict: Dict[str, List[Anno]]
            mapping from sample ids to annotations
        sample_to_text:
            mapping from sample ids to text
    """
    assert visualization_file_path.endswith(".bdocjs")
    sample_offset = 0
    document_text = ""
    ofsetted_annos = []
    for sample_id in sample_to_annos:
        document_text += (sample_to_text[sample_id] + '\n\n\n\n')
        ofsetted_annos.append(Annotation(sample_offset, len(
            document_text), 'Sample', '', {"id": sample_id}))
        for anno in sample_to_annos[sample_id]:
            new_start_offset = anno.begin_offset + sample_offset
            new_end_offset = anno.end_offset + sample_offset
            gate_features = anno.features.copy()
            gate_features['orig_start_offset'] = anno.begin_offset
            gate_features['orig_end_offset'] = anno.end_offset
            ofsetted_annos.append(Annotation(
                new_start_offset, new_end_offset, anno.label_type, anno.extraction, gate_features))
        sample_offset += (len(sample_to_text[sample_id]) + 4)  # account for added new lines
    gate_document = Document(document_text)
    default_ann_set = gate_document.annset()
    for ofsetted_annotation in ofsetted_annos:
        default_ann_set.add(
            int(ofsetted_annotation.begin_offset),
            int(ofsetted_annotation.end_offset),
            ofsetted_annotation.label_type,
            ofsetted_annotation.features)
    gate_document.save(visualization_file_path)


def read_meta_predictions_file_with_type_information(predictions_file_path: str) -> set[SampleAnnotation]:
    df = pd.read_csv(predictions_file_path, sep='\t')
    ret = set()
    num_removed = 0
    for _, row in df.iterrows():
        sample_id = str(row['sample_id'])
        original_sample_id, start, end, entity_type = sample_id.split('@@@')
        label = row['label']
        assert label in ['correct', 'incorrect']
        if label == 'correct':
            ret.add(SampleAnnotation(str(original_sample_id), entity_type, int(start), int(end)))
        else:
            num_removed += 1
    print(f"removed {num_removed} predictions")
    return ret


def get_train_samples_by_dataset_name(dataset_config_name: str) -> list[Sample]:
    return get_train_samples(get_dataset_config_by_name(dataset_config_name))


def get_valid_samples_by_dataset_name(dataset_config_name: str) -> list[Sample]:
    return get_valid_samples(get_dataset_config_by_name(dataset_config_name))


def get_test_samples_by_dataset_name(dataset_config_name: str) -> list[Sample]:
    return get_test_samples(get_dataset_config_by_name(dataset_config_name))


def evaluate_meta_predictions(meta_predictions_file_path: str, dataset_config_name: str):
    """
    Given meta's predictions in `meta_predictions_file_path` for a dataset
    corresponding to `dataset_config_name`, evaluate meta's performance.
    """
    meta_predictions_set = read_meta_predictions_file_with_type_information(meta_predictions_file_path)
    gold_samples = get_test_samples_by_dataset_name(dataset_config_name)
    gold_predictions: set[SampleAnnotation] = set()
    for gold_sample in gold_samples:
        for gold_anno in gold_sample.annos.gold:
            gold_predictions.add(SampleAnnotation(str(gold_sample.id), gold_anno.label_type, int(gold_anno.begin_offset), int(gold_anno.end_offset)))

    print(get_f1_score_from_sets(gold_predictions, meta_predictions_set))
