import torch
from transformers.optimization import Adafactor
from utils.structs import ExperimentConfig, Annotation, Label, BioTag, DatasetConfig
from utils.universal import Option


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


def get_labels_bio_new(token_anno_list: list[Option[Annotation]], gold_annos: list[Annotation]) -> list[Label]:
    """
    Takes all tokens and gold annotations for a sample
    and outputs a labels(one for each token) representing 
    whether a token is at the beginning(B), inside(I), or outside(O) of an entity.
    """
    new_labels = [Label.get_outside_label() for _ in token_anno_list]
    for gold_anno in gold_annos:
        found_start = False
        start_idx = None
        for i, token_anno in enumerate(token_anno_list):
            if token_anno.is_something() and (token_anno.get_value().begin_offset == gold_anno.begin_offset):
                found_start = True
                start_idx = i
                new_labels[i] = Label(gold_anno.label_type, BioTag.begin)
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
                    new_labels[i] = Label(gold_anno.label_type, BioTag.inside)
    return new_labels



def get_annos_from_bio_labels(
        prediction_labels: list[Label],
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


def get_spans_from_bio_labels_token_indices(predictions_sub: list[Label]) -> list[tuple]:
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


def get_bio_label_idx_dicts(all_types: list[str], dataset_config: DatasetConfig) -> tuple[dict[Label, int], dict[int, Label]]:
    """
    get dictionaries mapping from BIO labels to their corresponding indices.
    """
    label_to_idx_dict = {}
    for type_string in all_types:
        assert len(type_string), "Type cannot be an empty string"
        label_to_idx_dict[Label(type_string, BioTag.begin)] = len(
            label_to_idx_dict)
        label_to_idx_dict[Label(type_string, BioTag.inside)] = len(
            label_to_idx_dict)
    label_to_idx_dict[Label.get_outside_label()] = len(label_to_idx_dict)
    idx_to_label_dict = {}
    for label in label_to_idx_dict:
        idx_to_label_dict[label_to_idx_dict[label]] = label
    assert len(label_to_idx_dict) == len(idx_to_label_dict)
    assert len(label_to_idx_dict) == dataset_config.num_types * 2 + 1
    return label_to_idx_dict, idx_to_label_dict
