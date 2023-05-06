import transformers
import torch
from utils.structs import ExperimentConfig, Annotation
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
        return transformers.Adafactor(
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
