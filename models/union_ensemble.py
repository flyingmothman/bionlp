from utils.universal import read_predictions_file, get_f1_score_from_sets
from utils.structs import SampleAnnotation, DatasetSplit
from utils.training import get_gold_annos_set

def get_union_predictions(prediction_file_paths: list[str]) -> set[SampleAnnotation]:
    """
    Combine the prediction of agents by simply unioning(mathematical set union) them.
    """
    union_predictions: set[SampleAnnotation] = set()

    for prediction_file_path in prediction_file_paths:
        predictions = read_predictions_file(prediction_file_path)
        for sample_id, annos in predictions.items():
            for anno in annos:
                union_predictions.add(SampleAnnotation(sample_id, anno.label_type, anno.begin_offset, anno.end_offset))

    return union_predictions



def union_results(dataset_config_name: str, test_prediction_file_paths: list[str]):
    """
    Combine the given prediction files for the given dataset using union
    and evaluate the resulting predictions.
    Return the f1, precision, recall after unioning.
    """
    gold_predictions = get_gold_annos_set(dataset_config_name=dataset_config_name, split=DatasetSplit.test)
    union_predictions = get_union_predictions(test_prediction_file_paths)
    f1, precision, recall = get_f1_score_from_sets(gold_set=gold_predictions, predicted_set=union_predictions)
    return f1, precision, recall
