from utils.universal import read_predictions_file
from utils.structs import SampleAnnotation

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
