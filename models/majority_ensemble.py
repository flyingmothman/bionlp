from collections import defaultdict
from utils.structs import SampleAnnotation
from utils.universal import read_predictions_file


def get_majority_vote_predictions(prediction_file_paths: list[str]):
    """
    Combine the predictions of agents using the majority voting strategy i.e 
    only keep those predictions that the majority of agents voted 'yes' on.
    """

    votes: defaultdict[SampleAnnotation, int] = defaultdict(lambda: 0)

    for prediction_file_path in prediction_file_paths:
        predictions = read_predictions_file(prediction_file_path)
        for sample_id, annos in predictions.items():
            for anno in annos:
                votes[SampleAnnotation(sample_id, anno.label_type, anno.begin_offset, anno.end_offset)] += 1

    test_annos_majority_votes = {anno: count for anno, count in votes.items() if count > len(prediction_file_paths)//2 }


    majority_predictions = set([anno for anno in test_annos_majority_votes])
    assert len(majority_predictions) == len(test_annos_majority_votes)

    return majority_predictions

