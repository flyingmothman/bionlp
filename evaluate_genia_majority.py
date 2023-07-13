from models.majority_ensemble import get_majority_voting_results

def majority_genia_sentence_all():
    prediction_file_paths = [
            "./ensembling_data/majority/genia/experiment_genia_bionlp_all_adafactor_2_genia_config_vanilla_model_seq_large_bio_test_epoch_2_predictions.tsv",
            "./ensembling_data/majority/genia/experiment_genia_bionlp_all_adafactor_1_genia_config_vanilla_model_span_large_bio_default_test_epoch_4_predictions.tsv",
            "./ensembling_data/majority/genia/experiment_genia_bionlp_all_adafactor_0_genia_config_vanilla_model_seq_large_crf_bio_test_epoch_2_predictions.tsv"
    ]
    return get_majority_voting_results(dataset_config_name='genia', test_prediction_file_paths=prediction_file_paths)
