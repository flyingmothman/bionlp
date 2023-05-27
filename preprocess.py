from utils.structs import PreprocessorRunType, preprocess, DatasetSplit
from all_preprocessor_configs import config_meta_genia

preprocessor_config = config_meta_genia()
preprocess(
    preprocessor_config=preprocessor_config,
    run_mode=PreprocessorRunType.production,
    dataset_splits=[DatasetSplit.test, DatasetSplit.valid, DatasetSplit.train]
)
