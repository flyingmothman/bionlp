from utils.structs import PreprocessorRunType, DatasetSplit
from utils.preprocessing import preprocess
from configs.preprocessing_configs.all_preprocessing_configs import config_genia

preprocessor_config = config_genia()
preprocess(
    preprocessor_config=preprocessor_config,
    run_mode=PreprocessorRunType.production,
    dataset_splits=[DatasetSplit.test, DatasetSplit.valid, DatasetSplit.train]
)
