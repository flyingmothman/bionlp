from utils.preprocessing import get_sentence_annotator
from utils.structs import Dataset, PreprocessorConfig
import sys

# *********************
#  preprocessing configs for NER models below
# **********************

def config_genia() -> PreprocessorConfig:
    name_of_this_function = sys._getframe().f_code.co_name
    return PreprocessorConfig(
        preprocessor_config_name=name_of_this_function,
        preprocessor_class_path='preprocessors.genia_preprocessor.PreprocessGenia',
        preprocessor_class_init_params={
            'preprocessor_type': name_of_this_function,
            'annotators': []
        }
    )


def config_living_ner_vanilla() -> PreprocessorConfig:
    name_of_this_function = sys._getframe().f_code.co_name
    return PreprocessorConfig(
        preprocessor_config_name=name_of_this_function,
        preprocessor_class_path='preprocessors.livingner_preprocessor.PreprocessLivingNER',
        preprocessor_class_init_params={
            'preprocessor_type': name_of_this_function,
            'annotators': []
        }
    )


def config_ncbi_disease_sentence() -> PreprocessorConfig:
    name_of_this_function = sys._getframe().f_code.co_name
    return PreprocessorConfig(
        preprocessor_config_name=name_of_this_function,
        preprocessor_class_path='preprocessors.ncbi_disease_preprocessor.PreprocessNcbiDisease',
        preprocessor_class_init_params={
            'preprocessor_type': name_of_this_function,
            'annotators': [
                get_sentence_annotator()
            ]
        }
    )


# *********************
#  META preprocessing configs
# **********************
def config_ncbi_meta_all_mistakes_all_gold() -> PreprocessorConfig:
    name_of_this_function = sys._getframe().f_code.co_name
    return PreprocessorConfig(
        preprocessor_config_name=name_of_this_function,
        preprocessor_class_path='preprocessors.ncbi_disease_preprocessor.PreprocessNcbiMetaAllGoldAllMistakes',
        preprocessor_class_init_params={
            'preprocessor_type': name_of_this_function,
            'annotators': []
        }
    )


def config_meta_social_dis_ner() -> PreprocessorConfig:
    name_of_this_function = sys._getframe().f_code.co_name
    return PreprocessorConfig(
        preprocessor_config_name=name_of_this_function,
        preprocessor_class_path='preprocessing.preprocessors.meta.MetaPreprocessor',
        preprocessor_class_init_params={
            'preprocessor_type': name_of_this_function,
            'annotators': [],
            'test_files_folder_full_path': '/Users/harshverma/meta_bionlp/social_dis_ner/test/Apps/harshv_research_nlp',
            'valid_files_folder_full_path': '/Users/harshverma/meta_bionlp/social_dis_ner/training/Apps/harshv_research_nlp',
            'dataset_config_name': 'social_dis_ner_vanilla',
            'dataset': Dataset.social_dis_ner
        }
    )


def config_meta_living_ner() -> PreprocessorConfig:
    name_of_this_function = sys._getframe().f_code.co_name
    return PreprocessorConfig(
        preprocessor_config_name=name_of_this_function,
        preprocessor_class_path='preprocessing.preprocessors.meta.MetaPreprocessor',
        preprocessor_class_init_params={
            'preprocessor_type': name_of_this_function,
            'annotators': [],
            'test_files_folder_full_path': '/Users/harshverma/meta_bionlp/living_ner/test',
            'valid_files_folder_full_path': '/Users/harshverma/meta_bionlp/living_ner/training',
            'dataset_config_name': 'living_ner_window',
            'dataset': Dataset.living_ner
        }
    )


def config_meta_genia() -> PreprocessorConfig:
    name_of_this_function = sys._getframe().f_code.co_name
    return PreprocessorConfig(
        preprocessor_config_name=name_of_this_function,
        preprocessor_class_path='preprocessing.preprocessors.meta.MetaPreprocessor',
        preprocessor_class_init_params={
            'preprocessor_type': name_of_this_function,
            'annotators': [],
            'test_files_folder_full_path': './raw_data_for_meta/genia/test_predictions',
            'valid_files_folder_full_path': './raw_data_for_meta/genia/validation_predictions',
            'dataset_config_name': 'genia',
            'dataset': Dataset.genia
        }
    )
