"""
Preprocessing for LivingNER dataset. https://temu.bsc.es/livingner/
"""
import os
import pandas as pd
from utils.structs import Annotation, Sample, DatasetSplit, SampleId, Dataset, AnnotationCollection
from utils.preprocessing import Preprocessor, PreprocessorRunType, Annotator


class PreprocessLivingNER(Preprocessor):
    """
    The LivingNER dataset preprocessor.
    """

    def __init__(
            self,
            dataset_split: DatasetSplit,
            preprocessor_type: str,
            annotators: list[Annotator],
            run_mode: PreprocessorRunType
    ) -> None:
        super().__init__(
            dataset_split=dataset_split,
            preprocessor_type=preprocessor_type,
            dataset=Dataset.living_ner,
            annotators=annotators,
            run_mode=run_mode
        )
        match dataset_split:
            case DatasetSplit.train:
                self.raw_data_folder_path = './livingner-bundle_training_valid_test_background_multilingual/training'
            case DatasetSplit.valid:
                self.raw_data_folder_path = './livingner-bundle_training_valid_test_background_multilingual/valid'
            case DatasetSplit.test:
                self.raw_data_folder_path = './livingner-bundle_training_valid_test_background_multilingual/test'

    def __get_annos_dict(self) -> dict[SampleId, list[Annotation]]:
        """
        Read annotations for each sample from the given file and return
        a dict from sample_ids to corresponding annotations.
        """
        match self.dataset_split:
            case DatasetSplit.valid:
                annos_file_path = f"{self.raw_data_folder_path}/subtask1-NER/validation_entities_subtask1.tsv"
            case DatasetSplit.train:
                annos_file_path = f"{self.raw_data_folder_path}/subtask1-NER/training_entities_subtask1.tsv"
            case DatasetSplit.test:
                annos_file_path = f"{self.raw_data_folder_path}/subtask1-NER/test_entities_subtask1.tsv"
        data_frame = pd.read_csv(annos_file_path, sep='\t')
        sample_to_annos = {}
        for _, row in data_frame.iterrows():
            annos_list = sample_to_annos.get(str(row['filename']), [])
            annos_list.append(
                Annotation(row['off0'], row['off1'], row['label'], row['span']))
            sample_to_annos[str(row['filename'])] = annos_list
        return sample_to_annos

    def get_samples(self) -> list[Sample]:
        """
        Extract samples from the given raw data file provided
        by the organizers.

        """
        document_samples = []
        raw_text_files_folder_path = f"{self.raw_data_folder_path}/text-files"
        data_files_list = os.listdir(raw_text_files_folder_path)
        annos_dict = self.__get_annos_dict()
        for filename in data_files_list:
            data_file_path = os.path.join(raw_text_files_folder_path, filename)
            with open(data_file_path, 'r', encoding="utf-8") as f:
                data = f.read()
                new_str = str()
                for char in data:
                    if ord(char) < 2047:
                        new_str = new_str + char
                    else:
                        new_str = new_str + ' '
                data = new_str
            sample_id = filename[:-4]
            gold_annos = annos_dict.get(sample_id, [])
            document_samples.append(Sample(data, sample_id, AnnotationCollection(gold_annos, [])))
        return document_samples

    def get_entity_types(self) -> list[str]:
        return ['HUMAN', 'SPECIES']
