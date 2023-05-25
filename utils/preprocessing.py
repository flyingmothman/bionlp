from structs import *
import json
from abc import ABC, abstractmethod
from pydoc import locate
from enum import Enum
from typing import Type
import yaml
from utils.structs import PreprocessorConfig
from utils.universal import blue, green, red, open_make_dirs, \
        create_directory_structure, print_section, assert_equals 
from gatenlp import Document


class Annotator(ABC):
    """
    Represents a piece of computation that annotates text in some way.
    """

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def annotate(self, samples: list[Sample], dataset_split: DatasetSplit) -> list[Sample]:
        """
        Annotate the given samples.
        """
        print(f"Annotator : {self.name}")
        return self.annotate_helper(samples, dataset_split=dataset_split)

    @abstractmethod
    def annotate_helper(self, samples: list[Sample], dataset_split: DatasetSplit) -> list[Sample]:
        """
        Annotate the given samples.
        """


class Preprocessor(ABC):
    """
    An abstraction for preprocessing a dataset. To preprocess 
    a dataset, one needs to implement the `get_samples` and 
    `get_entity_types` methods.
    """

    def __init__(
            self,
            dataset_split: DatasetSplit,
            preprocessor_type: str,
            dataset: Dataset,
            annotators: list[Annotator],
            run_mode: PreprocessorRunType
    ) -> None:
        """
        Creates a preprocessor configured with some file paths
        that represent its output locations.

        Args:
            preprocessor_type: the type of preprocessor (mention details like annotators)
            dataset: the name of the dataset we are preprocessing
            annotators: the list of annotators we need to run on the dataset
            run_mode: In production-mode, all samples are preprocessed, and in sample-mode only
                the first 300 samples are preprocessed.
        """
        super().__init__()
        self.preprocessor_name = preprocessor_type
        self.preprocessor_full_name = f"{dataset.name}_{dataset_split.name}_{preprocessor_type}_{run_mode.name}"
        self.data_folder_path = f"./preprocessed_data"
        self.visualization_file_path = f"{self.data_folder_path}/{self.preprocessor_full_name}_visualization.bdocjs"
        self.samples_file_path = f"{self.data_folder_path}/{self.preprocessor_full_name}_samples.json"
        self.entity_types_file_path = f"{self.data_folder_path}/{self.preprocessor_full_name}_types.txt"
        self.samples: list[Sample] | None = None
        self.annotators = annotators
        self.dataset_split = dataset_split
        self.preprocessor_type = preprocessor_type
        self.dataset = dataset
        self.run_mode = run_mode
        self.print_info()

    def print_info(self):
        print("\n\n------ INFO --------")
        print(blue("Preprocessor Name:"), green(self.preprocessor_full_name))
        print(blue("Run Mode:"), green(self.run_mode.name))
        print(blue("Dataset:"), green(self.dataset.name))
        print("--------INFO----------\n\n")

    def run_annotation_pipeline(self):
        """
        All samples are annotated by the given annotators in a
        defined sequence (some annotator depend on others).
        """
        assert self.samples is not None
        for annotator in self.annotators:
            self.samples = annotator.annotate(self.samples, self.dataset_split)

    def get_samples_cached(self) -> list[Sample]:
        """
        We cache `Samples` after extracting them from raw data.
        """
        if self.samples is None:
            print(red("Creating Cache of Samples"))
            self.samples = self.get_samples()
            if self.run_mode == PreprocessorRunType.dry_run:
                self.samples = self.samples[:300]
            self.run_annotation_pipeline()
            if self.run_mode == PreprocessorRunType.dry_run:
                self.samples = self.samples[:300]
        else:
            print(green("using cache"))
        return self.samples

    @abstractmethod
    def get_samples(self) -> list[Sample]:
        """
        Extract samples from the given raw data file provided
        by the organizers.
        """

    @abstractmethod
    def get_entity_types(self) -> list[str]:
        """
        Returns:
            The list of all entity types (represented as unique strings).
        """

    def create_entity_types_file(self):
        """
        Creates a file that lists the entity types.
        """
        all_entity_types = self.get_entity_types()
        all_types_set = set(all_entity_types)
        assert len(all_types_set) == len(all_entity_types)  # no duplicates allowed
        with open_make_dirs(self.entity_types_file_path, 'w') as types_file:
            for type_name in all_types_set:
                print(type_name, file=types_file)

    def create_visualization_file(self) -> None:
        """
        Create a .bdocjs formatted file which can be directly imported 
        into gate developer using the gate bdocjs plugin. 
        """
        samples = self.get_samples_cached()
        sample_to_annos = {}
        sample_to_text = {}
        for sample in samples:
            gold_and_external_annos = sample.annos.gold + sample.annos.external
            sample_to_annos[sample.id] = gold_and_external_annos
            sample_to_text[sample.id] = sample.text

        create_visualization_file(
            self.visualization_file_path,
            sample_to_annos,
            sample_to_text
        )

    def store_samples(self) -> None:
        """
        Persist the samples on disk.
        """
        samples = self.get_samples_cached()
        write_samples(samples, self.samples_file_path)

    def run(self) -> None:
        """
        Execute the preprocessing steps that generate files which
        can be used to train models.
        """
        print_section()
        print(green(f"Preprocessing {self.preprocessor_full_name}"))
        print("Creating data folder")
        create_directory_structure(self.data_folder_path)
        print("Creating entity file... ")
        self.create_entity_types_file()
        print("Creating visualization file...")
        self.create_visualization_file()
        print("Creating samples json file")
        self.store_samples()
        print("Done Preprocessing!")




def create_visualization_file(
        visualization_file_path: str,
        sample_to_annos: dict[SampleId, list[Annotation]],
        sample_to_text: dict[SampleId, str]
) -> None:
    """
    Create a .bdocjs formatted file which can me directly imported into 
    Gate Developer for vizualisation.
    We create the file using the given text and annotations.

    Args:
        visualization_file_path: str
            the path of the visualization file we want to create
        annos_dict: Dict[str, List[Anno]]
            mapping from sample ids to annotations
        sample_to_text:
            mapping from sample ids to text
    """
    assert visualization_file_path.endswith(".bdocjs")
    sample_offset = 0
    document_text = ""
    ofsetted_annos = []
    for sample_id in sample_to_annos:
        document_text += (sample_to_text[sample_id] + '\n\n\n\n')
        ofsetted_annos.append(Annotation(sample_offset, len(
            document_text), 'Sample', '', {"id": sample_id}))
        for anno in sample_to_annos[sample_id]:
            new_start_offset = anno.begin_offset + sample_offset
            new_end_offset = anno.end_offset + sample_offset
            gate_features = anno.features.copy()
            gate_features['orig_start_offset'] = anno.begin_offset
            gate_features['orig_end_offset'] = anno.end_offset
            ofsetted_annos.append(Annotation(
                new_start_offset, new_end_offset, anno.label_type, anno.extraction, gate_features))
        sample_offset += (len(sample_to_text[sample_id]) + 4)  # account for added new lines
    gate_document = Document(document_text)
    default_ann_set = gate_document.annset()
    for ofsetted_annotation in ofsetted_annos:
        default_ann_set.add(
            int(ofsetted_annotation.begin_offset),
            int(ofsetted_annotation.end_offset),
            ofsetted_annotation.label_type,
            ofsetted_annotation.features)
    gate_document.save(visualization_file_path)



def write_samples(samples: list[Sample], output_json_file_path: str):
    with open(output_json_file_path, 'w') as output_file:
        json.dump(samples, output_file, default=vars)
