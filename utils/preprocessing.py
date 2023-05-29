from structs import *
import json
from abc import ABC, abstractmethod
from utils.structs import PreprocessorConfig
from utils.universal import blue, green, red, \
        open_make_dirs, \
        create_directory_structure, print_section,\
        contained_in

from tqdm import tqdm as show_progress
from pydoc import locate
from gatenlp import Document
import spacy
from spacy.tokens.span import Span as SpacySpan


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


def get_sentence_sample(sentence: SpacySpan, sentence_idx: int, sample: Sample) -> Sample:
    gold_annos = sample.annos.gold
    gold_annos_accross_boundary = [
        anno 
        for anno in gold_annos
        if (not contained_in(
                    outside=(sentence.start_char, sentence.end_char), 
                    inside=(anno.begin_offset, anno.end_offset))
            )
            and
            (
                (sentence.start_char <= anno.begin_offset <= sentence.end_char)
                or
                (sentence.start_char <= anno.end_offset <= sentence.end_char)
            )  
    ]
    if len(gold_annos_accross_boundary):
        print(red(f"WARN : Gold Annos accross sentence boundary \n annos: {gold_annos_accross_boundary} \n sample: {sample}"))

    gold_annos_in_sentence = [
        anno 
        for anno in gold_annos
        if contained_in(
            outside=(sentence.start_char, sentence.end_char), 
            inside=(anno.begin_offset, anno.end_offset)
        )
    ]

    annos_with_corrected_offsets = [
        Annotation(
            begin_offset=(anno.begin_offset - sentence.start_char),
            end_offset=(anno.end_offset - sentence.start_char),
            label_type=anno.label_type,
            extraction=anno.extraction
            )
        for anno in gold_annos_in_sentence
    ]

    for anno in annos_with_corrected_offsets:
        if sentence.text[anno.begin_offset:anno.end_offset] != anno.extraction:
            print(f"WARN: anno sentence text does not match extraction :\n"
                  f"anno text: {sentence.text[anno.begin_offset: anno.end_offset]}\n"
                  f"extraction: {anno.extraction}\n"
                  f"sample: {sample}")

    return Sample(
        text=sentence.text,
        id=sample.id + str(sentence_idx),
        annos=AnnotationCollection(gold=annos_with_corrected_offsets, external=[])
    ) 


class SentenceAnnotator(Annotator):
    """
    Break up a each sample into smaller samples,
    with each smaller sample corresponding to a sentence.
    Sentences are identified using Spacy.
    """
    def __init__(self) -> None:
        super().__init__("SentenceAnnotator")
        self.nlp = spacy.load('en_core_web_md')

    def annotate_helper(self, samples: list[Sample], dataset_split: DatasetSplit) -> list[Sample]:
        sentence_samples = []
        for sample in show_progress(samples):
            spacy_doc = self.nlp(sample.text)
            for sent_idx, spacy_sentence in enumerate(spacy_doc.sents):
                sentence_samples.append(
                    get_sentence_sample(
                        sentence=spacy_sentence,
                        sentence_idx=sent_idx,
                        sample=sample
                    )
                )
        return sentence_samples


def get_sentence_annotator():
    return SentenceAnnotator()



def get_preprocessor_class_from_path(preprocessor_class_path: str) -> type:
    return locate(preprocessor_class_path)


def preprocess(
        preprocessor_config: PreprocessorConfig,
        run_mode: PreprocessorRunType,
        dataset_splits: list[DatasetSplit]
):
    preprocessor_class = get_preprocessor_class_from_path(preprocessor_config.preprocessor_class_path)
    assert preprocessor_class is not None, f"Could not get preprocessor class {preprocessor_config.preprocessor_class_path}"
    for split in dataset_splits:
        preprocessor_params = preprocessor_config.preprocessor_class_init_params.copy()
        preprocessor_params['dataset_split'] = split
        preprocessor_params['run_mode'] = run_mode
        preprocessor = preprocessor_class(**preprocessor_params)
        preprocessor.run()
