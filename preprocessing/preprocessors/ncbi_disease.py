from utils.preprocessing import Preprocessor, PreprocessorRunType, Annotator
from utils.structs import Dataset, Sample, AnnotationCollection, Annotation, DatasetSplit
from utils.universal import red

def get_ncbi_sample(sample_raw: list) -> Sample:
    title_sample_id, title_tag, title_text = sample_raw[0].split('|')
    assert title_tag == 't'
    assert len(title_text)

    abstract_sample_id, abstract_tag, abstract_text = sample_raw[1].split('|')
    assert abstract_tag == 'a'
    assert len(abstract_text)
    assert abstract_sample_id == title_sample_id 

    disease_spans = []
    for anno_row in sample_raw[2:]:
        anno_columns = anno_row.split('\t')
        assert len(anno_columns) == 6
        span_start = int(anno_columns[1])
        span_end = int(anno_columns[2])
        extraction = anno_columns[3]
        disease_spans.append((span_start, span_end, extraction))

    full_text = title_text + ' ' + abstract_text
    for start, end, gold_extraction in disease_spans:
        if full_text[start:end] != gold_extraction:
            print("Annotation mismatch")
            print("text", full_text[start:end])
            print("gold", gold_extraction)
            print()


    disease_annos = [
        Annotation(
            begin_offset=start,
            end_offset=end,
            extraction=gold_extraction,
            label_type='Disease'
        )
        for start, end, gold_extraction in disease_spans
    ]

    return Sample(
        id=title_sample_id,
        text=full_text,
        annos=AnnotationCollection(gold=disease_annos, external=[])
    )

def get_ncbi_raw_samples(corpus_file_path) -> list[list]:
    samples = []
    curr_sample = []
    with open(corpus_file_path, 'r') as ncbi_file:
        for line in ncbi_file:
            line = line.strip()
            if not len(line):
                samples.append(curr_sample)
                curr_sample = []
            else:
                curr_sample.append(line)
    assert len(curr_sample)
    samples.append(curr_sample)
    print(f"found {len(samples)} samples")
    non_empty_samples = [sample for sample in samples if len(sample)]
    print(red(f"empty samples: {len(samples) - len(non_empty_samples)}"))
    return non_empty_samples


class PreprocessNcbiDisease(Preprocessor):
    def __init__(
            self,
            preprocessor_type: str,
            dataset_split: DatasetSplit,
            annotators: list[Annotator],
            run_mode: PreprocessorRunType
    ) -> None:
        super().__init__(
            preprocessor_type=preprocessor_type,
            dataset=Dataset.ncbi_disease,
            annotators=annotators,
            dataset_split=dataset_split,
            run_mode=run_mode,
        )

    def get_samples(self) -> list[Sample]:
        match self.dataset_split:
            case DatasetSplit.train:
                corpus_file_path = './NCBItrainset_corpus.txt'
            case DatasetSplit.valid:
                corpus_file_path = './NCBIdevelopset_corpus.txt'
            case DatasetSplit.test:
                corpus_file_path = './NCBItestset_corpus.txt'
        raw_samples = get_ncbi_raw_samples(corpus_file_path=corpus_file_path)
        samples = [get_ncbi_sample(raw_sample) for raw_sample in raw_samples]
        return samples

    def get_entity_types(self) -> list[str]:
        return ['Disease']
