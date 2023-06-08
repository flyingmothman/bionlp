from utils.structs import PreprocessorRunType
from utils.preprocessing import Preprocessor
from utils.structs import Annotation, Sample, DatasetSplit, Dataset, AnnotationCollection
from utils.preprocessing import Annotator
from bs4 import BeautifulSoup


def get_first_sample_soup():
    with open('GENIA_term_3.02/GENIAcorpus3.02.xml', 'r') as genia_file:
        soup = BeautifulSoup(genia_file, 'xml')
        return list(soup.find_all('sentence'))[0]


def get_text_from_tag():
    first_sample_soup = get_first_sample_soup()
    print(first_sample_soup)


def get_text(tag):
    ret = ''
    for child in tag.contents:
        if child.name == 'cons':
            ret += get_text(child)
        elif child.string is not None:
            ret += child.string
        else:
            NotImplementedError(f'cannot handle tag {child.name}')
    return ret


def get_annos(tag, offset=None, token_strings=None) -> list[Annotation]:
    if offset is None:
        offset = [0]
        assert token_strings is None
        token_strings = []
    ret = []
    anno_start_offset = offset[0]
    token_begin_idx = len(token_strings)
    for child in tag.contents:
        if child.name == 'cons':
            ret.extend(get_annos(child, offset, token_strings))
        else:
            offset[0] += len(child.string)
            token_strings.append(child.string)
    anno_end_offset = offset[0]
    if (tag.name == 'cons') and ('sem' in tag.attrs) and (tag['sem'][0:2] == 'G#'):
        ret.append(
            Annotation(
                anno_start_offset,
                anno_end_offset,
                tag['sem'],
                ''.join(token_strings[token_begin_idx:]),
                {}))
    return ret


def get_parent_label_from_anno(anno: Annotation) -> str | None:
    anno_label_type = anno.label_type.lower()
    if anno_label_type.startswith('G#DNA'.lower()):
        return 'dna'
    elif anno_label_type.startswith('G#protein'.lower()):
        return 'protein'
    elif anno_label_type.startswith('G#RNA'.lower()):
        return 'rna'
    elif anno_label_type.startswith('G#cell_line'.lower()):
        return 'cell_line'
    elif anno_label_type.startswith('G#cell_type'.lower()):
        return 'cell_type'
    else:
        return None


def get_parent_annos(anno_list: list[Annotation]) -> list[Annotation]:
    ret = []
    for anno in anno_list:
        if get_parent_label_from_anno(anno) is not None:
            anno.label_type = get_parent_label_from_anno(anno)
            ret.append(anno)
    return ret


def get_split_range(split: DatasetSplit) -> tuple:
    if split == DatasetSplit.train:
        return 0, 9273
    if split == DatasetSplit.valid:
        return 9273, 13909
    if split == DatasetSplit.test:
        return 13909, 18546
    raise Exception("should not reach here")


def get_sentence_samples_from_article(
        article_soup: BeautifulSoup
    ) -> list[Sample]:
    article_id = article_soup.find('bibliomisc').string
    assert article_id is not None
    ret = []
    for sent_tag in article_soup.find_all('sentence'):
        sample_text = get_text(sent_tag)
        sample_annos = get_parent_annos(get_annos(sent_tag))
        ret.append(
            Sample(
                text=sample_text,
                id=article_id,
                annos=AnnotationCollection(sample_annos, [])
            )
        )
    return ret

def make_sample_from_article(article_soup: BeautifulSoup) -> Sample:
    article_id = article_soup.find('bibliomisc').string
    assert article_id is not None
    sentence_samples = get_sentence_samples_from_article(article_soup)
    assert len(sentence_samples)
    sentence_offset = 0
    article_sample_text = ''
    article_gold_annos = []
    for sentence_sample in sentence_samples:
        article_sample_text = article_sample_text + sentence_sample.text + "  "
        adjusted_annos = [
            Annotation(
                begin_offset=anno.begin_offset + sentence_offset,
                end_offset=anno.end_offset + sentence_offset,
                label_type=anno.label_type,
                extraction=anno.extraction
            )
            for anno in sentence_sample.annos.gold
        ]
        article_gold_annos.extend(adjusted_annos)
        sentence_offset = len(article_sample_text)
    return Sample(
            text=article_sample_text,
            id=article_id,
            annos=AnnotationCollection(gold=article_gold_annos, external=[])
    )

def get_split_range_article(split: DatasetSplit) -> tuple[int,int]:
    if split == DatasetSplit.train:
        return 0, 1000
    if split == DatasetSplit.valid:
        return 1000, 1500
    if split == DatasetSplit.test:
        return 1500, 2000
    raise Exception("should not reach here")

def get_samples_article_level(split: DatasetSplit) -> list[Sample]:
    ret: list[Sample] = []
    split_range = get_split_range_article(split)
    with open('GENIA_term_3.02/GENIAcorpus3.02.xml', 'r') as genia_file:
        genia_soup = BeautifulSoup(genia_file, 'xml')
        all_article_soups = list(genia_soup.find_all('article'))
        assert len(all_article_soups) == 2000
        all_article_soups = all_article_soups[split_range[0]: split_range[1]]
        for article_soup in all_article_soups:
            ret.append(make_sample_from_article(article_soup))
    return ret

def get_samples(split: DatasetSplit) -> list[Sample]:
    split_range = get_split_range(split)
    with open('GENIA_term_3.02/GENIAcorpus3.02.xml', 'r') as genia_file:
        soup = BeautifulSoup(genia_file, 'xml')
        ret = []
        for sample_id, sent_tag in enumerate(soup.find_all('sentence')):
            if split_range[0] <= sample_id < split_range[1]:
                sample_text = get_text(sent_tag)
                sample_annos = get_parent_annos(get_annos(sent_tag))
                ret.append(
                    Sample(
                        text=sample_text,
                        id=str(sample_id),
                        annos=AnnotationCollection(sample_annos, [])
                    )
                )
        return ret


class PreprocessGenia(Preprocessor):
    """
    A preprocessor for the Genia dataset.
    """

    def __init__(
            self,
            preprocessor_type: str,
            dataset_split: DatasetSplit,
            annotators: list[Annotator],
            run_mode: PreprocessorRunType
    ) -> None:
        super().__init__(
            preprocessor_type=preprocessor_type,
            dataset=Dataset.genia,
            annotators=annotators,
            dataset_split=dataset_split,
            run_mode=run_mode,
        )

    def get_samples(self) -> list[Sample]:
        return get_samples(self.dataset_split)

    def get_entity_types(self) -> list[str]:
        all_types_set = set()
        for sample in self.get_samples():
            all_types_set.update([anno.label_type for anno in sample.annos.gold])
        assert len(all_types_set) == 5
        return list(all_types_set)


class PreprocessGeniaArticleLevel(PreprocessGenia):
    """
    Every sample is one entire annotated article.
    """

    def __init__(
        self,
        preprocessor_type: str,
        dataset_split: DatasetSplit,
        annotators: list[Annotator],
        run_mode: PreprocessorRunType
    ) -> None:
        super().__init__(
            preprocessor_type=preprocessor_type,
            dataset_split=dataset_split,
            annotators=annotators,
            run_mode=run_mode
        )


    def get_samples(self) -> list[Sample]:
        return get_samples_article_level(self.dataset_split)

