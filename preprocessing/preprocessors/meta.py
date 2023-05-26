from collections import defaultdict
from overrides import overrides
from utils.preprocessing import Preprocessor, Annotator
from utils.structs import Annotation, Sample, Dataset, \
        DatasetSplit, PreprocessorRunType, AnnotationCollection, SampleAnnotation
from random import shuffle
from utils.universal import read_predictions_file
from utils.training import get_test_samples_by_dataset_name, get_valid_samples_by_dataset_name, get_train_samples_by_dataset_name
from glob import glob

class MetaPreprocessor(Preprocessor):
    """This preprocessor uses the systems' predictions 
    on the validation set to prepare input data for Meta.
    """
    def __init__(
        self,
        preprocessor_type: str,
        dataset_split: DatasetSplit,
        annotators: list[Annotator],
        run_mode: PreprocessorRunType,
        test_files_folder_full_path: str,
        valid_files_folder_full_path: str,
        dataset_config_name: str,
        dataset: Dataset
    ) -> None:
        super().__init__(
            dataset_split=dataset_split,
            preprocessor_type=preprocessor_type,
            dataset=dataset,
            annotators=annotators,
            run_mode=run_mode
        )
        self.test_prediction_file_paths = glob(f"{test_files_folder_full_path}/*.tsv")
        self.valid_prediction_file_paths = glob(f"{valid_files_folder_full_path}/*.tsv")
        self.dataset_config_name = dataset_config_name
        assert len(self.test_prediction_file_paths) == 2
        assert len(self.valid_prediction_file_paths) == 40


    def create_meta_sample(self, sample: Sample, span: SampleAnnotation, label_type: str):
        text = sample.text
        
        text_before_entity = text[:span.begin_offset]
        text_after_entity = text[span.end_offset:]
        entity_text = text[span.begin_offset: span.end_offset]
        sample_text_with_special_tokens = span.type_string + ' ' + text_before_entity + ' <e> ' + entity_text + ' </e> ' + text_after_entity

        return Sample(
                text= sample_text_with_special_tokens,
                id=f"{sample.id}@@@{span.begin_offset}@@@{span.end_offset}@@@{span.type_string}",
                annos=AnnotationCollection(
                    gold=[Annotation(
                        begin_offset=0,
                        end_offset=len(sample.text),
                        label_type=label_type,
                        extraction=entity_text
                    )],
                    external=[]
                )
            )


    def get_test_set_for_meta(self):
        all_predictions_dict = defaultdict(list)
        assert len(self.test_prediction_file_paths) == 2
        for prediction_file_path in self.test_prediction_file_paths:
            predictions = read_predictions_file(prediction_file_path)
            for sample_id, annos in predictions.items():
                all_predictions_dict[sample_id].extend(annos)
        samples = get_test_samples_by_dataset_name(self.dataset_config_name)

        gold_samples = {sample.id: sample for sample in samples}
        for sample_id in all_predictions_dict:
            assert sample_id in gold_samples

        samples: list[Sample] = []
        for sample_id in gold_samples:
            sample = gold_samples[sample_id]
            all_prediction_spans = set()
            if sample_id in all_predictions_dict:
                all_prediction_spans = set([SampleAnnotation(
                                                begin_offset=anno.begin_offset,
                                                end_offset=anno.end_offset,
                                                sample_id=sample_id,
                                                type_string=anno.label_type
                                            ) 
                                            for anno in all_predictions_dict[sample_id]])
            for prediction_span in all_prediction_spans:
                samples.append(
                    self.create_meta_sample(
                        sample=sample,
                        span=prediction_span,
                        label_type='correct')
                )
        shuffle(samples)
        return samples


    def get_training_and_valid_set_for_meta(self):
        all_predictions_dict = defaultdict(list)
        for prediction_file_path in self.valid_prediction_file_paths:
            predictions = read_predictions_file(prediction_file_path)
            for sample_id, annos in predictions.items():
                all_predictions_dict[sample_id].extend(annos)

        valid_samples = get_valid_samples_by_dataset_name(self.dataset_config_name)
        original_train_samples = get_train_samples_by_dataset_name(self.dataset_config_name)

        gold_samples = valid_samples + original_train_samples
        print("num correct samples", len(gold_samples))
        gold_samples = {sample.id: sample for sample in gold_samples}
        for sample_id in all_predictions_dict:
            assert sample_id in gold_samples
        
        num_incorrect = 0
        num_correct = 0

        meta_samples: list[Sample] = []
        for sample_id in gold_samples:
            sample = gold_samples[sample_id]
            gold_spans = set([
                                SampleAnnotation(
                                    begin_offset=anno.begin_offset,
                                    end_offset=anno.end_offset,
                                    sample_id=sample_id,
                                    type_string=anno.label_type
                                ) 
                                for anno in sample.annos.gold])
            prediction_spans: set[SampleAnnotation] = set()
            if sample_id in all_predictions_dict:
                prediction_spans = set([SampleAnnotation(
                                            begin_offset=anno.begin_offset,
                                            end_offset=anno.end_offset,
                                            sample_id=sample_id,
                                            type_string=anno.label_type
                                        )
                                        for anno in all_predictions_dict[sample_id]])

            incorrect_prediction_spans = prediction_spans.difference(gold_spans)
            correct_prediction_spans = gold_spans
            num_incorrect += len(incorrect_prediction_spans)
            num_correct += len(correct_prediction_spans)

            assert len(incorrect_prediction_spans.intersection(correct_prediction_spans)) ==  0
            for correct_span in correct_prediction_spans:
                meta_samples.append(
                        self.create_meta_sample(sample=sample, span=correct_span, label_type='correct')
                )
            for incorrect_span in incorrect_prediction_spans:
                meta_samples.append(
                        self.create_meta_sample(sample=sample, span=incorrect_span, label_type='incorrect')
                )

        shuffle(meta_samples)
        percent_85 = int(len(meta_samples)*0.85)
        train_samples = meta_samples[:percent_85]
        valid_samples = meta_samples[percent_85:]
        assert len(train_samples) + len(valid_samples) == len(meta_samples)

        print("correct", num_correct)
        print("incorrect", num_incorrect)
        print("ratio", num_incorrect/(num_correct + num_incorrect))


        return train_samples, valid_samples


    @overrides
    def get_samples(self) -> list[Sample]:
        test = self.get_test_set_for_meta()
        print("test size", len(test))
        train, valid = self.get_training_and_valid_set_for_meta()
        print("train size", len(train))
        print("valid size", len(valid))
        match self.dataset_split:
            case DatasetSplit.train:
                samples = train
            case DatasetSplit.valid:
                samples = valid
            case DatasetSplit.test:
                samples = test
        return samples

    @overrides
    def get_entity_types(self) -> list[str]:
        return ['correct', 'incorrect']
