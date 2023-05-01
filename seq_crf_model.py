from models.all_models import SeqLabelerNoTokenization
from utils.config import DatasetConfig, ModelConfig
from typing import List
from flair.models.sequence_tagger_utils.crf import CRF
from flair.models.sequence_tagger_utils.viterbi import ViterbiLoss, ViterbiDecoder
from flair.data import Dictionary
import torch.nn as nn
import torch
import train_util
from preamble import *
import util
from structs import Annotation, Sample
from utils.model import SeqLabelPredictions


class SeqLabelerDefaultCRF(SeqLabelerNoTokenization):
    def __init__(self, all_types: List[str], model_config: ModelConfig, dataset_config: DatasetConfig): 
        super().__init__(all_types=all_types, model_config=model_config, dataset_config=dataset_config)
        self.flair_dictionary = self.get_flair_label_dictionary()
        self.loss_function = ViterbiLoss(self.flair_dictionary)
        assert len(self.idx_to_label) + 2 == len(self.flair_dictionary)
        self.crf = CRF(self.flair_dictionary, len(self.flair_dictionary), False)
        self.viterbi_decoder = ViterbiDecoder(self.flair_dictionary)
        self.linear = nn.Linear(self.input_dim, len(self.flair_dictionary))

    def get_flair_label_dictionary(self):
        flair_dictionary = Dictionary(add_unk=False)
        for i in range(len(self.idx_to_label)):
            assert flair_dictionary.add_item(str(self.idx_to_label[i])) == i
        flair_dictionary.set_start_stop_tags()
        return flair_dictionary

    def get_predicted_labels(self, predictions_batch):
        predicted_label_strings = [
            [label for label, score in predictions]
            for predictions in predictions_batch
        ]
        predicted_label_indices = [
            [self.flair_dictionary.get_idx_for_item(label_string) for label_string in predicted_label_strings_sample]
            for predicted_label_strings_sample in predicted_label_strings
        ]
        predicted_labels = [
            [self.idx_to_label[label_idx] for label_idx in predicted_label_indices_sample]
            for predicted_label_indices_sample in predicted_label_indices
        ]
        return predicted_labels

    def forward(
            self,
            samples: List[Sample],
    ) -> tuple[torch.Tensor, SeqLabelPredictions]:
        assert isinstance(samples, list)
        # encoding helps manage tokens created by bert
        bert_encoding_for_batch = self.get_bert_encoding_for_batch(samples, self.model_config)
        # print("encoding new", bert_encoding_for_batch)
        # SHAPE (batch_size, seq_len, bert_emb_len)
        bert_embeddings_batch = self.get_bert_embeddings_for_batch(bert_encoding_for_batch, samples=samples)
        # SHAPE (batch_size, seq_len, num_types)
        linear_logits = self.linear(bert_embeddings_batch)
        crf_logits = self.crf(linear_logits)

        gold_labels_batch = train_util.get_bio_labels_for_bert_tokens_batch(
            self.get_token_annos_batch(bert_encoding_for_batch, samples),
            [sample.annos.gold for sample in samples]
        )

        lengths = [len(gold_labels) for gold_labels in gold_labels_batch]  
        # SHAPE (batch_size, num_tokens)
        lengths = torch.tensor(lengths, dtype=torch.long)

        features_tuple = (crf_logits, lengths, self.crf.transitions)

        assert len(gold_labels_batch) == len(samples)  # labels for each sample in batch
        assert len(gold_labels_batch[0]) == bert_embeddings_batch.shape[1]  # same num labels as tokens

        gold_label_indices = []
        for gold_labels_sample in gold_labels_batch:
            for label in gold_labels_sample:
                gold_label_indices.append(self.label_to_idx[label])

        # SHAPE (batch*seq_len)
        gold_label_indices = torch.tensor(gold_label_indices).to(device)

        assert len(gold_label_indices.shape) == 1, "needs to be a tensor with one dimension"

        loss = self.loss_function(features_tuple, gold_label_indices)

        predictions, all_tags = self.viterbi_decoder.decode(features_tuple, False, None)

        predicted_labels_batch = self.get_predicted_labels(predictions)

        predicted_annos_batch: List[List[Annotation]] = [
            util.get_annos_from_bio_labels(
                prediction_labels=predicted_labels,
                batch_encoding=bert_encoding_for_batch,
                batch_idx=batch_idx,
                sample_text=samples[batch_idx].text
            )
            for batch_idx, predicted_labels in enumerate(predicted_labels_batch)
        ]
        return loss, predicted_annos_batch

