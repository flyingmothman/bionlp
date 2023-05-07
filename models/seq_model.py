import torch
from torch import nn
from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding, CharSpan
from utils.model import ModelBase
from utils.structs import DatasetConfig, ModelConfig, Sample, Annotation
from utils.universal import Option, device
from utils.training import get_bio_label_idx_dicts, get_bio_labels_for_bert_tokens_batch, get_annos_from_bio_labels
from utils.model import SeqLabelPredictions

class SeqDefault(ModelBase):
    def __init__(self, all_types: list[str], model_config: ModelConfig, dataset_config: DatasetConfig):
        super().__init__(model_config=model_config, dataset_config=dataset_config)
        self.bert_model = AutoModel.from_pretrained(model_config.pretrained_model_name)
        self.bert_tokenizer = AutoTokenizer.from_pretrained(model_config.pretrained_model_name)
        self.input_dim = model_config.pretrained_model_output_dim
        self.num_class = (dataset_config.num_types * 2) + 1
        self.classifier = nn.Linear(self.input_dim, self.num_class)
        label_to_idx, idx_to_label = get_bio_label_idx_dicts(all_types, dataset_config)
        self.label_to_idx = label_to_idx
        self.idx_to_label = idx_to_label
        self.loss_function = nn.CrossEntropyLoss()


    def get_bert_encoding_for_batch(self, samples: list[Sample],
                                    model_config: ModelConfig) -> BatchEncoding:
        batch_of_sample_texts = [sample.text for sample in samples]
        bert_encoding_for_batch = self.bert_tokenizer(batch_of_sample_texts,
                                                      return_tensors="pt",
                                                      is_split_into_words=False,
                                                      add_special_tokens=model_config.use_special_bert_tokens,
                                                      truncation=True,
                                                      padding=True,
                                                      max_length=512).to(device)
        return bert_encoding_for_batch


    def get_bert_embeddings_for_batch(self, encoding: BatchEncoding, samples: list[Sample]):
        bert_embeddings_batch = self.bert_model(encoding['input_ids'], return_dict=True)
        # SHAPE: (batch, seq, emb_dim)
        bert_embeddings_batch = bert_embeddings_batch['last_hidden_state']
        return bert_embeddings_batch


    def check_if_tokens_overlap(self, token_annos: list[Option[Annotation]], sample_id: str):
        for idx_curr, curr_anno in enumerate(token_annos):
            for idx_other, other_anno in enumerate(token_annos):
                if (idx_curr != idx_other) and (curr_anno.is_something() and other_anno.is_something()):
                    no_overlap =  (curr_anno.get_value().end_offset <= other_anno.get_value().begin_offset) \
                        or (other_anno.get_value().end_offset <= curr_anno.get_value().begin_offset)
                    if not no_overlap:
                        assert ((curr_anno.get_value().end_offset - curr_anno.get_value().begin_offset) == 1) \
                                or ((other_anno.get_value().end_offset - other_anno.get_value().begin_offset) == 1) \
                                , f"one of the annos needs to be the roberta space character {curr_anno}, {other_anno}"
                        raise RuntimeError(f"token annos should never overlap"
                                           f"\n annos: {(curr_anno.get_value(), other_anno.get_value())}"
                                           f"\n sampleId: {sample_id}"
                                           )
    

    def remove_roberta_overlaps(self, tokens_batch: list[list[Option[Annotation]]], model_config: ModelConfig) \
        -> list[list[Option[Annotation]]]:
        if 'roberta' in model_config.pretrained_model_name:
            tokens_batch_without_overlaps = []
            for tokens in tokens_batch:
                tokens_without_overlap = []
                for curr_token_idx in range(len(tokens) - 1):
                    curr_token = tokens[curr_token_idx]
                    next_token = tokens[curr_token_idx + 1]
                    if (curr_token.is_something() and next_token.is_something()) and \
                       (curr_token.get_value().begin_offset == next_token.get_value().begin_offset):
                        assert (curr_token.get_value().end_offset - curr_token.get_value().begin_offset) == 1
                        tokens_without_overlap.append(Option(None))
                    else:
                        tokens_without_overlap.append(curr_token)
                tokens_without_overlap.append(tokens[len(tokens) - 1])
                assert len(tokens_without_overlap) == len(tokens)
                tokens_batch_without_overlaps.append(tokens_without_overlap)
            return tokens_batch_without_overlaps
        else:
            return tokens_batch




    def get_token_annos_batch(self, bert_encoding, samples: list[Sample]) -> list[list[Option[Annotation]]]:
        expected_batch_size = len(samples)
        token_ids_matrix = bert_encoding['input_ids']
        batch_size = len(token_ids_matrix)
        num_tokens = len(token_ids_matrix[0])
        for batch_idx in range(batch_size):
            assert len(token_ids_matrix[batch_idx]) == num_tokens, "every sample should have the same number of tokens"
        assert batch_size == expected_batch_size
        token_annos_batch: list[list[Option[Annotation]]] = []
        for batch_idx in range(batch_size):
            char_spans: list[Option[CharSpan]] = [
                Option(bert_encoding.token_to_chars(batch_or_token_index=batch_idx, token_index=token_idx))
                for token_idx in range(num_tokens)
            ]

            token_annos_batch.append(
                [
                    Option(Annotation(begin_offset=span.get_value().start, end_offset=span.get_value().end,
                                label_type='BertTokenAnno', extraction=None))
                    if span.state == OptionState.Something else Option(None)
                    for span in char_spans
                ]
            )

        return token_annos_batch

    def forward(
            self,
            samples: list[Sample],
    ) -> tuple[torch.Tensor, SeqLabelPredictions]:
        assert isinstance(samples, list)
        # encoding helps manage tokens created by bert
        bert_encoding_for_batch = self.get_bert_encoding_for_batch(samples, self.model_config)
        # print("encoding new", bert_encoding_for_batch)
        # collect.append(bert_encoding_for_batch)
        # SHAPE (batch_size, seq_len, bert_emb_len)
        bert_embeddings_batch = self.get_bert_embeddings_for_batch(bert_encoding_for_batch, samples=samples)
        predictions_logits_batch = self.classifier(bert_embeddings_batch)

        gold_labels_batch = get_bio_labels_for_bert_tokens_batch(
            self.get_token_annos_batch(bert_encoding_for_batch, samples),
            [sample.annos.gold for sample in samples]
        )
        assert len(gold_labels_batch) == len(samples)  # labels for each sample in batch
        assert len(gold_labels_batch[0]) == bert_embeddings_batch.shape[1]  # same num labels as tokens
        #  print("gold bio labels", gold_labels_batch[0])
        #  collect.append(gold_labels_batch[0])

        gold_label_indices = [
            [self.label_to_idx[label] for label in gold_labels]
            for gold_labels in gold_labels_batch
        ]
        gold_label_indices = torch.tensor(gold_label_indices).to(device)

        loss = self.loss_function(
            torch.permute(predictions_logits_batch, (0, 2, 1)),
            gold_label_indices
        )

        predicted_label_indices_batch = torch.argmax(predictions_logits_batch, dim=2).cpu().detach().numpy()
        predicted_labels_batch = [
            [self.idx_to_label[label_id] for label_id in predicted_label_indices]
            for predicted_label_indices in predicted_label_indices_batch
        ]
        predicted_annos_batch: list[list[Annotation]] = [
            get_annos_from_bio_labels(
                prediction_labels=predicted_labels,
                batch_encoding=bert_encoding_for_batch,
                batch_idx=batch_idx,
                sample_text=samples[batch_idx].text
            )
            for batch_idx, predicted_labels in enumerate(predicted_labels_batch)
        ]
        return loss, predicted_annos_batch
