import torch
from allennlp.modules.span_extractors.endpoint_span_extractor import EndpointSpanExtractor
from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.auto.tokenization_auto import AutoTokenizer
from utils.structs import Annotation, Sample, ModelConfig, DatasetConfig
from utils.universal import device
from utils.model import ModelBase, SeqLabelPredictions 
from utils.training import enumerate_spans
from transformers.tokenization_utils_base import BatchEncoding
import torch.nn as nn

class SpanDefault(ModelBase):
    def __init__(self, all_types: list[str], model_config: ModelConfig, dataset_config: DatasetConfig):
        super(SpanDefault, self).__init__(model_config, dataset_config)
        self.bert_model = AutoModel.from_pretrained(model_config.pretrained_model_name)
        self.bert_tokenizer = AutoTokenizer.from_pretrained(model_config.pretrained_model_name)
        self.input_dim = model_config.pretrained_model_output_dim
        self.num_class = len(all_types) + 1
        print("types new\n:", sorted(all_types))
        self.classifier = self.get_classifier()
        self.endpoint_span_extractor = self.get_endpoint_span_extractor()
        self.loss_function = nn.CrossEntropyLoss()
        self.type_to_idx = {type_name: i for i, type_name in enumerate(all_types)}
        # Add NO_TYPE type which represents "no annotation"
        self.type_to_idx['NO_TYPE'] = len(self.type_to_idx)
        self.idx_to_type = {i: type_name for type_name, i in self.type_to_idx.items()}
        assert len(self.type_to_idx) == self.num_class, "Num of classes should be equal to num of types"
        self.max_span_width = model_config.max_span_length

    def get_endpoint_span_extractor(self):
        return EndpointSpanExtractor(self.input_dim)

    def get_classifier(self):
        return nn.Linear(self.input_dim * 2, self.num_class)

    def get_bert_encoding_for_batch(self, samples: list[Sample], model_config: ModelConfig) \
            -> BatchEncoding:
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


    def forward(
            self,
            samples: list[Sample],
            # collect: List
    ) -> tuple[torch.Tensor, SeqLabelPredictions]:
        batch_encoding = self.get_bert_encoding_for_batch(samples, self.model_config)
        # collect.append(batch_encoding)

        gold_token_level_annos_batch = get_annos_token_level(
            samples=samples,
            batch_encoding=batch_encoding
        )
        # SHAPE: (batch, seq, embed_dim)
        bert_embeddings_batch = self.get_bert_embeddings_for_batch(encoding=batch_encoding, samples=samples)
        # enumerate all possible spans
        # spans are inclusive
        all_possible_spans_list_batch = [
            enumerate_spans(batch_encoding.word_ids(batch_index=batch_idx), max_span_width=self.max_span_width)
            for batch_idx in range(len(samples))
        ]
        # SHAPE: (batch_size, num_spans)
        all_possible_spans_labels_batch = self.label_all_possible_spans_batch(all_possible_spans_list_batch,
                                                                              gold_token_level_annos_batch)
        # collect.append(all_possible_spans_labels_batch)
        # SHAPE: (batch_size, seq_len, 2)
        all_possible_spans_tensor_batch: torch.Tensor = torch.tensor(all_possible_spans_list_batch, device=device)
        # SHAPE: (batch_size, num_spans, endpoint_dim)
        all_possible_span_embeddings = self.get_span_embeddings(bert_embeddings_batch, all_possible_spans_tensor_batch)
        # SHAPE: (batch_size, num_spans, num_classes)
        predicted_all_possible_spans_logits_batch = self.classifier(all_possible_span_embeddings)
        loss: torch.Tensor = self.loss_function(torch.permute(predicted_all_possible_spans_logits_batch, (0, 2, 1)),
                                                all_possible_spans_labels_batch)

        predicted_annos = self.get_predicted_annos(
            predicted_all_possible_spans_logits_batch=predicted_all_possible_spans_logits_batch,
            all_possible_spans_list_batch=all_possible_spans_list_batch,
            batch_encoding=batch_encoding,
            samples=samples
        )
        return loss, predicted_annos

    def get_span_embeddings(self, bert_embeddings_batch, all_possible_spans_tensor_batch) -> torch.Tensor:
        return self.endpoint_span_extractor(bert_embeddings_batch, all_possible_spans_tensor_batch)

    def label_all_possible_spans_batch(self, all_possible_spans_list_batch, gold_token_level_annos_batch):
        all_possible_spans_labels_batch = self._label_all_possible_spans_batch(
            all_possible_spans_list_batch=all_possible_spans_list_batch,
            sub_token_level_annos_batch=gold_token_level_annos_batch
        )
        # SHAPE: (batch_size, num_spans)
        all_possible_spans_labels_batch = torch.tensor(all_possible_spans_labels_batch, device=device)
        return all_possible_spans_labels_batch

    def get_predicted_annos(
            self,
            predicted_all_possible_spans_logits_batch,
            all_possible_spans_list_batch,
            batch_encoding: BatchEncoding,
            samples: list[Sample]
    ) -> list[list[Annotation]]:
        ret = []
        # SHAPE: (batch_size, num_spans)
        pred_all_possible_spans_type_indices_list_batch = torch \
            .argmax(predicted_all_possible_spans_logits_batch, dim=2) \
            .cpu() \
            .detach().numpy()

        assert len(pred_all_possible_spans_type_indices_list_batch.shape) == 2
        assert pred_all_possible_spans_type_indices_list_batch.shape[0] == len(samples)

        for batch_idx, sample in enumerate(samples):
            sample_annos = []
            pred_all_possible_spans_type_indices_list = pred_all_possible_spans_type_indices_list_batch[batch_idx]
            for i, span_type_idx in enumerate(pred_all_possible_spans_type_indices_list):
                if span_type_idx != self.type_to_idx['NO_TYPE']:  # found a span with a label
                    # get token level spans
                    span_start_token_idx = all_possible_spans_list_batch[batch_idx][i][0]
                    span_end_token_idx = all_possible_spans_list_batch[batch_idx][i][1]  # inclusive
                    # get char offsets
                    start_char_span = batch_encoding.token_to_chars(batch_or_token_index=batch_idx,
                                                                    token_index=span_start_token_idx)
                    end_char_span = batch_encoding.token_to_chars(batch_or_token_index=batch_idx,
                                                                  token_index=span_end_token_idx)
                    # Ignore any prediction which has a boundary made of special tokens
                    if (start_char_span is not None) and (end_char_span is not None):
                        span_start_char_offset = start_char_span.start
                        span_end_char_offset = end_char_span.end
                        sample_annos.append(
                            Annotation(
                                span_start_char_offset,
                                span_end_char_offset,
                                self.idx_to_type[span_type_idx],
                                sample.text[span_start_char_offset:span_end_char_offset],
                            )
                        )
            ret.append(sample_annos)
        return ret

    def _label_all_possible_spans(self, all_possible_spans_list, sub_token_level_annos):
        all_possible_spans_labels = []
        for span in all_possible_spans_list:
            corresponding_anno_list = [anno for anno in sub_token_level_annos if
                                       (anno.begin_offset == span[0]) and (anno.end_offset == (span[1] + 1))]  # spans are inclusive
            if len(corresponding_anno_list):
                if len(corresponding_anno_list) > 1:
                    print(f"WARN: Didn't expect multiple annotations to match one span: {corresponding_anno_list}")
                corresponding_anno = corresponding_anno_list[0]
                all_possible_spans_labels.append(self.type_to_idx[corresponding_anno.label_type])
            else:
                all_possible_spans_labels.append(self.type_to_idx["NO_TYPE"])
        assert len(all_possible_spans_labels) == len(all_possible_spans_list)
        return all_possible_spans_labels

    def _label_all_possible_spans_batch(
            self,
            all_possible_spans_list_batch: list[list[tuple]],
            sub_token_level_annos_batch: list[list[Annotation]]
    ):
        assert len(all_possible_spans_list_batch) == len(sub_token_level_annos_batch)
        ret = []
        for all_possible_spans_list, sub_token_level_annos in zip(all_possible_spans_list_batch,
                                                                  sub_token_level_annos_batch):
            ret.append(self._label_all_possible_spans(all_possible_spans_list, sub_token_level_annos))
        assert len(ret) == len(all_possible_spans_list_batch)
        return ret




def get_annos_token_level(samples: list[Sample], batch_encoding: BatchEncoding) -> list[list[Annotation]]:
    ret = []
    for batch_idx, sample in enumerate(samples):
        token_level_annos_for_sample = []
        for gold_anno in sample.annos.gold:
            start_token_idx = batch_encoding.char_to_token(batch_or_char_index=batch_idx,
                                                           char_index=gold_anno.begin_offset)
            end_token_idx = batch_encoding.char_to_token(batch_or_char_index=batch_idx,
                                                         char_index=(gold_anno.end_offset - 1))
            if (start_token_idx is not None) and (end_token_idx is not None):
                token_level_annos_for_sample.append(
                    Annotation(
                        begin_offset=start_token_idx,
                        end_offset=end_token_idx + 1,
                        label_type=gold_anno.label_type,
                        extraction=gold_anno.extraction,
                        features=gold_anno.features
                    )
                )
            else:
                print(f"WARN: No char_index to token_index mapping:")
                print(f"DEBUG: SampleId: {sample.id}")
                print(f"DEBUG: Annotation: {gold_anno}")
                # print(f"DEBUG: Sample id: {sample.id}")
                # print(f"DEBUG: Sample text: {sample.text}")
        ret.append(token_level_annos_for_sample)
    return ret
