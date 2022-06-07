import torch
# from torch import nn, Tensor
import os

from typing import List, Dict, Tuple, Union

from runx.logx import logx
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

from utils.cos import cos_sim
from numpy import ndarray
from torch import nn, Tensor

from sentence_transformers import SentenceTransformer, models

from utils.message_utils import infor_msg


class wrapper_model(nn.Module):
    def __init__(self, config, hugging_dir):
        super(wrapper_model, self).__init__()

        hugging_dir = os.path.join(config.get("model", "student_model_path"))
        word_embedding_model = models.Transformer(hugging_dir, max_seq_length=config.getint("data", "max_seq_length"),
                                                  cache_dir=hugging_dir)
        # Apply mean pooling to get one fixed sized sentence vector
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

        # self.model = SentenceTransformer(hugging_dir)


class single_bottle_mcl(nn.Module):
    def __init__(self, config, *args, **params):
        super(single_bottle_mcl, self).__init__()

        hugging_dir = os.path.join(config.get("model", "student_model_path"))

        layers_to_keep = [*range(config.getint("model", "layers_to_keep"))]
        logx.msg(infor_msg("Remove layers from student. Only keep these layers: {}".format(layers_to_keep)))

        emb_model = wrapper_model(config, hugging_dir)  # ._first_module().auto_model.embeddings
        emb_model.model._first_module().auto_model.embeddings = Small_Embeddings(config)

        new_layers = torch.nn.ModuleList(
            [layer_module for i, layer_module in enumerate(emb_model.model._first_module().auto_model.encoder.layer) if
             i in layers_to_keep])
        emb_model.model._first_module().auto_model.encoder.layer = new_layers

        emb_model.model._first_module().auto_model.config.num_hidden_layers = len(layers_to_keep)

        # param = torch.load(config.get("model", "emb_ckp_path"), map_location=torch.device('cpu'))
        # emb_model = emb_model.module if hasattr(emb_model, 'module') else emb_model
        # emb_model.load_state_dict(param["model"])

        self.model = emb_model.model

        self.config = config
        self.similarity_fct = cos_sim
        self.en_loss_fct = nn.MSELoss()
        self.mul_loss_fct = nn.MSELoss()
        self.loss_fct = nn.MSELoss()

    def layers_forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.model._first_module().auto_model.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None

        for j in range(self.config.getint("model", "repeat")):
            for i, layer_module in enumerate(self.model._first_module().auto_model.encoder.layer):

                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                layer_head_mask = head_mask[i] if head_mask is not None else None
                past_key_value = past_key_values[i] if past_key_values is not None else None

                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

                hidden_states = layer_outputs[0]
                if use_cache:
                    next_decoder_cache += (layer_outputs[-1],)
                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)
                    if self.model._first_module().auto_model.config.add_cross_attention:
                        all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

    def roberta_model_forward(
            self,
            input_ids=None,
            attention_mask=None,
    ):

        input_shape = input_ids.size()
        batch_size, seq_length = input_shape
        device = input_ids.device

        token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        extended_attention_mask: torch.Tensor = self.model._first_module().auto_model.get_extended_attention_mask(
            attention_mask, input_shape, device)

        embedding_output = self.model._first_module().auto_model.embeddings(
            input_ids=input_ids,
            position_ids=None,
            token_type_ids=token_type_ids,
            inputs_embeds=None,
            past_key_values_length=0,
        )

        head_mask = self.model._first_module().auto_model.get_head_mask(None,
                                                                        self.model._first_module().auto_model.config.num_hidden_layers)

        encoder_outputs = self.layers_forward(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
        )
        sequence_output = encoder_outputs[0]

        pooled_output = self.model._first_module().auto_model.pooler(
            sequence_output) if self.model._first_module().auto_model.pooler is not None else None

        output_states = (sequence_output, pooled_output) + encoder_outputs[1:]
        output_tokens = output_states[0]

        # output_tokens = pooled_output
        features = {'input_ids': input_ids, 'attention_mask': attention_mask}
        cls_tokens = output_tokens[:, 0, :]  # CLS token is first token
        features.update({'token_embeddings': output_tokens, 'cls_token_embeddings': cls_tokens,
                         'attention_mask': features['attention_mask']})

        if self.model._first_module().auto_model.config.output_hidden_states:
            all_layer_idx = 2
            if len(output_states) < 3:  # Some models only output last_hidden_states and all_hidden_states
                all_layer_idx = 1

            hidden_states = output_states[all_layer_idx]
            features.update({'all_layer_embeddings': hidden_states})

        output = self.model._last_module()(features)
        return output

    def forward(self, data):

        embeddings_a = self.roberta_model_forward(input_ids=data["en_input_ids"],
                                                  attention_mask=data["en_attention_mask"])['sentence_embedding']
        embeddings_b = self.roberta_model_forward(input_ids=data["mul_input_ids"],
                                                  attention_mask=data["mul_attention_mask"])['sentence_embedding']

        teacher_embedding = data["labels"]
        label_scores = data["scores"]
        scores = self.similarity_fct(embeddings_a, embeddings_b)  # * self.scale

        cot_loss = self.loss_fct(label_scores, scores)

        en_loss = self.en_loss_fct(embeddings_a, teacher_embedding)
        mul_loss = self.mul_loss_fct(embeddings_b, teacher_embedding)
        total_loss = cot_loss + en_loss + mul_loss
        return {"total_loss": total_loss,
                "ce_loss": cot_loss,
                "en_loss": en_loss,
                "mul_loss": mul_loss
                }

    def get_config_dict(self):
        return {'similarity_fct': self.similarity_fct.__name__}

    def encode(self, sentences: Union[str, List[str], List[int]],
               batch_size: int = 32,
               show_progress_bar: bool = None,
               output_value: str = 'sentence_embedding',
               convert_to_numpy: bool = True,
               convert_to_tensor: bool = False,
               device: str = None,
               normalize_embeddings: bool = False) -> Union[List[Tensor], ndarray, Tensor]:

        import logging
        import numpy as np
        from tqdm.autonotebook import trange
        from sentence_transformers.util import batch_to_device
        logger = logging.getLogger(__name__)

        model_to_encode = self.model.module if hasattr(self.model, 'module') else self.model

        model_to_encode.eval()
        if show_progress_bar is None:
            show_progress_bar = (
                    logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG)

        if convert_to_tensor:
            convert_to_numpy = False

        if output_value == 'token_embeddings':
            convert_to_tensor = False
            convert_to_numpy = False

        input_was_string = False
        if isinstance(sentences, str) or not hasattr(sentences,
                                                     '__len__'):  # Cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True

        if device is None:
            device = model_to_encode._target_device

        model_to_encode.to(device)

        all_embeddings = []
        length_sorted_idx = np.argsort([-model_to_encode._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
            sentences_batch = sentences_sorted[start_index:start_index + batch_size]
            features = model_to_encode.tokenize(sentences_batch)
            features = batch_to_device(features, device)

            with torch.no_grad():
                out_features = self.roberta_model_forward(input_ids=features['input_ids'],
                                                          attention_mask=features['attention_mask'])
                # out_features = xxx.forward(features)

                if output_value == 'token_embeddings':
                    embeddings = []
                    for token_emb, attention in zip(out_features[output_value], out_features['attention_mask']):
                        last_mask_id = len(attention) - 1
                        while last_mask_id > 0 and attention[last_mask_id].item() == 0:
                            last_mask_id -= 1

                        embeddings.append(token_emb[0:last_mask_id + 1])
                else:  # multilingual embeddings
                    embeddings = out_features[output_value]
                    embeddings = embeddings.detach()
                    if normalize_embeddings:
                        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                    if convert_to_numpy:
                        embeddings = embeddings.cpu()

                all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

        if convert_to_tensor:
            all_embeddings = torch.stack(all_embeddings)
        elif convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])

        if input_was_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings

    def tokenize(self, texts: Union[List[str], List[Dict], List[Tuple[str, str]]]):
        model_to_tokenize = self.model.module if hasattr(self.model, 'module') else self.model
        return model_to_tokenize.tokenize(texts)


def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx


class Small_Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.getint("model", "vocab_size"),
                                            config.getint("model", "embedding_size"),
                                            padding_idx=config.getint("model", "pad_token_id"))
        # project layer
        self.word_embeddings_2 = nn.Linear(config.getint("model", "embedding_size"),
                                           config.getint("model", "hidden_size"), bias=False)

        self.position_embeddings = nn.Embedding(config.getint("model", "max_position_embeddings"),
                                                config.getint("model", "hidden_size"))
        self.token_type_embeddings = nn.Embedding(config.getint("model", "type_vocab_size"),
                                                  config.getint("model", "hidden_size"))

        self.LayerNorm = nn.LayerNorm(config.getint("model", "hidden_size"),
                                      eps=config.getfloat("model", "layer_norm_eps"))
        self.dropout = nn.Dropout(config.getfloat("model", "hidden_dropout_prob"))

        self.register_buffer("position_ids",
                             torch.arange(config.getint("model", "max_position_embeddings")).expand((1, -1)))
        self.position_embedding_type = "absolute"

        self.padding_idx = config.getint("model", "pad_token_id")
        self.position_embeddings = nn.Embedding(
            config.getint("model", "max_position_embeddings"), config.getint("model", "hidden_size"),
            padding_idx=self.padding_idx
        )

    def forward(
            self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if position_ids is None:
            if input_ids is not None:
                position_ids = create_position_ids_from_input_ids(
                    input_ids, self.padding_idx, past_key_values_length
                ).to(input_ids.device)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
            # project transform
            inputs_embeds = self.word_embeddings_2(inputs_embeds)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)
