import os

from typing import List, Dict, Tuple, Union
from utils.cos import cos_sim
from numpy import ndarray
from torch import nn, Tensor

from sentence_transformers import models, SentenceTransformer


class mse(nn.Module):
    def __init__(self, config, *args, **params):

        super(mse, self).__init__()
        # ver 1 Sbert
        hugging_dir = os.path.join(config.get("model", "student_model_path"))
        self.model = SentenceTransformer(hugging_dir)
        # ver2 hugging face
        # hugging_dir = os.path.join(config.get("model", "student_model_path"))
        # word_embedding_model = models.Transformer(hugging_dir, max_seq_length=config.getint("data", "max_seq_length"),
        #                                           cache_dir=hugging_dir)
        # # Apply mean pooling to get one fixed sized sentence vector
        # pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        # self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

        self.config = config

        self.similarity_fct = cos_sim
        self.en_loss_fct = nn.MSELoss()
        self.mul_loss_fct = nn.MSELoss()

    def forward(self, data):

        embeddings_a = self.model({'input_ids': data["en_input_ids"], 'attention_mask': data["en_attention_mask"]})['sentence_embedding']
        embeddings_b = self.model({'input_ids': data["mul_input_ids"], 'attention_mask': data["mul_attention_mask"]})['sentence_embedding']
        teacher_embedding = data["labels"]

        en_loss = self.en_loss_fct(embeddings_a, teacher_embedding)
        mul_loss = self.mul_loss_fct(embeddings_b, teacher_embedding)
        total_loss = en_loss + mul_loss

        return {"total_loss": total_loss,
                "en_loss": en_loss,
                "mul_loss": mul_loss
                }

    def get_config_dict(self):
        return {'scale': self.scale, 'similarity_fct': self.similarity_fct.__name__}


    def encode(self, sentences: Union[str, List[str], List[int]],
               batch_size: int = 32,
               show_progress_bar: bool = None,
               output_value: str = 'sentence_embedding',
               convert_to_numpy: bool = True,
               convert_to_tensor: bool = False,
               device: str = None,
               normalize_embeddings: bool = False) -> Union[List[Tensor], ndarray, Tensor]:

        model_to_encode = self.model.module if hasattr(self.model, 'module') else self.model
        return model_to_encode.encode(sentences, batch_size, show_progress_bar, output_value, convert_to_numpy,convert_to_tensor,
                                 device, normalize_embeddings)

    def tokenize(self, texts: Union[List[str], List[Dict], List[Tuple[str, str]]]):
        model_to_tokenize = self.model.module if hasattr(self.model, 'module') else self.model
        return model_to_tokenize.tokenize(texts)