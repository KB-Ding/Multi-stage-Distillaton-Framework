import torch
from torch import nn, Tensor
from typing import Iterable, Dict
from sentence_transformers import SentenceTransformer, util
import copy
import random
import math

from . import TripletDistanceMetric
from .. import InputExample
import numpy as np

# class Multilingual_ContrastiveLoss(nn.Module):
#     """
#         This loss expects as input a batch consisting of multiple mini-batches of sentence pairs (a_1, p_1), (a_2, p_2)..., (a_{K+1}, p_{K+1})
#         where p_1 = a_1 = a_2 = ... a_{K+1} and p_2, p_3, ..., p_{K+1} are expected to be different from p_1 (this is done via random sampling).
#         The corresponding labels y_1, y_2, ..., y_{K+1} for each mini-batch are assigned as: y_i = 1 if i == 1 and y_i = 0 otherwise.
#         In other words, K represent the number of negative pairs and the positive pair is actually made of two identical sentences. The data generation
#         process has already been implemented in readers/ContrastiveTensionReader.py
#         For tractable optimization, two independent encoders ('model1' and 'model2') are created for encoding a_i and p_i, respectively. For inference,
#         only model2 are used, which gives better performance. The training objective is binary cross entropy.
#         For more information, see: https://openreview.net/pdf?id=Ov_sMNau-PF
#
#     """
#     def __init__(self, model: SentenceTransformer):
#         """
#         :param model: SentenceTransformer model
#         """
#         super(Multilingual_ContrastiveLoss, self).__init__()
#         self.model = model  # This will be the final model used during the inference time.
#         # self.model1 = copy.deepcopy(model)
#         self.criterion = nn.BCEWithLogitsLoss(reduction='sum')
#
#     def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
#         sentence_features1, sentence_features2 = tuple(sentence_features)
#         reps_1 = self.model1(sentence_features1)['sentence_embedding']  # (bsz, hdim)
#         reps_2 = self.model2(sentence_features2)['sentence_embedding']
#
#         sim_scores = torch.matmul(reps_1[:,None], reps_2[:,:,None]).squeeze(-1).squeeze(-1)  # (bsz,) dot product, i.e. S1S2^T
#
#         loss = self.criterion(sim_scores, labels.type_as(sim_scores))
#         return loss


class Multilingual_Teacher_TripletLoss(nn.Module):
    def __init__(self, model: SentenceTransformer, triplet_margin: float = 5):
        """
        :param model: SentenceTransformer model
        """
        super(Multilingual_Teacher_TripletLoss, self).__init__()
        self.model = model  # This will be the final model used during the inference time.
        # self.model1 = copy.deepcopy(model)
        self.similarity_fct = TripletDistanceMetric.COSINE
        self.triplet_margin = triplet_margin
        #self.scale = scale

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        pos_emb, neg_emb = labels[:, 0, :], labels[:, 1, :]

        embeddings = self.model(sentence_features[0])['sentence_embedding']  # (bsz, hdim)

        # aa = TripletDistanceMetric.COSINE(embeddings, pos_emb)

        distance_pos = self.similarity_fct(embeddings, pos_emb)
        distance_neg = self.similarity_fct(embeddings, neg_emb)

        losses = nn.functional.relu(distance_pos - distance_neg + self.triplet_margin)
        return losses.mean()

        # # embeddings_b = self.model2(sentence_features2)['sentence_embedding']
        #
        # scores = self.similarity_fct(embeddings, sentence_features1) * self.logit_scale.exp()  #self.scale
        # labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)
        # return (self.cross_entropy_loss(scores, labels) + self.cross_entropy_loss(scores.t(), labels))/2

################# CT Data Loader #################
# For CT, we need batches in a specific format
# In each batch, we have one positive pair (i.e. [sentA, sentA]) and 7 negative pairs (i.e. [sentA, sentB]).
# To achieve this, we create a custom DataLoader that produces batches with this property

class ContrastiveTensionDataLoader:
    def __init__(self, sentences, batch_size): # pos_neg_ratio=8)
        self.sentences = sentences
        self.batch_size = batch_size
        # self.pos_neg_ratio = pos_neg_ratio
        self.collate_fn = None

        # if self.batch_size % self.pos_neg_ratio != 0:
        #     raise ValueError(f"ContrastiveTensionDataLoader was loaded with a pos_neg_ratio of {pos_neg_ratio} and a batch size of {batch_size}. The batch size must be devisable by the pos_neg_ratio")

    def __iter__(self):
        random.shuffle(self.sentences)
        sentence_idx = 0
        batch = []

        while sentence_idx + 1 < len(self.sentences):
            s1 = self.sentences[sentence_idx]
            # if len(batch) % self.pos_neg_ratio > 0:    #Negative (different) pair
            sentence_idx += 1
            s2 = self.sentences[sentence_idx]
            label = 0
           #  else:   #Positive (identical pair)
           #  s2 = self.sentences[sentence_idx]
           #  label = 1

            sentence_idx += 1
            batch.append(InputExample(texts=[s1, s2], label=label))

            if len(batch) >= self.batch_size:
                yield self.collate_fn(batch) if self.collate_fn is not None else batch
                batch = []

    def __len__(self):
        return math.floor(len(self.sentences)/(2*self.batch_size))