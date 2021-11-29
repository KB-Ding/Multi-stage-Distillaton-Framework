# from utils import InputType
from evaluator.evaluate.basic_evaluator import basic_evaluator
import os
import csv
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from scipy.stats import pearsonr, spearmanr
import numpy as np
from typing import List
from enum import Enum

from utils.input_type import parallel_input

class similarity_function(Enum):
    COSINE = 0
    EUCLIDEAN = 1
    MANHATTAN = 2
    DOT_PRODUCT = 3



class embedding_similarity_evaluator(basic_evaluator):

    def __init__(self, sentences1: List[str], sentences2: List[str], scores: List[float], batch_size: int = 16, main_similarity: similarity_function = None, name: str = '', show_progress_bar: bool = False, write_csv: bool = True):

        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.scores = scores
        self.write_csv = write_csv

        assert len(self.sentences1) == len(self.sentences2)
        assert len(self.sentences1) == len(self.scores)

        self.main_similarity = main_similarity
        self.name = name

        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar

        self.csv_file = "similarity_evaluation"+("_"+name if name else '')+"_results.csv"
        self.csv_headers = ["epoch", "steps", "cosine_pearson", "cosine_spearman", "euclidean_pearson", "euclidean_spearman", "manhattan_pearson", "manhattan_spearman", "dot_pearson", "dot_spearman"]

    @classmethod
    def from_input_examples(cls, examples: List[parallel_input], **kwargs):
        sentences1 = []
        sentences2 = []
        scores = []

        for example in examples:
            sentences1.append(example.texts[0])
            sentences2.append(example.texts[1])
            scores.append(example.label)
        return cls(sentences1, sentences2, scores, **kwargs)


    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1, bistream = False) -> float:
        # if epoch != -1:
        #     if steps == -1:
        #         out_txt = " after epoch {}:".format(epoch)
        #     else:
        #         out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        # else:
        #     out_txt = ":"
        # logx.msg(infor_msg("EmbeddingSimilarityEvaluator: Evaluating the model on " + self.name + " dataset" + out_txt))

        if not bistream:
            embeddings1 = model.encode(self.sentences1, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_numpy=True)
            embeddings2 = model.encode(self.sentences2, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_numpy=True)
            labels = self.scores
        else:
            embeddings1 = model[0].encode(self.sentences1, batch_size=self.batch_size,
                                       show_progress_bar=self.show_progress_bar, convert_to_numpy=True)
            embeddings2 = model[1].encode(self.sentences2, batch_size=self.batch_size,
                                       show_progress_bar=self.show_progress_bar, convert_to_numpy=True)
            labels = self.scores

        cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))
        manhattan_distances = -paired_manhattan_distances(embeddings1, embeddings2)
        euclidean_distances = -paired_euclidean_distances(embeddings1, embeddings2)
        dot_products = [np.dot(emb1, emb2) for emb1, emb2 in zip(embeddings1, embeddings2)]


        eval_pearson_cosine, _ = pearsonr(labels, cosine_scores)
        eval_spearman_cosine, _ = spearmanr(labels, cosine_scores)

        eval_pearson_manhattan, _ = pearsonr(labels, manhattan_distances)
        eval_spearman_manhattan, _ = spearmanr(labels, manhattan_distances)

        eval_pearson_euclidean, _ = pearsonr(labels, euclidean_distances)
        eval_spearman_euclidean, _ = spearmanr(labels, euclidean_distances)

        eval_pearson_dot, _ = pearsonr(labels, dot_products)
        eval_spearman_dot, _ = spearmanr(labels, dot_products)

        # logx.msg(infor_msg("Cosine-Similarity :\tPearson: {:.4f}\tSpearman: {:.4f}".format(
        #     eval_pearson_cosine, eval_spearman_cosine))
        # )
        #
        # logx.msg(infor_msg("Manhattan-Distance:\tPearson: {:.4f}\tSpearman: {:.4f}".format(
        #     eval_pearson_manhattan, eval_spearman_manhattan))
        # )
        #
        # logx.msg(infor_msg("Euclidean-Distance:\tPearson: {:.4f}\tSpearman: {:.4f}".format(
        #     eval_pearson_euclidean, eval_spearman_euclidean))
        # )
        #
        # logx.msg(infor_msg("Dot-Product-Similarity:\tPearson: {:.4f}\tSpearman: {:.4f}".format(
        #     eval_pearson_dot, eval_spearman_dot))
        # )


        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, newline='', mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow([epoch, steps, eval_pearson_cosine, eval_spearman_cosine, eval_pearson_euclidean,
                                 eval_spearman_euclidean, eval_pearson_manhattan, eval_spearman_manhattan, eval_pearson_dot, eval_spearman_dot])


        if self.main_similarity == similarity_function.COSINE:
            return eval_spearman_cosine
        elif self.main_similarity == similarity_function.EUCLIDEAN:
            return eval_spearman_euclidean
        elif self.main_similarity == similarity_function.MANHATTAN:
            return eval_spearman_manhattan
        elif self.main_similarity == similarity_function.DOT_PRODUCT:
            return eval_spearman_dot
        elif self.main_similarity is None:
            return max(eval_spearman_cosine, eval_spearman_manhattan, eval_spearman_euclidean, eval_spearman_dot)
        else:
            raise ValueError("Unknown main_similarity value")
