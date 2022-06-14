import os

from sentence_transformers.SentenceTransformer import SentenceTransformer
import copy
import gzip
from runx.logx import logx
# from model.sentence_transformers.readers import InputExample
from typing import List
import random
from torch.utils.data import Dataset
from sentence_transformers import models

from utils.input_type import parallel_input
from utils.message_utils import infor_msg


class parallel_all_distill_dataset(Dataset):


    def __init__(self, config, mode, encoding="utf8"):

        self.count = 0

        self.teacher_model = SentenceTransformer(config.get("data", "teacher_path"))
        self.datasets = []
        self.datasets_iterator = []
        self.datasets_tokenized = []
        self.dataset_indices = []
        self.copy_dataset_indices = []
        self.cache = []
        self.batch_size = config.getint("train", "batch_size")
        self.use_embedding_cache = config.getboolean("train", "use_embedding_cache")
        self.embedding_cache = {}
        self.num_sentences = 0

        source_languages = ['en']
        target_languages = ['de', 'es', 'it', 'fr', 'ar', 'tr', 'nl']
        data_dir_cache_folder = config.get("data", "data_dir")
        parallel_sentences_folder = os.path.join(data_dir_cache_folder, 'parallel-sentences/')
        os.makedirs(parallel_sentences_folder, exist_ok=True)
        train_files = []
        dev_files = []
        files_to_create = []
        for source_lang in source_languages:
            for target_lang in target_languages:
                output_filename_train = os.path.join(parallel_sentences_folder,
                                                     "TED2020-{}-{}-train.tsv.gz".format(source_lang, target_lang))
                output_filename_dev = os.path.join(parallel_sentences_folder,
                                                   "TED2020-{}-{}-dev.tsv.gz".format(source_lang, target_lang))
                train_files.append(output_filename_train)
                dev_files.append(output_filename_dev)
                if not os.path.exists(output_filename_train) or not os.path.exists(output_filename_dev):
                    files_to_create.append({'src_lang': source_lang, 'trg_lang': target_lang,
                                            'fTrain': gzip.open(output_filename_train, 'wt', encoding='utf8'),
                                            'fDev': gzip.open(output_filename_dev, 'wt', encoding='utf8'),
                                            'devCount': 0
                                            })

        # add wikimatrix
        for source_lang in source_languages:
            for target_lang in target_languages:
                wikifile = os.path.join(parallel_sentences_folder, "WikiMatrix-{}-{}-train.tsv.gz".format(source_lang, target_lang))
                train_files.append(wikifile)
        # add News-Commentary
        for source_lang in source_languages:
            for target_lang in target_languages:
                News = os.path.join(parallel_sentences_folder, "News-Commentary-{}-{}.tsv.gz".format(source_lang, target_lang))
                train_files.append(News)

        # add Europarl
        for source_lang in source_languages:
            for target_lang in target_languages:
                Europarl = os.path.join(parallel_sentences_folder, "Europarl-{}-{}.tsv.gz".format(source_lang, target_lang))
                train_files.append(Europarl)


        max_sentences_per_language = 300000  # Maximum number of  parallel sentences for training
        train_max_sentence_length = 250  # Maximum length (characters) for parallel training sentences

        for train_file in train_files:
            self.load_data(train_file, max_sentences=max_sentences_per_language,
                                 max_sentence_length=train_max_sentence_length)

    def load_data(self, filepath: str, weight: int = 100, max_sentences: int = None, max_sentence_length: int = 128):

        logx.msg(infor_msg("Load " + filepath))

        parallel_sentences = []

        with gzip.open(filepath, 'rt', encoding='utf8') if filepath.endswith('.gz') else open(filepath,
                                                                                              encoding='utf8') as fIn:
            count = 0
            for line in fIn:
                sentences = line.strip().split("\t")
                if max_sentence_length is not None and 0 < max_sentence_length < max(
                        [len(sent) for sent in sentences]):
                    continue

                parallel_sentences.append(sentences)
                count += 1
                if max_sentences is not None and max_sentences > 0 and count >= max_sentences:
                    break

        self.add_dataset(parallel_sentences, weight=weight, max_sentences=max_sentences,
                         max_sentence_length=max_sentence_length)
        logx.msg(str(self.num_sentences))

    def add_dataset(self, parallel_sentences: List[List[str]], weight: int = 100, max_sentences: int = None,
                    max_sentence_length: int = 128):
        sentences_map = {}
        for sentences in parallel_sentences:
            if max_sentence_length is not None and max_sentence_length > 0 and max(
                    [len(sent) for sent in sentences]) > max_sentence_length:
                continue

            source_sentence = sentences[0]
            if source_sentence not in sentences_map:
                sentences_map[source_sentence] = set()

            for sent in sentences:
                sentences_map[source_sentence].add(sent)

            if max_sentences is not None and max_sentences > 0 and len(sentences_map) >= max_sentences:
                break

        if len(sentences_map) == 0:
            return

        self.num_sentences += sum([len(sentences_map[sent]) for sent in sentences_map])

        dataset_id = len(self.datasets)

        self.datasets.append(list(sentences_map.items()))
        self.datasets_iterator.append(0)
        self.dataset_indices.extend([dataset_id] * weight)

    def generate_data(self):
        source_sentences_list = []
        target_sentences_list = []
        mul_sentence_list = []

        for data_idx in self.dataset_indices:
            src_sentence, trg_sentences = self.next_entry(data_idx)
            source_sentences_list.append(src_sentence)
            target_sentences_list.append(trg_sentences)
            mul_sentence_list.append(trg_sentences[-1])

        # Generate embeddings
        src_embeddings = self.get_embeddings(source_sentences_list)
        mul_embeddings = self.get_embeddings(mul_sentence_list)


        for src_embedding, mul_embedding, trg_sentences in zip(src_embeddings, mul_embeddings, target_sentences_list):
            for i in range(0, len(trg_sentences) - 1, 2):
                self.cache.append(parallel_input(texts=[trg_sentences[i], trg_sentences[i + 1]], label=[src_embedding, mul_embedding]))

        random.shuffle(self.cache)

    def next_entry(self, data_idx):
        srcsource, srctarget_sentences = self.datasets[data_idx][self.datasets_iterator[data_idx]]

        self.datasets_iterator[data_idx] += 1
        if self.datasets_iterator[data_idx] >= len(self.datasets[data_idx]):
            self.datasets_iterator[data_idx] = 0
            random.shuffle(self.datasets[data_idx])
        source = copy.deepcopy(srcsource)
        target_sentences = copy.deepcopy(srctarget_sentences)
        en_trg = ""
        parall_trg = ""
        ret = []
        if len(target_sentences) == 0:
            raise ValueError
        elif len(target_sentences) == 1:
            for trg in target_sentences:
                en_trg = trg
                parall_trg = trg
            ret.append(en_trg)
            ret.append(parall_trg)

        elif len(target_sentences) == 2:
            for trg in target_sentences:
                if trg == source:
                    en_trg = trg
                else:
                    parall_trg = trg

            ret.append(en_trg)
            ret.append(parall_trg)
        else:
            while len(target_sentences) > 0:
                trg = target_sentences.pop()
                if trg == source:
                    continue
                else:
                    ret.append(source)
                    ret.append(trg)

        return source, ret

    def get_embeddings(self, sentences):
        if not self.use_embedding_cache:
            return self.teacher_model.encode(sentences, batch_size=self.batch_size, show_progress_bar=False,
                                             convert_to_numpy=True)

        # Use caching
        new_sentences = []
        for sent in sentences:
            if sent not in self.embedding_cache:
                new_sentences.append(sent)

        if len(new_sentences) > 0:
            new_embeddings = self.teacher_model.encode(new_sentences, batch_size=self.batch_size,
                                                       show_progress_bar=False, convert_to_numpy=True)
            for sent, embedding in zip(new_sentences, new_embeddings):
                self.embedding_cache[sent] = embedding

        return [self.embedding_cache[sent] for sent in sentences]

    def __len__(self):
        return self.num_sentences

    def __getitem__(self, idx):
        if len(self.cache) == 0:
            self.generate_data()

        return self.cache.pop()
