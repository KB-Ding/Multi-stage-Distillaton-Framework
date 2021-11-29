from torch.utils.data import Dataset
import logging
import gzip
from .. import SentenceTransformer
from ..readers import InputExample
from typing import List
import random

logger = logging.getLogger(__name__)


class Teacher_Triplet_Dataset(Dataset):
    """
    This dataset reader can be used to read-in parallel sentences, i.e., it reads in a file with tab-seperated sentences with the same
    sentence in different languages. For example, the file can look like this (EN\tDE\tES):
    hello world     hallo welt  hola mundo
    second sentence zweiter satz    segunda oración

    The sentence in the first column will be mapped to a sentence embedding using the given the embedder. For example,
    embedder is a mono-lingual sentence embedding method for English. The sentences in the other languages will also be
    mapped to this English sentence embedding.

    When getting a sample from the dataset, we get one sentence with the according sentence embedding for this sentence.

    teacher_model can be any class that implement an encode function. The encode function gets a list of sentences and
    returns a list of sentence embeddings
    """

    def __init__(self, student_model: SentenceTransformer, teacher_model: SentenceTransformer, batch_size: int = 8,
                 use_embedding_cache: bool = True):
        """
        Parallel sentences dataset reader to train student model given a teacher model
        :param student_model: Student sentence embedding model that should be trained
        :param teacher_model: Teacher model, that provides the sentence embeddings for the first column in the dataset file
        """
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.datasets = []
        # self.positive_datasets = []
        # self.negative_datasets = []
        self.datasets_iterator = []
        self.datasets_tokenized = []
        self.dataset_indices = []
        self.copy_dataset_indices = []
        self.cache = []
        self.batch_size = batch_size
        self.use_embedding_cache = use_embedding_cache
        self.embedding_cache = {}
        self.num_sentences = 0

    def load_data(self, filepath: str, weight: int = 100, max_sentences: int = None, max_sentence_length: int = 128):
        """
        Reads in a tab-seperated .txt/.csv/.tsv or .gz file. The different columns contain the different translations of the sentence in the first column

        :param filepath: Filepath to the file
        :param weight: If more than one dataset is loaded with load_data: With which frequency should data be sampled from this dataset?
        :param max_sentences: Max number of lines to be read from filepath
        :param max_sentence_length: Skip the example if one of the sentences is has more characters than max_sentence_length
        :param batch_size: Size for encoding parallel sentences
        :return:
        """

        logger.info("Load " + filepath)
        parallel_sentences = []

        with gzip.open(filepath, 'rt', encoding='utf8') if filepath.endswith('.gz') else open(filepath,
                                                                                              encoding='utf8') as fIn:
            count = 0
            for line in fIn:
                sentences = line.strip().split("\t")
                if max_sentence_length is not None and max_sentence_length > 0 and max(
                        [len(sent) for sent in sentences]) > max_sentence_length:
                    continue

                parallel_sentences.append(sentences)
                count += 1
                if max_sentences is not None and max_sentences > 0 and count >= max_sentences:
                    break
        # for sentence_idx, sent in enumerate(parallel_sentences):
        #     if sentence_idx < len(parallel_sentences) - 1:
        #         negative_idx = sentence_idx + 1
        #         ss = parallel_sentences[negative_idx][0]
        #     else:
        #         ss = parallel_sentences[0][0]
        #     parallel_sentences[sentence_idx].append(ss)
        self.add_dataset(parallel_sentences, weight=weight, max_sentences=max_sentences,
                         max_sentence_length=max_sentence_length)

    def add_dataset(self, parallel_sentences: List[List[str]], weight: int = 100, max_sentences: int = None,
                    max_sentence_length: int = 128):
        sentences_map = {}
        # negative_map = {}
        for sentences in parallel_sentences:
            if max_sentence_length is not None and max_sentence_length > 0 and max(
                    [len(sent) for sent in sentences]) > max_sentence_length:
                continue

            source_sentence = sentences[0]
            # negative_sentence = sentences[-1]
            if source_sentence not in sentences_map:
                sentences_map[source_sentence] = set()

            for sent in sentences:
                sentences_map[source_sentence].add(sent)
            # sentences_map[source_sentence] = list(sentences_map[source_sentence])
            # sentences_map[source_sentence].append(negative_sentence)
            # negative_map[source_sentence] =

            if max_sentences is not None and max_sentences > 0 and len(sentences_map) >= max_sentences:
                break

        if len(sentences_map) == 0:
            return

        self.num_sentences += sum([len(sentences_map[sent]) for sent in sentences_map])

        # assert len(self.positive_datasets) == len(self.negative_datasets)

        dataset_id = len(self.datasets)

        self.datasets.append(list(sentences_map.items()))
        self.datasets_iterator.append(0)
        # dataset_indices : 700长度，每个语言100，一共七种语言
        self.dataset_indices.extend([dataset_id] * weight)

    def generate_data(self):
        source_sentences_list = []
        target_sentences_list = []
        negative_sentence_list = []
        # TODO
        for data_idx in self.dataset_indices:
            src_sentence, trg_sentences = self.next_entry(data_idx)
            source_sentences_list.append(src_sentence)
            target_sentences_list.append(trg_sentences)
        for idx, ss in enumerate(source_sentences_list):
            if idx < len(source_sentences_list) - 1:
                negative_sentence_list.append(source_sentences_list[idx + 1])
            else:
                negative_sentence_list.append(source_sentences_list[0])

        # Generate embeddings
        src_embeddings = self.get_embeddings(source_sentences_list)
        negative_embeddings = self.get_embeddings(negative_sentence_list)

        for src_embedding, trg_sentences, neg_embedding in zip(src_embeddings, target_sentences_list,
                                                               negative_embeddings):
            for trg_sentence in trg_sentences:  # 平行语料，trg_sentence:set{eng, ens}
                self.cache.append(InputExample(texts=[trg_sentence], label=[src_embedding, neg_embedding]))
                # for each eng / ens , InputExample[text = eng1, label = eng1_src_embedding, eng1_neg_embedding]
                #                      InputExample[text = ens1, label = eng1_src_embedding, eng1_neg_embedding]
        # random.shuffle(self.cache)

    def next_entry(self, data_idx):  # self.datasets_iterator，len为语言种类数量，记录每种语言读取到了哪个位置，每次每种语言读100条
        source, target_sentences = self.datasets[data_idx][self.datasets_iterator[data_idx]]

        self.datasets_iterator[data_idx] += 1
        if self.datasets_iterator[data_idx] >= len(self.datasets[data_idx]):  # Restart iterator
            self.datasets_iterator[data_idx] = 0
            random.shuffle(self.datasets[data_idx])

        return source, target_sentences

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



class Parallel_Negative_Dataset(Dataset):
    """
    This dataset reader can be used to read-in parallel sentences, i.e., it reads in a file with tab-seperated sentences with the same
    sentence in different languages. For example, the file can look like this (EN\tDE\tES):
    hello world     hallo welt  hola mundo
    second sentence zweiter satz    segunda oración

    The sentence in the first column will be mapped to a sentence embedding using the given the embedder. For example,
    embedder is a mono-lingual sentence embedding method for English. The sentences in the other languages will also be
    mapped to this English sentence embedding.

    When getting a sample from the dataset, we get one sentence with the according sentence embedding for this sentence.

    teacher_model can be any class that implement an encode function. The encode function gets a list of sentences and
    returns a list of sentence embeddings
    """

    def __init__(self, student_model: SentenceTransformer, teacher_model: SentenceTransformer, batch_size: int = 8,
                 use_embedding_cache: bool = True):
        """
        Parallel sentences dataset reader to train student model given a teacher model
        :param student_model: Student sentence embedding model that should be trained
        :param teacher_model: Teacher model, that provides the sentence embeddings for the first column in the dataset file
        """
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.datasets = []
        # self.positive_datasets = []
        # self.negative_datasets = []
        self.datasets_iterator = []
        self.datasets_tokenized = []
        self.dataset_indices = []
        self.copy_dataset_indices = []
        self.cache = []
        self.batch_size = batch_size
        self.use_embedding_cache = use_embedding_cache
        self.embedding_cache = {}
        self.num_sentences = 0

    def load_data(self, filepath: str, weight: int = 100, max_sentences: int = None, max_sentence_length: int = 128):
        """
        Reads in a tab-seperated .txt/.csv/.tsv or .gz file. The different columns contain the different translations of the sentence in the first column

        :param filepath: Filepath to the file
        :param weight: If more than one dataset is loaded with load_data: With which frequency should data be sampled from this dataset?
        :param max_sentences: Max number of lines to be read from filepath
        :param max_sentence_length: Skip the example if one of the sentences is has more characters than max_sentence_length
        :param batch_size: Size for encoding parallel sentences
        :return:
        """

        logger.info("Load " + filepath)
        parallel_sentences = []

        with gzip.open(filepath, 'rt', encoding='utf8') if filepath.endswith('.gz') else open(filepath,
                                                                                              encoding='utf8') as fIn:
            count = 0
            for line in fIn:
                sentences = line.strip().split("\t")
                if max_sentence_length is not None and max_sentence_length > 0 and max(
                        [len(sent) for sent in sentences]) > max_sentence_length:
                    continue

                parallel_sentences.append(sentences)
                count += 1
                if max_sentences is not None and max_sentences > 0 and count >= max_sentences:
                    break
        # for sentence_idx, sent in enumerate(parallel_sentences):
        #     if sentence_idx < len(parallel_sentences) - 1:
        #         negative_idx = sentence_idx + 1
        #         ss = parallel_sentences[negative_idx][0]
        #     else:
        #         ss = parallel_sentences[0][0]
        #     parallel_sentences[sentence_idx].append(ss)
        self.add_dataset(parallel_sentences, weight=weight, max_sentences=max_sentences,
                         max_sentence_length=max_sentence_length)

    def add_dataset(self, parallel_sentences: List[List[str]], weight: int = 100, max_sentences: int = None,
                    max_sentence_length: int = 128):
        sentences_map = {}
        # negative_map = {}
        for sentences in parallel_sentences:
            if max_sentence_length is not None and max_sentence_length > 0 and max(
                    [len(sent) for sent in sentences]) > max_sentence_length:
                continue

            source_sentence = sentences[0]
            # negative_sentence = sentences[-1]
            if source_sentence not in sentences_map:
                sentences_map[source_sentence] = set()

            for sent in sentences:
                sentences_map[source_sentence].add(sent)
            # sentences_map[source_sentence] = list(sentences_map[source_sentence])
            # sentences_map[source_sentence].append(negative_sentence)
            # negative_map[source_sentence] =

            if max_sentences is not None and max_sentences > 0 and len(sentences_map) >= max_sentences:
                break

        if len(sentences_map) == 0:
            return

        self.num_sentences += sum([len(sentences_map[sent]) for sent in sentences_map])

        # assert len(self.positive_datasets) == len(self.negative_datasets)

        dataset_id = len(self.datasets)

        self.datasets.append(list(sentences_map.items()))
        self.datasets_iterator.append(0)
        # dataset_indices : 700长度，每个语言100，一共七种语言
        self.dataset_indices.extend([dataset_id] * weight)

    def generate_data(self):
        source_sentences_list = []
        target_sentences_list = []
        negative_sentence_list = []
        # TODO
        for data_idx in self.dataset_indices:
            src_sentence, trg_sentences = self.next_entry(data_idx)
            source_sentences_list.append(src_sentence)
            target_sentences_list.append(trg_sentences)
        # for idx, ss in enumerate(source_sentences_list):
        #     if idx < len(source_sentences_list) - 1:
        #         negative_sentence_list.append(source_sentences_list[idx + 1])
        #     else:
        #         negative_sentence_list.append(source_sentences_list[0])

        # Generate embeddings
        src_embeddings = self.get_embeddings(source_sentences_list)
        # negative_embeddings = self.get_embeddings(negative_sentence_list)

        for src_embedding, trg_sentences in zip(src_embeddings, target_sentences_list):
            for trg_sentence in trg_sentences:  # 平行语料，trg_sentence:set{eng, ens}
                self.cache.append(InputExample(texts=[trg_sentence], label=src_embedding))
                # for each eng / ens , InputExample[text = eng1, label = eng1_src_embedding, eng1_neg_embedding]
                #                      InputExample[text = ens1, label = eng1_src_embedding, eng1_neg_embedding]
        random.shuffle(self.cache)

    def next_entry(self, data_idx):  # self.datasets_iterator，len为语言种类数量，记录每种语言读取到了哪个位置，每次每种语言读100条
        source, target_sentences = self.datasets[data_idx][self.datasets_iterator[data_idx]]

        self.datasets_iterator[data_idx] += 1
        if self.datasets_iterator[data_idx] >= len(self.datasets[data_idx]):  # Restart iterator
            self.datasets_iterator[data_idx] = 0
            random.shuffle(self.datasets[data_idx])

        return source, target_sentences

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