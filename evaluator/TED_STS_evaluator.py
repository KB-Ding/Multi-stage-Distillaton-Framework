import os
import gzip

import numpy as np
from runx.logx import logx
from evaluator.evaluate.embedding_similarity_evaluator import embedding_similarity_evaluator
from evaluator.evaluate.mse_evaluator import mse_evaluator
from evaluator.evaluate.translation_evaluator import translation_evaluator

from sentence_transformers import SentenceTransformer
from evaluator.evaluate.sequence_evaluator import sequence_evaluator
from utils.message_utils import infor_msg

def TED_STS_evaluator(config, mode):
    # hugging_dir = os.path.join(config.get("data", "teacher_path"))
    # word_embedding_model = models.Transformer(hugging_dir, max_seq_length=config.getint("data", "max_seq_length"),
    #                                           cache_dir=hugging_dir)  # !!!!!!
    # # Apply mean pooling to get one fixed sized sentence vector
    # pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    # teacher_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    # ver 2
    teacher_model = SentenceTransformer(config.get("data", "teacher_path"))

    source_languages = ['en']
    target_languages = ['de', 'es', 'it', 'fr', 'ar', 'tr', 'nl']
    parallel_sentences_folder = os.path.join(config.get("data", "data_dir"), 'parallel-sentences/')
    os.makedirs(parallel_sentences_folder, exist_ok=True)
    train_files = []
    dev_files = []
    files_to_create = []
    for source_lang in source_languages:
        for target_lang in target_languages:
            output_filename_train = os.path.join(parallel_sentences_folder, "TED2020-{}-{}-train.tsv.gz".format(source_lang, target_lang))
            output_filename_dev = os.path.join(parallel_sentences_folder, "TED2020-{}-{}-dev.tsv.gz".format(source_lang, target_lang))
            train_files.append(output_filename_train)
            dev_files.append(output_filename_dev)
            if not os.path.exists(output_filename_train) or not os.path.exists(output_filename_dev):
                files_to_create.append({'src_lang': source_lang, 'trg_lang': target_lang,
                                        'fTrain': gzip.open(output_filename_train, 'wt', encoding='utf8'),
                                        'fDev': gzip.open(output_filename_dev, 'wt', encoding='utf8'),
                                        'devCount': 0
                                        })

    evaluators = []  # evaluators has a list of different evaluator classes we call periodically

    for dev_file in dev_files:
        logx.msg(infor_msg("Create evaluator for " + dev_file))
        src_sentences = []
        trg_sentences = []
        with gzip.open(dev_file, 'rt', encoding='utf8') as fIn:
            for line in fIn:
                splits = line.strip().split('\t')
                if splits[0] != "" and splits[1] != "":
                    src_sentences.append(splits[0])
                    trg_sentences.append(splits[1])

        # Mean Squared Error (MSE) measures the (euclidean) distance between teacher and student embeddings
        dev_mse = mse_evaluator(src_sentences, trg_sentences, name=os.path.basename(dev_file),
                                          teacher_model=teacher_model, batch_size=config.getint(mode, "batch_size"))
        evaluators.append(dev_mse)

        # translation_evaluator computes the embeddings for all parallel sentences. It then check if the embedding of source[i] is the closest to target[i] out of all available target sentences
        dev_trans_acc = translation_evaluator(src_sentences, trg_sentences, name=os.path.basename(dev_file),
                                              batch_size=config.getint(mode, "batch_size"))
        evaluators.append(dev_trans_acc)


    #########
    import zipfile
    import io
    data_dir_cache_folder = config.get("data", "data_dir")
    sts_data = {}
    all_languages = list(set(list(source_languages) + list(target_languages)))
    sts_corpus = os.path.join(config.get("data", "data_dir"), 'STS2017-extended.zip')
    # Open the ZIP File of STS2017-extended.zip and check for which language combinations we have STS data
    with zipfile.ZipFile(sts_corpus) as zip:
        filelist = zip.namelist()
        # 添加绝对路径，与filepath保持一致
        filelist = [os.path.join(data_dir_cache_folder, l) for l in filelist]
        for i in range(len(all_languages)):
            for j in range(i, len(all_languages)):
                lang1 = all_languages[i]
                lang2 = all_languages[j]
                filepath = os.path.join(data_dir_cache_folder,
                                        'STS2017-extended/STS.{}-{}.txt'.format(lang1, lang2))
                if filepath not in filelist:
                    lang1, lang2 = lang2, lang1
                    filepath = os.path.join(data_dir_cache_folder,
                                            'STS2017-extended/STS.{}-{}.txt'.format(lang1, lang2))

                if filepath in filelist:
                    filename = os.path.basename(filepath)
                    sts_data[filename] = {'sentences1': [], 'sentences2': [], 'scores': []}

                    fIn = zip.open('STS2017-extended/STS.{}-{}.txt'.format(lang1, lang2))
                    for line in io.TextIOWrapper(fIn, 'utf8'):
                        sent1, sent2, score = line.strip().split("\t")
                        score = float(score)
                        sts_data[filename]['sentences1'].append(sent1)
                        sts_data[filename]['sentences2'].append(sent2)
                        sts_data[filename]['scores'].append(score)

    for filename, data in sts_data.items():
        test_evaluator = embedding_similarity_evaluator(data['sentences1'],
                                                      data['sentences2'],
                                                      data['scores'],
                                                      batch_size=config.getint(mode, "batch_size"),
                                                      name=filename,
                                                      show_progress_bar=False)

        evaluators.append(test_evaluator)

    return sequence_evaluator(evaluators, main_score_function=lambda scores: np.mean(scores))
