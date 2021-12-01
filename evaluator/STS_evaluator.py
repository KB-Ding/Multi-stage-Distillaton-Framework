import os
import gzip

import numpy as np
from runx.logx import logx
from evaluator.evaluate.embedding_similarity_evaluator import embedding_similarity_evaluator
from evaluator.evaluate.sequence_evaluator import sequence_evaluator
from utils.message_utils import infor_msg
import zipfile
import io

def STS_evaluator(config, mode):
    source_languages = ['en']
    target_languages = ['de', 'es', 'it', 'fr', 'ar', 'tr', 'nl']
    parallel_sentences_folder = os.path.join(config.get("data", "data_dir"), 'parallel-sentences/')
    os.makedirs(parallel_sentences_folder, exist_ok=True)
    evaluators = []  # evaluators has a list of different evaluator classes we call periodically

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
