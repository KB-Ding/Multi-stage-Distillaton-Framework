import torch

from formatter.basic_formatter import basic_formatter
from utils.cos import cos_sim


class teacher_sim_formatter(basic_formatter):
    def __init__(self, config, model, *args, **params):
        super().__init__(config, model, *args, **params)

        self.model = model.module if hasattr(model, 'module') else model
        self.max_len = config.getint("data", "max_seq_length")

    def process(self, data, config, mode, *args, **params):

        num_texts = 2
        texts = [[] for _ in range(num_texts)]
        labels = []

        for example in data:
            for idx, text in enumerate(example.texts):
                texts[idx].append(text)

            labels.append(example.label)

        labels = torch.tensor(labels)

        scores = cos_sim(labels, labels)


        en_tokenized = self.model.tokenize(texts[0])
        mul_tokenized = self.model.tokenize(texts[1])

        return {'en_input_ids': en_tokenized['input_ids'],
                'en_attention_mask': en_tokenized['attention_mask'],
                'mul_input_ids': mul_tokenized['input_ids'],
                'mul_attention_mask': mul_tokenized['attention_mask'],
                'labels': labels,
                "scores": scores
                }
