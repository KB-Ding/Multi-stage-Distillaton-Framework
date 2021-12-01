from init.init_formatter import init_formatter
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

formatter = {}


def route_formatter(config, model, *args, **params):
    formatter["train"] = init_formatter(config, model, *args, **params)

def get_formatter(config, task, *args, **params):
    def train_collate_fn(data):
        return formatter["train"].process(data, config, "train")
    return train_collate_fn
