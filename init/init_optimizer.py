import torch.optim as optim

from optim_scheduler.sentence_optimizer import get_sentence_opt


def init_optimizer(model, config, *args, **params):
    optimizer_type = config.get("optim", "optimizer")
    learning_rate = config.getfloat("optim", "learning_rate")
    if optimizer_type == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                               weight_decay=config.getfloat("optim", "weight_decay"))
    elif optimizer_type == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                              weight_decay=config.getfloat("optim", "weight_decay"))

    elif optimizer_type == "multilingual":
        optimizer = get_sentence_opt(model, config)

    else:
        raise NotImplementedError

    return optimizer
