import torch

from formatter.router import route_formatter
from init.init_evalutor import init_evaluator
from init.init_model import get_model
from init.init_optimizer import init_optimizer
from runx.logx import logx
from utils.message_utils import warning_msg, infor_msg, erro_msg, correct_msg


def init_all(config, gpu_mode, checkpoint, mode, *args, **params):
    result = {}
    '''
    step1. model
    '''
    logx.msg(infor_msg('Begin to initialize models...'))
    model = get_model(config.get("model", "model_name"))(config, *args, **params)
    logx.msg(correct_msg(f'Get model: {config.get("model", "model_name")}'))
    optimizer = init_optimizer(model, config, *args, **params)
    trained_epoch = 0
    global_step = 0

    if 2 == gpu_mode:

        torch.distributed.init_process_group(backend="nccl")
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[local_rank],
                                                          output_device=local_rank,
                                                          find_unused_parameters=True)
        logx.msg(correct_msg('Train mode: distribute'))

    elif 1 == gpu_mode:
        model = model.cuda()
        logx.msg(correct_msg('Train mode: single gpu'))
    else:
        logx.msg(correct_msg('Train mode: cpu'))

    '''
    step2. checkpoints
    '''
    logx.msg(infor_msg("Loading checkpoint..."))
    try:
        if 2 == gpu_mode:
            parameters = torch.load(checkpoint, map_location=torch.device('cpu'))
            model.module.load_state_dict(parameters["model"])
        else:
            parameters = torch.load(checkpoint)
            model.load_state_dict(parameters["model"])

        trained_epoch = parameters["trained_epoch"]

        if mode == "train":
            if config.get("optim", "optimizer") == parameters["optimizer_name"]:
                optimizer.load_state_dict(parameters["optimizer"])
            else:
                logx.msg(warning_msg("Optimizer changed, do not load parameters of optimizer."))

            if "global_step" in parameters:
                global_step = parameters["global_step"]

    except Exception as e:
        information = "Cannot load checkpoint file with error:[ %s]" % str(e)
        if mode == "test":
            logx.msg(erro_msg(f'{information}'))
            logx.msg(erro_msg(f'{str(e)}'))
        else:
            logx.msg(warning_msg(f'{information}'))

    '''
    step3. formatter
    '''
    logx.msg(infor_msg("Use router to initialize formatter..."))
    if mode == "train":
        route_formatter(config, model, *args, **params)
    else:
        logx.msg(infor_msg("test mode, pass formatter..."))
        pass

    result["model"] = model
    result["trained_epoch"] = trained_epoch
    result["valid_evaluator"] = init_evaluator(config, 'eval')(config, 'eval')
    result["test_evaluator"] = init_evaluator(config, 'test')(config, 'test')

    if mode == 'train':
        result["optimizer"] = optimizer
        result["global_step"] = global_step

    logx.msg(infor_msg("Initialize done."))

    return result
