from torch.optim import lr_scheduler

def init_lr_scheduler(optimizer, config, leng, *args, **params):
    lr_scheduler_type = config.get("optim", "lr_scheduler")

    if lr_scheduler_type == "Step":
        step_size = config.getint("optim", "step_size")
        gamma = config.getfloat("optim", "lr_multiplier")
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    elif lr_scheduler_type == "multilingual":
        param_model = params['model'].module if hasattr(params['model'], 'module') else params['model']
        scheduler = param_model.model._get_scheduler(optimizer,
                                       scheduler='WarmupLinear',
                                       warmup_steps= int(leng * config.getfloat("optim", "warmup_rate")),
                                       t_total=leng)
    else:
        raise NotImplementedError

    return scheduler
