"""''''''''''''''''''''''''''''''''''''''
'           # 模型训练与记录 #
'
'  训练模型，每个epoch储存checkpoint并在验证集进行验证
'  每'config.getint("output", "output_time")'个batch，输出loss信息
'
''''''''''''''''''''''''''''''''''''''"""

from runx.logx import logx
import torch
from torch.autograd import Variable

from timeit import default_timer as timer
from torch.utils.data import DataLoader

from init.init_lr_scheduler import init_lr_scheduler
from utils.message_utils import gen_time_str, warning_msg, infor_msg, erro_msg, epoch_msg, correct_msg
from init.init_dataset import dataset_list
from formatter.router import route_formatter
from formatter.router import get_formatter
from process.test_process import test_process


def checkpoint(model, optimizer, trained_epoch, config, global_step, metric):
    """
    储存checkpoint文件
    :param model:           model
    :param optimizer:       optimizer
    :param trained_epoch:   本次训练的epoch数，从1开始计数
    :param config:          config
    :param global_step:     全局训练轮次， 从0开始计数

    """
    if logx.rank0:
        model_to_save = model.module if hasattr(model, 'module') else model
        save_params = {
            "model": model_to_save.state_dict(),
            "optimizer_name": config.get("optim", "optimizer"),
            "optimizer": optimizer.state_dict(),
            "trained_epoch": trained_epoch,
            "global_step": global_step
        }

        try:
            logx.save_model(
                save_params,
                metric=metric,
                epoch=trained_epoch,
                higher_better=True,
                delete_old=config.get("optim", "delete_old"))
            logx.msg(correct_msg(f"epoch{trained_epoch} save done, file path:{str(logx.save_ckpt_fn)}"))

        except Exception as e:
            logx.msg(warning_msg("Cannot save models with error:[ %s], continue anyway" % str(e)))


def train_process(parameters, config, gpu_mode, do_test):
    """
    模型的详细训练过程
    """

    which_train = config.get("data", "train_dataset_type")
    train_data = dataset_list[which_train](config, "train")

    batch_size = config.getint("train", "batch_size")
    shuffle = config.getboolean("train", "shuffle")
    reader_num = config.getint("train", "reader_num")

    if 2 == gpu_mode:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        dataset = DataLoader(dataset=train_data,
                             batch_size=batch_size,
                             num_workers=reader_num,
                             collate_fn=get_formatter(config, "train"),
                             pin_memory=True,
                             sampler=train_sampler)
    else:
        dataset = DataLoader(dataset=train_data,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=reader_num,
                             collate_fn=get_formatter(config, "train"))
    # eval
    try:
        batch_size = config.getint("eval", "batch_size")
    except Exception as e:
        logx.msg(warning_msg(
            f"[eval] batch size has not been defined in config file, use [train] batch_size instead:{batch_size}"))
    try:
        reader_num = config.getint("eval", "reader_num")
    except Exception as e:
        logx.msg(warning_msg(
            f"[eval] reader num has not been defined in config file, use [train] reader num instead:{reader_num}"))

    # dataset
    epoch = config.getint("train", "epoch") + 1
    output_time = config.getint("output", "output_time")

    # 训练参数
    trained_epoch = parameters["trained_epoch"] + 1  # 初始化为 1
    model = parameters["model"]
    optimizer = parameters["optimizer"]
    global_step = parameters["global_step"]
    valid_evaluator = parameters["valid_evaluator"]
    test_evaluator = parameters["test_evaluator"]
    output_path = parameters["output_path"]

    # optimizer 参数
    exp_lr_scheduler = init_lr_scheduler(optimizer, config, config.getint("train", "epoch") * len(dataset), model=model)

    logx.msg(infor_msg("==============Training start....=============="))

    total_len = len(dataset)
    model.train()

    # 开始训练，epoch范围：[trained_epoch, config.train.epoch + 1)
    for epoch_num in range(trained_epoch, epoch):
        # #############一个epoch开始#############
        start_time = timer()
        current_epoch = epoch_num
        if 2 == gpu_mode:
            train_sampler.set_epoch(epoch_num)

        # total_loss = 0

        step = -1

        for step, data in enumerate(dataset):
            # ############# 一个step开始 #############
            for key in data.keys():
                # 将经过Formatter处理后的tensor变量封装为Variable
                if isinstance(data[key], torch.Tensor):
                    if gpu_mode > 0:
                        data[key] = Variable(data[key].cuda(non_blocking=True))
                    else:
                        data[key] = Variable(data[key])

            model.zero_grad()
            loss_dic = model(data)

            loss_value = loss_dic["total_loss"]
            loss_value.backward()

            if config.getfloat("optim", "max_grad_norm") > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.getfloat("optim", "max_grad_norm"))

            optimizer.step()
            if 'step' == config.get("optim", "update_scheduler") and exp_lr_scheduler is not None:
                exp_lr_scheduler.step()

            if step % output_time == 0:
                delta_t = timer() - start_time

                show_dic = {}
                for key, value in loss_dic.items():
                    show_dic[key] = str(value.item())

                logx.msg(epoch_msg(Epoch=current_epoch,
                                   Stage='train',
                                   Iterations="%d/%d" % (step + 1, total_len),
                                   Time_Usage="%s/%s" % (
                                       gen_time_str(delta_t),
                                       gen_time_str(delta_t * (total_len - step - 1) / (step + 1))
                                   ),
                                   # **show_dic
                                   ))

                if step != 0 and logx.rank0:
                    with torch.no_grad():
                        score = test_process(model, current_epoch, config, step,
                                                   valid_evaluator,
                                                   output_path, 'validate')
                        model.train()
                    # 每个epoch存一次 tensorboard 和 checkpoint,
                    # 这里的logx.metric只存tensorboard，不存入metrics.csv，csv文件只存validate/test阶段的指标结果
                    checkpoint(model, optimizer, current_epoch, config, global_step, score)
                    logx.metric('val', {'loss': float(loss_value.item()),
                                          'score': score}, current_epoch)

            global_step += 1

            # ############# 一个step结束 #############
        if 'epoch' == config.get("optim", "update_scheduler") and exp_lr_scheduler is not None:
            exp_lr_scheduler.step()

        # 最后一个step的输出
        delta_t = timer() - start_time
        logx.msg(epoch_msg(Epoch=current_epoch,
                           Stage='train',
                           Iterations="%d/%d" % (step + 1, total_len),
                           Time_Usage="%s/%s" % (
                               gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                           # Loss="%.3lf" % (total_loss / (step + 1)),
                           ))

        if logx.rank0:
            with torch.no_grad():
                score = test_process(model, current_epoch, config, step,
                                     valid_evaluator,
                                     output_path, 'validate')
                model.train()
            # 每个epoch存一次 tensorboard 和 checkpoint,
            # 这里的logx.metric只存tensorboard，不存入metrics.csv，csv文件只存validate/test阶段的指标结果
            checkpoint(model, optimizer, current_epoch, config, global_step, score)

        if step == -1:
            logx.msg(erro_msg("There is no data given to the model in this epoch, check your data."))
            raise NotImplementedError
        # #############一个epoch结束#############

    # 所有epoch结束
    with torch.no_grad():
        if do_test and logx.rank0:
            test_process(model, -1, config, -1,
                         test_evaluator,
                         output_path, 'test')