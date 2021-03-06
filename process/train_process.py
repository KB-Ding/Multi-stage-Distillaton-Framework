from runx.logx import logx
import torch
from torch.autograd import Variable

from timeit import default_timer as timer
from torch.utils.data import DataLoader

from init.init_lr_scheduler import init_lr_scheduler
from utils.message_utils import gen_time_str, warning_msg, infor_msg, erro_msg, epoch_msg, correct_msg
from init.init_dataset import dataset_list
from formatter.router import get_formatter
from process.test_process import test_process


def checkpoint(model, optimizer, trained_epoch, config, global_step, metric):

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

    # parameters
    trained_epoch = parameters["trained_epoch"] + 1  # initialized to 1
    model = parameters["model"]
    optimizer = parameters["optimizer"]
    global_step = parameters["global_step"]
    valid_evaluator = parameters["valid_evaluator"]
    test_evaluator = parameters["test_evaluator"]
    output_path = parameters["output_path"]

    # optimizer parameter
    exp_lr_scheduler = init_lr_scheduler(optimizer, config, config.getint("train", "epoch") * len(dataset), model=model)

    logx.msg(infor_msg("==============Training start....=============="))

    total_len = len(dataset)
    model.train()

    # train start, epoch???[trained_epoch, config.train.epoch + 1)
    for epoch_num in range(trained_epoch, epoch):
        # ############# epoch start #############
        start_time = timer()
        current_epoch = epoch_num
        if 2 == gpu_mode:
            train_sampler.set_epoch(epoch_num)

        # total_loss = 0

        step = -1

        for step, data in enumerate(dataset):
            # ############# step start #############
            for key in data.keys():
                # Formatter-->tensor-->Variable
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
                    # each epoch -- > save tensorboard & checkpoint,
                    # logx.metric --> tensorboard???not --> metrics.csv
                    # metrics.csv only saves the metrics of validate/test stage
                    
                    checkpoint(model, optimizer, current_epoch, config, global_step, score)
                    logx.metric('val', {'loss': float(loss_value.item()),
                                          'score': score}, current_epoch)

            global_step += 1

            # ############# step end #############
        if 'epoch' == config.get("optim", "update_scheduler") and exp_lr_scheduler is not None:
            exp_lr_scheduler.step()

        # last step
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

            checkpoint(model, optimizer, current_epoch, config, global_step, score)

        if step == -1:
            logx.msg(erro_msg("There is no data given to the model in this epoch, check your data."))
            raise NotImplementedError
        # #############epoch end#############

    # end
    with torch.no_grad():
        if do_test and logx.rank0:
            test_process(model, -1, config, -1,
                         test_evaluator,
                         output_path, 'test')