import argparse
import os
import torch


from utils.message_utils import infor_msg, erro_msg, warning_msg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', help="specific config file", required=True)
    parser.add_argument('--gpu', '-g', help="single gpu id", required=True)
    parser.add_argument('--checkpoint', help="checkpoint file path", required=True)
    parser.add_argument('--logdir', type=str, help="logdir file path", default='/apdcephfs/share_1157269/karlding/clean_code/output_test')

    args = parser.parse_args()

    configFilePath = args.config

    gpu_mode = 0
    if args.gpu is None:
        print(erro_msg("Do not support cpu version, please use gpu"))
        raise NotImplementedError
    else:
        gpu_mode = 1
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    os.system("clear")

    from process.init_process import init_all
    from config.parser import create_config
    from process.test_process import test_process
    from runx.logx import logx
    import datetime
    import pytz

    config = create_config(configFilePath)
    cur_time = datetime.datetime.now(pytz.timezone('PRC')).strftime("%Y_%m_%d__%H_%M_%S")

    identifier = config.get('logging', 'identifier')
    logdir = os.path.join(args.logdir, identifier, cur_time)

    logx.initialize(logdir=logdir, coolname=True, tensorboard=True)

    logx.msg(infor_msg(args))
    for section in config.sections():
        logx.msg(infor_msg(section))
        logx.msg(infor_msg(config.items(section)))

    # 检查 CUDA
    cuda = torch.cuda.is_available()
    logx.msg(infor_msg("CUDA available: %s" % str(cuda)))
    if not cuda and gpu_mode > 0:
        logx.msg(erro_msg("CUDA is not available but specific gpu id"))
        raise NotImplementedError

    # 初始化
    parameters = init_all(config, gpu_mode, args.checkpoint, "test")

    try:
        batch_size = config.getint("eval", "batch_size")
    except Exception as e:
        batch_size = config.getint("train", "batch_size")
        logx.msg(warning_msg(f"[eval] batch size has not been defined in config file, use [train] batch_size instead:{batch_size}"))

    try:
        reader_num = config.getint("eval", "reader_num")
    except Exception as e:
        reader_num = config.getint("train", "reader_num")
        logx.msg(warning_msg(f"[eval] reader num has not been defined in config file, use [train] reader num instead:{reader_num}"))


    test_process(parameters["model"], -1, config, -1,
                 parameters["test_evaluator"],
                 logdir,
                 'test')

