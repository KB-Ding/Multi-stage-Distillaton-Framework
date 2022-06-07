import argparse
import os
import torch

from utils.message_utils import infor_msg, erro_msg, warning_msg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', help="specific config file", required=True)
    parser.add_argument('--gpu', '-g', help="single gpu id", required=True)
    parser.add_argument('--checkpoint', help="checkpoint file path", required=True)
    parser.add_argument('--logdir', type=str, help="logdir file path", default='/apdcephfs/share_1157269/karlding/converts')

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
    from runx.logx import logx
    import datetime
    import pytz

    config = create_config(configFilePath)
    cur_time = datetime.datetime.now(pytz.timezone('PRC')).strftime("%Y_%m_%d__%H_%M_%S")

    identifier = config.get('logging', 'identifier')
    logdir = os.path.join(args.logdir, 'log', identifier, cur_time)
    output_path = os.path.join(args.logdir, 'model', identifier, cur_time)
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

    model = parameters["model"]
    model.eval()

    model_convert = model.module if hasattr(model, 'module') else model


    eval_path = output_path
    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)
    model_convert.model.save(output_path)