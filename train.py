"""''''''''''''''''''''''''''''''''''''''
'           # 解析与初始化 #
'
'  解析命令行传入的config文件/gpu列表/checkpoint路径/do_test测试参数与config中的信息
'  对整体模型进行init_all初始化
'  根据初始化的parameters进行训练。
'
''''''''''''''''''''''''''''''''''''''"""
import argparse
import os
from config.parser import create_config
from utils.seed import set_seed
import torch
if __name__ == "__main__":

    # 处理命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', help="specific config file", required=True)
    parser.add_argument('--checkpoint', help="checkpoint file path")
    parser.add_argument('--logdir', type=str, help="logdir file path", default='/apdcephfs/share_1157269/karlding/clean_code/output_train')

    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument('--do_test', help="do test while training or not", action="store_true")
    parser.add_argument('--distributed', help="do train in multi-gpu", action="store_true")
    parser.add_argument('--gpu', '-g', help="single gpu id, if use --distributed, ignore this argument")
    parser.add_argument('--local_rank', help='local rank', default=0)

    args = parser.parse_args()

    configFilePath = args.config

    # 根据是否使用gpu与gpu编号建立gpu_list
    gpu_mode = 2
    if not args.distributed:
        if args.gpu is None:
            gpu_mode = 0
            print("Do not support cpu version, please use gpu")
            raise NotImplementedError
        else:
            set_seed(args.seed, for_multi_gpu=False)
            gpu_mode = 1
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    else:
        set_seed(args.seed, for_multi_gpu=True)

    # 构建 config 参数
    os.system("clear")
    config = create_config(configFilePath)

    # 导入框架包，防止os.environ["CUDA_VISIBLE_DEVICES"]失效，所有涉及gpu操作均在其后面
    from runx.logx import logx
    import torch
    from process.init_process import init_all
    from process.train_process import train_process
    import torch
    from utils.message_utils import warning_msg, infor_msg, erro_msg, correct_msg
    import datetime
    import pytz

    cur_time = datetime.datetime.now(pytz.timezone('PRC')).strftime("%Y_%m_%d__%H_%M_%S")

    identifier = config.get('logging', 'identifier')
    logdir = os.path.join(args.logdir, 'logs_train', identifier, cur_time)

    logx.initialize(logdir=logdir, coolname=True, tensorboard=True, global_rank=int(args.local_rank))

    logx.msg(infor_msg('logging time: ' + datetime.datetime.now(pytz.timezone('PRC')).strftime("%Y_%m_%d %H:%M:%S")))

    logx.msg(infor_msg(args))
    logx.msg(infor_msg(args.config)+': ')
    for section in config.sections():
        logx.msg(infor_msg(section))
        logx.msg(infor_msg(config.items(section)))

    logx.msg('default config : ')
    for section in config.default_config.sections():
        logx.msg(infor_msg(section))
        logx.msg(infor_msg(config.default_config.items(section)))

    # 检查 CUDA
    cuda = torch.cuda.is_available()
    logx.msg(infor_msg("CUDA available: %s" % str(cuda)))
    if not cuda and gpu_mode > 0:
        logx.msg(erro_msg("CUDA is not available but specific gpu id"))
        raise NotImplementedError

    # 初始化
    parameters = init_all(config, gpu_mode, args.checkpoint, "train")

    parameters['output_path'] = os.path.join(args.logdir, 'logs_dev', identifier, cur_time)
    # 根据do_test参数与初始化参数
    # 开始训练
    do_test = False
    if args.do_test:
        do_test = True
    train_process(parameters, config, gpu_mode, do_test)
