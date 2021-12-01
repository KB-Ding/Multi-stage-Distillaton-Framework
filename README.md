# Multi-stage Distillation Framework

Source code of [paper](https://github.com)

## 目录

* [运行方式](#运行方式)
* [框架结构](#框架结构)
* [配置文件](#配置文件)
* [模型列表](#模型列表)
* [运行环境](#运行环境)
* [致谢](#致谢)

## 运行方式

支持单gpu，多gpu的训练/验证与单gpu的测试

**单卡训练：**

```bash
python train.py --config config文件 -checkpoint checkpoint文件 --gpu gpu编号 --do_test --logdir 训练文件夹
```

``--config``：必要参数，给定模型配置文件位置

``--checkpoint``：可选参数，给定保存的checkpoint文件位置

``--gpu``：必要参数，给定想运行的gpu编号

``--do_test``：可选参数，若希望训练后在测试集上进行测试，则给定本参数，否则省略

``--logdir``：可选参数，训练日志以及checkpoint存储位置

------

**多卡训练：**

```bash
CUDA_VISIBLE_DEVICES = gpu列表 python -m torch.distributed.launch --nproc_per_node = gpu数量 --master_port 指定端口号 train.py --config config文件 --checkpoint checkpoint文件 --distributed --do_test --logdir 训练文件夹
```

``--CUDA_VISIBLE_DEVICES``：必要参数，给定gpu列表

``--nproc_per_node``：必要参数，给定每个节点启动进程数量，通常与训练使用的gpu数量一致

``--master_port``：必要参数，给定主节点的端口号

``--config``：必要参数，给定模型配置文件位置

``--checkpoint``：可选参数，给定保存的checkpoint文件位置

``--distributed``：必要参数，指定使用torch.distributed进行初始化

``--do_test``：可选参数，若希望训练后在测试集上进行测试，则给定本参数，否则省略

``--logdir``：可选参数，训练日志以及checkpoint存储位置

------

**单卡测试：**

```bash
python test.py --config config文件 -checkpoint checkpoint文件 --gpu gpu编号 --logdir 测试文件夹
```

``--config``：必要参数，给定模型配置文件位置

``--checkpoint``：可选参数，给定保存的checkpoint文件位置

``--gpu``：必要参数，给定想运行的gpu编号

``--logdir``：可选参数，测试日志以及结果存储位置

------

例子：

> ```python
> CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.launch \
> --nproc_per_node=8 \
> --master_port 29502 train.py \
> --config /apdcephfs/share_1157269/karlding/mul_sentence_transformers/config/multilingual/xlmr_rec_bottle_mcl.config \
> --do_test \
> --distributed \
> --logdir /apdcephfs/share_1157269/karlding/mul_output_train
> ```

## 框架结构

**config文件夹：** 配置文件列表，设置各个模型对应的训练/测试/验证参数与模型参数

**dataset文件夹：** 数据集文件列表，从训练/验证/测试数据文件中读取数据，构建为dataset数据集

**evaluator文件夹：** 模型的验证/测试过程

**formatter文件夹：** 构建dataloader之前的预处理过程，将dataset整理为每个训练batch

**init文件夹：** 对dataset，evaluator, formatter，model，optimizer，lr_shceduler根据config文件进行初始化

**model文件夹：** 模型实现过程

**optim_scheduler文件夹：** 若不使用torch内置的optimizer，在这里自己定义对应的optimizer与lr_scheduler

**process文件夹：** 初始化框架（加载checkpoint等）以及训练/验证/测试/的详细过程

**utils文件夹：** 一些工具代码，例如cos相似度，随机种子，logging信息格式等

## 配置文件

<u>*Stage 1*：</u>

[获取sentence-transformers处理后的模型](https://github.com/UKPLab/sentence-transformers)

<u>*Stage 2*：</u>

*Our Method on MiniLM：* Distill/minilm_bottle_distill.config

*Our Method on XLM-R：* Distill/xlmr_bottle_distill.config

<u>*Stage 3：*</u>

*Our Method on MiniLM：* Distill/minilm_rec_bottle_distill.config，Distill/minilm_rec_distill.config

*Our Method on XLM-R：* Distill/xlmr_rec_bottle_distill.config，Distill/xlmr_rec_distill.config

<u>Stage 4：</u>

*[Reimerts Method](https://arxiv.org/abs/2004.09813)：* multilingual/mse.config

*Our Method on MiniLM：* multilingual/minilm_rec_bottle_mcl.config，multilingual/minilm_rec_mcl.config		

*Our Method on XLM-R：* multilingual/xlmr_rec_bottle_mcl.config，multilingual/xlmr_rec_mcl.config

*消融实验：* multilingual/ablation_wo_all.config，multilingual/ablation_wo_recursive.config，multilingual/xlmr_rec_bottle_mse.config

*对比其他cl方式：* multilingual/xlmr_rec_bottle_ce.config，multilingual/xlmr_rec_bottle_bool.config

## 模型列表

```python
"mul_mse": mse, # Reimerts Method
# stage 2
"bottle_distill": bottle_distill, # 若使用bottleneck layer，对齐pretrained model的embedding层与初始化的bottleneck层
# stage 3
"rec_distill": rec_distill,# 保留所有embedding层，只做parameter recurrent
"rec_bottle_distill": rec_bottle_distill, # 同时使用parameter recurrent与bottleneck layer
# stage 4
"rec_bottle_mcl": rec_bottle_mcl, # 使用parameter recurrent，bottleneck layer，MCL
"rec_mcl": rec_mcl，# 保留所有embedding层，只使用parameter recurrent，MCL
"rec_bottle_mse": rec_bottle_mse,  # 使用parameter recurrent，bottleneck layer，MCL任务替换为MSE
"rec_bottle_bool": rec_bottle_bool,# 使用parameter recurrent，bottleneck layer，MCL任务替换为Bool
"rec_bottle_ce": rec_bottle_ce, # 使用parameter recurrent，bottleneck layer，MCL任务替换为CE
```

## 运行环境

请参考``requirements.txt``。

## 致谢

[pytorch-worker](https://github.com/haoxizhong/pytorch-worker)

[sentence-transformers](https://github.com/UKPLab/sentence-transformers)

[transformers](https://github.com/huggingface/transformers)
