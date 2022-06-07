# Multi-stage Distillation Framework

The PyTorch implementation for our paper:

 [Multi-stage Distillation Framework for Cross-Lingual Semantic Similarity Matching](https://arxiv.org/)

## Usage

#### Dataset

Parallel sentences corpus: you can get *TED2020* from [here](https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/datasets/ted2020.tsv.gz) and other datasets from [OPUS](https://opus.nlpl.eu/) .

Test dsatasets: *STS2017* and *STS2017-extend* can be obtained from [here](https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/datasets/STS2017-extended.zip).

#### Train

- Single-GPU

```bash
python train.py --config [config] -checkpoint [checkpoint] --gpu [gpu number] --do_test --logdir [log folder]
```

``--config``: config file directory, ``--checkpoint``: checkpoint file directory, ``--gpu``: gpu number , ``--do_test``: Whether to test after training, ``--logdir``：The file directory where logs and checkpoints are saved

- Multi-GPU

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

* An example：

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.launch \
--nproc_per_node=8 \
--master_port 29502 train.py \
--config /xxx/config/multilingual/xlmr_rec_bottle_mcl.config \
--do_test \
--distributed \
--logdir /xxx/mul_output_train
```

**Evaluate**

```bash
python test.py --config config文件 -checkpoint checkpoint文件 --gpu gpu编号 --logdir 测试文件夹
```

``--config``：必要参数，给定模型配置文件位置

``--checkpoint``：可选参数，给定保存的checkpoint文件位置

``--gpu``：必要参数，给定想运行的gpu编号

``--logdir``：可选参数，测试日志以及结果存储位置

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

## Model Config

<u>*Stage 1*：</u>

You can get well-trained models from [here](https://github.com/UKPLab/sentence-transformers) or write your own script based on multilingual/mse.config.

<u>*Stage 2*：</u>

*Our Method on MiniLM：* Distill/minilm_bottle_distill.config

*Our Method on XLM-R：* Distill/xlmr_bottle_distill.config

<u>*Stage 3：*</u>

*Our Method on MiniLM：* Distill/minilm_rec_bottle_distill.config，Distill/minilm_rec_distill.config

*Our Method on XLM-R：* Distill/xlmr_rec_bottle_distill.config，Distill/xlmr_rec_distill.config

<u>*Stage 4：*</u>

*Our Method on MiniLM：* multilingual/minilm_rec_bottle_mcl.config，multilingual/minilm_rec_mcl.config	

*Our Method on XLM-R：* multilingual/xlmr_rec_bottle_mcl.config，multilingual/xlmr_rec_mcl.config



We provide the model config list in: /init/init_model.py:

```python
# stage 1
"mul_mse": mse, # Multilingual KD.
# stage 2
"bottle_distill": bottle_distill, # If use bottleneck,
  																# Align the bottleneck embedding layer with the PLM.
# stage 3
"rec_distill": rec_distill,# Using parameter recurrent only.
"rec_bottle_distill": rec_bottle_distill, # Using parameter recurrent and bottleneck layer.
# stage 4
"rec_bottle_mcl": rec_bottle_mcl, # Using parameter recurrent, bottleneck layer, and MCL.
"rec_mcl": rec_mcl，# Using parameter recurrent and MCL.
```

## Requirements

Python 3.6

pytorch 1.7.1

transformers 4.6.0

For other packages, please refer to requirements.txt

## Model zoo

Coming soon

## Acknowledgement

Thanks to [sentence-transformers](https://github.com/UKPLab/sentence-transformers), [huggingface-transformers](https://github.com/huggingface/transformers), [pytorch-worker](https://github.com/haoxizhong/pytorch-worker), and [UER](https://github.com/dbiir/UER-py) for their open source code.

This work is supported by Peking University and Tencent Inc. If you use our code, please cite this paper:

```latex
@inproceedings{kunbo2022multistage,
  title={Multi-stage Distillation Framework for Cross-Lingual Semantic Similarity Matching},
  author={Kunbo Ding and Weijie Liu and Yuejian Fang and Zhe Zhao and Qi Ju and Xuefeng Yang and Rong Tian and Tao Zhu and Haoyan Liu and Han Guo and Xingyu Bai and Weiquan Mao and Yudong Li and Weigang Guo and Taiqiang Wu and Ningyuan Sun},
  booktitle={Proceedings of NAACL 2022},
  year={2022}
}
```

