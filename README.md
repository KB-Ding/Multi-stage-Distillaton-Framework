# Multi-stage Distillation Framework

The PyTorch implementation for our paper:

 [Multi-stage Distillation Framework for Cross-Lingual Semantic Similarity Matching](https://aclanthology.org/2022.findings-naacl.167.pdf)

## Usage

### Dataset

Parallel sentences corpus: You can get *TED2020* from [here](https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/datasets/ted2020.tsv.gz) and other datasets from [OPUS](https://opus.nlpl.eu/) .

Test datasets: *STS2017* and *STS2017-extend* can be obtained from [here](https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/datasets/STS2017-extended.zip).

### Train

- ***Single-GPU***

```bash
python train.py --config [config] -checkpoint [checkpoint] --gpu [gpu number] --do_test --logdir [log folder]
```

``[config]`` : Directory for the configuration file.

``[checkpoint]``: Training from the [checkpoint] file.

 ``[gpu]``: The GPU index.											      

 ``--do_test`` : Whether to do test after training.

`[log folder]`: The file directory where training logs and checkpoints are saved.

- ***Multi-GPU***

```bash
CUDA_VISIBLE_DEVICES = [gpu list] python -m torch.distributed.launch --nproc_per_node = [number] --master_port [port] train.py --config [config] --checkpoint [checkpoint] --distributed --do_test --logdir [log folder]
```

``[gpu list]``：list of GPUs.

``[number]``：The number of GPUs.

``[port]``：The port number of the master node.

``--distributed``：Specify to use torch.distributed for initialization.

* *An example*：

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.launch \
--nproc_per_node=8 \
--master_port 29502 train.py \
--config /xxx/config/multilingual/xlmr_rec_bottle_mcl.config \
--do_test \
--distributed \
--logdir /xxx/output_train
```

- ***Evaluate***

```bash
python test.py --config [config] -checkpoint [checkpoint] --gpu [gpu] --logdir [log folder]
```

## Directory structure

```shell
├── config # Configuration: Set hyperparameters.
│   ├── default.config
│   ├── ...
├── dataset # Preprocessing: Read data and build the dataset
│   ├── parallel_all_distill_dataset.py
│   ├── ...
├── evaluator # Evaluator for test/validation sets
│   ├── STS_evaluator.py
│   ├── ...
├── formatter # Format the dataloader: Build each batch
│   ├── __init__.py
│   ├── basic_formatter.py
│   ├── ...
├── init # Initialize according to the configuration
│   ├── __init__.py
│   ├── init_dataset.py
│   ├── ...
├── model # Implementation of models
│   ├── __init__.py
│   ├── ...
├── optim_scheduler # Implementation of optimizers and lr_schedulers
│   ├── basic_optimizer.py
│   └── ...
├── process # Loading checkpoints and models
│   ├── init_process.py
│   ├── test_process.py
│   └── train_process.py
├── sentence_transformers # sentence_transformers package
├── test.py
├── train.py
└── utils
    ├── converts.py
    ├── ...
```

## Configuration

- <u> Stage 1:</u>

You can get well-trained models from [here](https://www.sbert.net/) or write your own script based on multilingual/mse.config.

- <u> Stage 2:</u>

*Our Method on MiniLM：* Distill/minilm_bottle_distill.config

*Our Method on XLM-R：* Distill/xlmr_bottle_distill.config

- <u>Stage 3:</u>

*Our Method on MiniLM：* Distill/minilm_rec_bottle_distill.config，Distill/minilm_rec_distill.config

*Our Method on XLM-R：* Distill/xlmr_rec_bottle_distill.config，Distill/xlmr_rec_distill.config

- <u>Stage 4:</u>

*Our Method on MiniLM：* multilingual/minilm_rec_bottle_mcl.config，multilingual/minilm_rec_mcl.config	

*Our Method on XLM-R：* multilingual/xlmr_rec_bottle_mcl.config，multilingual/xlmr_rec_mcl.config



We provide the model configuration list in: /init/init_model.py:

```shell
# stage 1
"mul_mse": mse, # Multilingual KD.
# stage 2
"bottle_distill": bottle_distill, # If use bottleneck, align the bottleneck embedding layer with the assistant model.
# stage 3
"rec_distill": rec_distill,# Using parameter recurrent only.
"rec_bottle_distill": rec_bottle_distill, # Using parameter recurrent and bottleneck layer.
# stage 4
"rec_bottle_mcl": rec_bottle_mcl, # Using parameter recurrent, bottleneck layer, and MCL.
"rec_mcl": rec_mcl，# Using parameter recurrent and MCL.
```

## Requirements

Python 3.6

PyTorch 1.7.1

transformers 4.6.0

For other packages, please refer to the requirements.txt.

## Model Zoo

Coming soon

## Acknowledgement

Thanks to [sentence-transformers](https://github.com/UKPLab/sentence-transformers), [huggingface-transformers](https://github.com/huggingface/transformers), [pytorch-worker](https://github.com/haoxizhong/pytorch-worker), and [UER](https://github.com/dbiir/UER-py) for their open source code.

This work is supported by Peking University and Tencent Inc. If you use the code, please cite this paper:

<pre>
@inproceedings{ding-etal-2022-multi,
    title = "Multi-stage Distillation Framework for Cross-Lingual Semantic Similarity Matching",
    author = "Ding, Kunbo  and
      Liu, Weijie  and
      Fang, Yuejian  and
      Zhao, Zhe  and
      Ju, Qi  and
      Yang, Xuefeng  and
      Tian, Rong  and
      Tao, Zhu  and
      Liu, Haoyan  and
      Guo, Han  and
      Bai, Xingyu  and
      Mao, Weiquan  and
      Li, Yudong  and
      Guo, Weigang  and
      Wu, Taiqiang  and
      Sun, Ningyuan",
    booktitle = "Findings of the Association for Computational Linguistics: NAACL 2022",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-naacl.167",
    pages = "2171--2181",
}<pre>
