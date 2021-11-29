CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch \
--nproc_per_node=4 \
--master_port 29501 train.py \
--config /apdcephfs/share_1157269/karlding/mul_sentence_transformers/config/Multilingual/mse.config \
--do_test \
--distributed \
--logdir /apdcephfs/share_1157269/karlding/output_train


CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch \
--nproc_per_node=4 \
--master_port 29502 train.py \
--config /apdcephfs/share_1157269/karlding/mul_sentence_transformers/config/Multilingual/mse_contrastive.config \
--do_test \
--distributed \
--logdir /apdcephfs/share_1157269/karlding/output_train
