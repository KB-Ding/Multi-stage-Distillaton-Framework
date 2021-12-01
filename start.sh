CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.launch \
--nproc_per_node=8 \
--master_port 29502 train.py \
--config /apdcephfs/share_1157269/karlding/mul_sentence_transformers/config/Multilingual/albert_bottleneck_ver4.config \
--do_test \
--distributed \
--logdir /apdcephfs/share_1157269/karlding/mul_output_train