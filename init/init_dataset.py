from dataset.parallel_all_emb_dataset import parallel_all_emb_dataset
from dataset.parallel_all_distill_dataset import parallel_all_distill_dataset
from dataset.parallel_mul_disitll_dataset import parallel_separate_dataset

dataset_list = {
    "parallel_separate_dataset": parallel_separate_dataset,
    "parallel_all_distill_dataset": parallel_all_distill_dataset,
    "parallel_all_emb_dataset": parallel_all_emb_dataset
}