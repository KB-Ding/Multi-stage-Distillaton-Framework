from model.distillation.rec_bottle_distill import rec_bottle_distill
from model.distillation.rec_distill import rec_distill
from model.distillation.bottle_distill import bottle_distill
from model.multilingual.rec_bottle_mse import rec_bottle_mse
from model.multilingual.rec_bottle_bool import rec_bottle_bool
from model.multilingual.rec_bottle_ce import rec_bottle_ce
from model.multilingual.rec_bottle_mcl import rec_bottle_mcl

from model.multilingual.rec_mcl import rec_mcl
from model.multilingual.mse import mse
from model.single.align_bottle_mse import align_bottle_mse
from model.single.origin_single_bottle_mse import origin_single_bottle_mse
from model.single.pre_distilled import pre_distilled
from model.single.single_bottle_mse import single_bottle_mse

model_list = {
    # multilingual kd
    "mul_mse": mse,
    # before disillation
    "bottle_distill": bottle_distill,
    # distillation
    "rec_distill": rec_distill,
    "rec_bottle_distill": rec_bottle_distill,
    # after distillation
    "rec_bottle_mse": rec_bottle_mse,
    "rec_bottle_bool": rec_bottle_bool,
    "rec_bottle_ce": rec_bottle_ce,
    "rec_bottle_mcl": rec_bottle_mcl,
    "rec_mcl": rec_mcl,
    # singal
    "single_bottle_mse": single_bottle_mse,
    "origin_single_bottle_mse": origin_single_bottle_mse,
    "align_bottle_mse": align_bottle_mse,
    "pre_distilled": pre_distilled
}

def get_model(model_name):
    if model_name in model_list.keys():
        return model_list[model_name]
    else:
        raise NotImplementedError