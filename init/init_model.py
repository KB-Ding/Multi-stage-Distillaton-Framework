from model.distillation.rec_bottle_distill import rec_bottle_distill
from model.distillation.rec_distill import rec_distill
from model.distillation.bottle_distill import bottle_distill
from model.multilingual.rec_bottle_mcl import rec_bottle_mcl

from model.multilingual.rec_mcl import rec_mcl
from model.multilingual.mse import mse

model_list = {
    # multilingual kd
    "mul_mse": mse,
    # before disillation
    "bottle_distill": bottle_distill,
    # distillation
    "rec_distill": rec_distill,
    "rec_bottle_distill": rec_bottle_distill,
    # after distillation
    "rec_bottle_mcl": rec_bottle_mcl,
    "rec_mcl": rec_mcl,
}

def get_model(model_name):
    if model_name in model_list.keys():
        return model_list[model_name]
    else:
        raise NotImplementedError