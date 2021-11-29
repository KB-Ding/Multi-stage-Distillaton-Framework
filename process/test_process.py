import torch
from torch.autograd import Variable
from runx.logx import logx

from evaluator.evaluate.basic_evaluator import basic_evaluator
from utils.message_utils import infor_msg, warning_msg, erro_msg
import os


def eval_during_training(model_test, evaluator, output_path, epoch, steps, callback):
    eval_path = output_path
    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)
        eval_path = os.path.join(output_path, "eval")
        os.makedirs(eval_path, exist_ok=True)

    if evaluator is not None:
        score = evaluator(model=model_test, output_path=eval_path, epoch=epoch, steps=steps)
        if callback is not None:
            callback(score, epoch, steps)
        return score

def evaluate(model_test, evaluator: basic_evaluator, output_path: str = None):
    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)
    return evaluator(model=model_test, output_path=output_path)



def test_process(model, epoch, config, step, evaluator, output_path, mode):
    model.eval()
    model_test = model.module if hasattr(model, 'module') else model
    if 'test' == mode:
        score = evaluate(model_test, evaluator, output_path)
    else:
        score = eval_during_training(model_test, evaluator, output_path, epoch, step, None)
    return score

