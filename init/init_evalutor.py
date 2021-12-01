from evaluator.STS_evaluator import STS_evaluator
from evaluator.TED_STS_evaluator import TED_STS_evaluator

evaluator_dic = {
    "TED_STS_Evaluator": TED_STS_evaluator,
    "STS_Evaluator": STS_evaluator

}


def init_evaluator(config, mode, *args, **params):
    name = config.get(mode, "evaluator")

    if name in evaluator_dic:
        return evaluator_dic[name]
    else:
        raise NotImplementedError
