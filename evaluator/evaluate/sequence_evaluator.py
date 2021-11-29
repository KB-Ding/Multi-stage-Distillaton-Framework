from evaluator.evaluate.basic_evaluator import basic_evaluator
from typing import Iterable

class sequence_evaluator(basic_evaluator):
    def __init__(self, evaluators: Iterable[basic_evaluator], main_score_function = lambda scores: scores[-1]):
        self.evaluators = evaluators
        self.main_score_function = main_score_function

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        scores = []
        for evaluator in self.evaluators:
            scores.append(evaluator(model, output_path, epoch, steps))

        return self.main_score_function(scores)
