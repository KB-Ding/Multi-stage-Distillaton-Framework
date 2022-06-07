import csv
import os
from typing import List

from evaluator.evaluate.basic_evaluator import basic_evaluator


class mse_evaluator(basic_evaluator):

    def __init__(self, source_sentences: List[str], target_sentences: List[str], teacher_model = None, show_progress_bar: bool = False, batch_size: int = 32, name: str = '', write_csv: bool = True):
        self.source_embeddings = teacher_model.encode(source_sentences, show_progress_bar=show_progress_bar, batch_size=batch_size, convert_to_numpy=True)

        self.target_sentences = target_sentences
        self.show_progress_bar = show_progress_bar
        self.batch_size = batch_size
        self.name = name

        self.csv_file = "mse_evaluation_" + name + "_results.csv"
        self.csv_headers = ["epoch", "steps", "MSE"]
        self.write_csv = write_csv

    def __call__(self, model, output_path, epoch  = -1, steps = -1):


        target_embeddings = model.encode(self.target_sentences, show_progress_bar=self.show_progress_bar, batch_size=self.batch_size, convert_to_numpy=True)

        mse = ((self.source_embeddings - target_embeddings)**2).mean()
        mse *= 100

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, newline='', mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow([epoch, steps, mse])

        return -mse
