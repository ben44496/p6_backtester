import numpy as np
# from model import Model
from models.model import Model
import logging

class MyModel(Model):

  def __init__(self, n_estimators: int, max_depth: int):
    # super().__init__()
    logging.info(" ".join(["MyModel", str(n_estimators), str(max_depth)]))
    self.prev_error = 0


  def run_inference(self, input: np.ndarray) -> dict:
    output = {
      "sum": np.sum(input),
      "avg": np.average(input),
    }
    return output


  def calculate_error(self, input: np.ndarray, actual: np.ndarray) -> dict:
    self.prev_error += 1
    return {"error": self.prev_error}