import numpy as np
from abc import ABC, abstractmethod

class Model(ABC):

  def __init__(self) -> None:
    pass

  @abstractmethod
  def run_inference(self, input: np.ndarray) -> float:
    print("Inference Input: ", input)
    pass

  @abstractmethod
  def calculate_error(self, input: np.ndarray) -> float:
    print("Error Input: ", input)
    pass

# if __name__ == "__main__":
#   model = Model()