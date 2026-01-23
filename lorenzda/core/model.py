# Create an abstract base class for models
# The base class should have methods to initialize the model, advance the model state in time, and
# compute the tangent linear model (TLM) if applicable.
from abc import ABC, abstractmethod

class AbsModel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def model(x0, t, *args, **kwargs):
        return None

    @abstractmethod
    def model_tlm(self, *args, **kwargs):
        return None

    @abstractmethod
    def advance(self, *args, **kwargs):
        return None

    @abstractmethod
    def advance_tlm(self, *args, **kwargs):
        pass