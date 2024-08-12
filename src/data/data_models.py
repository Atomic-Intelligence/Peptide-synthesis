from abc import ABC, abstractmethod

import polars as pl
from pydantic import BaseModel


class Data(BaseModel):
    clinical: pl.DataFrame
    peptides: pl.DataFrame

    class Config:
        arbitrary_types_allowed = True


class Processor(ABC):
    @abstractmethod
    def preprocess_data(self, data: Data) -> Data:
        pass

    @abstractmethod
    def postprocess_data(self, data: Data) -> Data:
        pass

