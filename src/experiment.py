import itertools
from copy import deepcopy
from typing import Dict, List, Any, Callable
import matplotlib.pyplot as plt
import numpy as np
from abc import ABC, abstractmethod
import csv
import os

class Saver(ABC):
    def __init__(self, path: str) -> None:
        self.path = path

    @abstractmethod
    def visit(self, experiment: 'Experiment') -> None:
        pass

class CsvSaver(Saver):
    def visit(self, experiment: 'Experiment') -> None:
        results = experiment.summary()
        
        if not results:
            return

        file_exists = os.path.isfile(self.path)
        
        with open(self.path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=results.keys())
            
            if not file_exists:
                writer.writeheader()
                
            writer.writerow(results)

class Experiment(ABC):
    def __init__(self, results_saver: Saver, save_path: str = "") -> None:
        self.results_saver = results_saver
        self.save_path = save_path or self.results_saver.path

    @abstractmethod
    def conduct(self, **kwargs) -> None:
        pass

    @abstractmethod
    def summary(self, **kwargs) -> Dict[str, Any]:
        pass

    def visit_saver(self) -> None:
        self.results_saver.visit(self)



