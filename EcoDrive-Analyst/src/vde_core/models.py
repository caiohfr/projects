from dataclasses import dataclass
import numpy as np

@dataclass
class RoadLoad:
    f0: float
    f1: float
    f2: float
    mass: float

@dataclass
class Cycle:
    t: np.ndarray
    v: np.ndarray
