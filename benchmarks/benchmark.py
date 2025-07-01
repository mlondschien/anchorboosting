import timeit
from abc import ABC, abstractmethod
from typing import Any

from lightgbm import LGBMModel

from anchorboosting import AnchorBooster

from .load import load_nyc_taxi


class Benchmark(ABC):
    # Measure wall time instead of CPU usage
    timer = timeit.default_timer

    name = ""

    params: Any
    param_names: Any

    def setup(self, *params):
        self._setup_data(params)
        self._setup_model(params)

    def _setup_data(self, params):
        idx = self.param_names.index("n")
        n = params[idx]
        self.X, self.Z, self.y, self.y_binary = load_nyc_taxi(n)

    @abstractmethod  # Required for asv to skip.
    def _setup_model(self, params):
        pass


class AnchorBoosterBenchmark(Benchmark):
    def __init__(self, n_jobs=2):
        self.n_jobs = n_jobs  # GitHub CI has two nodes.

    name = "AnchorBooster"
    param_names = ["objective", "n", "num_boost_round", "max_depth"]
    params = (["regression", "binary"], [100_000], [100], [3])

    def _setup_model(self, params):
        objective, _, num_boost_round, max_depth = params
        self.model = AnchorBooster(
            gamma=2,
            num_boost_round=num_boost_round,
            max_depth=max_depth,
            n_jobs=self.n_jobs,
            objective=objective,
        )

    def time_fit(self, *args):
        y = self.y_binary if self.model.objective == "binary" else self.y
        self.model.fit(self.X, y, self.Z)


class LGBMBenchmark(Benchmark):
    def __init__(self, n_jobs=2):
        self.n_jobs = n_jobs  # GitHub CI has two nodes.

    name = "LGBM Booster"
    param_names = ["objective", "n", "num_boost_round", "max_depth"]
    params = (["regression", "binary"], [100_000], [100], [3])

    def _setup_model(self, params):
        objective, _, num_boost_round, max_depth = params
        self.model = LGBMModel(
            num_boost_round=num_boost_round,
            max_depth=max_depth,
            objective=objective,
            n_jobs=self.n_jobs,
        )

    def time_fit(self, *args):
        y = self.y_binary if self.model.objective == "binary" else self.y
        self.model.fit(self.X, y)
