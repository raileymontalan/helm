from dataclasses import dataclass
from typing import List

from common.hierarchical_logger import hlog, htrack_block
from .scenario import ScenarioSpec, create_scenario
from .adapter import AdapterSpec, Adapter
from .executor import ExecutionSpec, Executor
from .metric import MetricSpec, create_metric


@dataclass(frozen=True)
class RunSpec:
    """
    Specifies how to do a single run, which gets a scenario, adapts it, and
    computes a list of metrics.
    """

    scenario: ScenarioSpec  # Which scenario
    adapter_spec: AdapterSpec  # Specifies how to adapt an instance into a set of requests
    metrics: List[MetricSpec]  # What to evaluate on


class Runner:
    """
    The main entry point for running the entire benchmark.  Mostly just
    dispatches to other classes.
    """

    def __init__(self, execution_spec: ExecutionSpec, run_specs: List[RunSpec]):
        self.executor = Executor(execution_spec)
        self.run_specs = run_specs

    def run_all(self):
        for run_spec in self.run_specs:
            self.run_one(run_spec)

    def run_one(self, run_spec: RunSpec):
        # Load the scenario
        scenario = create_scenario(run_spec.scenario)

        # Adaptation
        adapter = Adapter(run_spec.adapter_spec)
        scenario_state = adapter.adapt(scenario)

        # Execution
        scenario_state = self.executor.execute(scenario_state)

        # Apply the metrics
        metrics = [create_metric(metric) for metric in run_spec.metrics]
        hlog(f"{len(metrics)} metrics")
        stats = []
        for metric in metrics:
            stats.extend(metric.evaluate(scenario_state))

        # Print out stats
        with htrack_block("Stats"):
            for stat in stats:
                hlog(stat)
