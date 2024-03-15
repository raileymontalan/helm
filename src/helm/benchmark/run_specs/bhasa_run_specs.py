from typing import Any, Dict, List, Optional, Set

from helm.benchmark.adaptation.adapter_spec import (
    ADAPT_GENERATION,
    ADAPT_MULTIPLE_CHOICE_JOINT,
    ADAPT_MULTIPLE_CHOICE_SEPARATE_ORIGINAL,
    ADAPT_RANKING_BINARY,
    AdapterSpec,
)
from helm.benchmark.adaptation.adapters.binary_ranking_adapter import BinaryRankingAdapter
from helm.benchmark.adaptation.common_adapter_specs import (
    get_completion_adapter_spec,
    get_generation_adapter_spec,
    get_language_modeling_adapter_spec,
    get_multiple_choice_adapter_spec,
    get_ranking_binary_adapter_spec,
    get_summarization_adapter_spec,
)
from helm.benchmark.metrics.common_metric_specs import (
    get_basic_metric_specs,
    get_bias_metric_specs,
    get_classification_metric_specs,
    get_copyright_metric_specs,
    get_disinformation_metric_specs,
    get_exact_match_metric_specs,
    get_f1_metric_specs,
    get_generative_harms_metric_specs,
    get_language_modeling_metric_specs,
    get_numeracy_metric_specs,
    get_open_ended_generation_metric_specs,
    get_summarization_metric_specs,
    get_basic_generation_metric_specs,
    get_basic_reference_metric_specs,
    get_generic_metric_specs,
)
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.runner import get_benchmark_output_path
from helm.benchmark.scenarios.scenario import ScenarioSpec, get_scenario_cache_path
from helm.common.hierarchical_logger import hlog, htrack

@run_spec_function("uit_vsfc_sa_five_shot")
def get_uit_vsfc_sa_five_shot_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.bhasa_scenario.UITVSFCSentimentAnalysisScenario"
    )

    adapter_spec = get_generation_adapter_spec(
        input_noun="Passage",
        output_noun="Sentiment",
        max_train_instances=5,
    )

    return RunSpec(
        name="uit_vsfc_sa_five_shot",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs() + get_classification_metric_specs() + get_f1_metric_specs(),
        groups=["uit_vsfc_sa"],
    )

@run_spec_function("uit_vsfc_sa_one_shot")
def get_uit_vsfc_sa_one_shot_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.bhasa_scenario.UITVSFCSentimentAnalysisScenario"
    )

    adapter_spec = get_generation_adapter_spec(
        input_noun="Passage",
        output_noun="Sentiment",
        max_train_instances=1,
    )

    return RunSpec(
        name="uit_vsfc_sa_one_shot",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs() + get_classification_metric_specs() + get_f1_metric_specs(),
        groups=["uit_vsfc_sa"],
    )

@run_spec_function("uit_vsfc_sa_zero_shot")
def get_uit_vsfc_sa_zero_shot_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.bhasa_scenario.UITVSFCSentimentAnalysisScenario"
    )

    adapter_spec = get_generation_adapter_spec(
        input_noun="Passage",
        output_noun="Sentiment",
        max_train_instances=0,
    )

    return RunSpec(
        name="uit_vsfc_sa_zero_shot",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs() + get_classification_metric_specs() + get_f1_metric_specs(),
        groups=["uit_vsfc_sa"],
    )