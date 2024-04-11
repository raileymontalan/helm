from typing import Callable, List, Dict
from functools import partial

from helm.common.hierarchical_logger import hlog
from helm.benchmark.adaptation.scenario_state import ScenarioState
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.adaptation.adapter_spec import AdapterSpec

from .metric import Metric, MetricResult
from .metric_name import MetricName
from .metric_service import MetricService
from .statistic import Stat

from .evaluate_reference_metrics import rouge_score as rouge_score_fn
from helm.benchmark.metrics.xlsum import rouge_scorer
from helm.benchmark.metrics.xlsum import tokenizers

import numpy as np
from sacrebleu.metrics import CHRF
from evaluate import load

class BhasaSummarizationMetric(Metric):
    """Summarization Metrics

    This class computes the following standard summarization metrics

    1. Rouge L
    """

    def __init__(self, language: str = 'en'):
        self.language: str = language
        self.rouge_fns = {
            "rouge_l": self._get_bhasa_rouge_function("rougeL"),
        }

    def _get_bhasa_rouge_function(self, rouge_type: str) -> Callable[[str, str], float]:
        if self.language == "th":
            scorer = rouge_scorer.RougeScorer(
                [rouge_type], 
                use_stemmer=True, 
                callable_tokenizer=tokenizers.ThaiTokenizer())
        else:
            scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=True)
        return partial(rouge_score_fn, scorer=scorer, rouge_type=rouge_type)

    def evaluate(
        self, scenario_state: ScenarioState, metric_service: MetricService, eval_cache_path: str, parallelism: int
    ) -> MetricResult:
        return super().evaluate(scenario_state, metric_service, eval_cache_path, parallelism=parallelism)

    def _compute_rouge(self, refs: List[str], pred: str) -> Dict[str, float]:
        metrics: Dict[str, float] = {}

        for metric, metric_fn in self.rouge_fns.items():
            metrics[metric] = np.max([metric_fn(ref, pred) for ref in refs])

        return metrics

    def _remove_braces(self, text: str) -> str:
        if text.startswith("{"):
            text = text[1:]
        if text.endswith("}"):
            text = text[:-1]
        return text

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        refs: List[str] = [self._remove_braces(ref.output.text) for ref in request_state.instance.references]
        inp: str = self._remove_braces(request_state.instance.input.text)

        assert request_state.result is not None
        pred: str = self._remove_braces(request_state.result.completions[0].text.strip())

        result: List[Stat] = []

        # Compute rouge metrics
        result.extend([Stat(MetricName(name)).add(float(val)) for name, val in self._compute_rouge(refs, pred).items()])

        return result
    
class BhasaMachineTranslationMetric(Metric):
    """Machine Translation Metrics

    This class computes the following standard machine translation metrics

    1. ChrF
    2. COMET
    """

    def __init__(self):
        self.chrf_scorer = CHRF(word_order=2)
        self.comet_scorer = load('comet') 

    def evaluate(
        self, scenario_state: ScenarioState, metric_service: MetricService, eval_cache_path: str, parallelism: int
    ) -> MetricResult:
        return super().evaluate(scenario_state, metric_service, eval_cache_path, parallelism=parallelism)

    def _compute_chrf(self, refs: List[str], pred: str) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        metrics['ChrF++'] = np.max([self.chrf_scorer.corpus_score(ref, pred).score for ref in refs])
        return metrics
    
    # def _compute_comet(self, refs: List[str], pred: str, inp: str) -> Dict[str, float]:
    #     metrics: Dict[str, float] = {}
    #     metrics['COMET'] = np.max([self.comet_scorer.compute(pred, ref, inp)['mean_score'] for ref in refs])
    #     return metrics

    def _remove_braces(self, text: str) -> str:
        if text.startswith("{"):
            text = text[1:]
        if text.endswith("}"):
            text = text[:-1]
        return text

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        refs: List[str] = [self._remove_braces(ref.output.text) for ref in request_state.instance.references]
        inp: str = self._remove_braces(request_state.instance.input.text)

        assert request_state.result is not None
        pred: str = self._remove_braces(request_state.result.completions[0].text.strip())

        result: List[Stat] = []

        # Compute ChrF++ metrics
        result.extend([Stat(MetricName(name)).add(float(val)) for name, val in self._compute_chrf(refs, pred).items()])

         # Compute COMET metrics
        # result.extend([Stat(MetricName(name)).add(float(val)) for name, val in self._compute_comet(refs, pred, inp).items()])

        return result