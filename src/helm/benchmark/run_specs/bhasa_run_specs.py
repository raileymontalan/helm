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

# NLU
@run_spec_function("xquad_vi_qa")
def get_xquad_vi_qa_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.bhasa_scenario.XQuAD_VI_QA_Scenario"
    )

    adapter_spec = get_generation_adapter_spec(
        instructions="Bạn sẽ được cho một đoạn văn và một câu hỏi.\nTrả lời câu hỏi bằng cách trích xuất câu trả lời từ đoạn văn.",
        output_noun="Câu trả lời",
        max_train_instances=5,
    )

    return RunSpec(
        name="xquad_vi_qa",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_f1_metric_specs(),
        groups=["bhasa_nlu"],
    )

@run_spec_function("xquad_th_qa")
def get_xquad_th_qa_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.bhasa_scenario.XQuAD_TH_QA_Scenario"
    )

    adapter_spec = get_generation_adapter_spec(
        instructions="คุณจะได้รับข้อความและคําถาม กรุณาตอบคําถาม\nโดยแยกคําตอบจากข้อความ",
        output_noun="คําตอบ",
        max_train_instances=5,
    )

    return RunSpec(
        name="xquad_th_qa",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_f1_metric_specs(),
        groups=["bhasa_nlu"],
    )

@run_spec_function("uit_vsfc_vi_sa")
def get_uit_vsfc_vi_sa_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.bhasa_scenario.UIT_VSFC_VI_SA_Scenario"
    )

    adapter_spec = get_generation_adapter_spec(
        instructions="Sắc thái của câu sau đây là gì?\nTrả lời với một từ duy nhất: Tích cực/Tiêu cực/Trung lập",
        input_noun="Câu",
        output_noun="Câu trả lời",
        max_train_instances=5,
    )

    return RunSpec(
        name="uit_vsfc_vi_sa",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_f1_metric_specs(),
        groups=["bhasa_nlu"],
    )

@run_spec_function("nusax_id_sa")
def get_nusax_id_sa_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.bhasa_scenario.NusaX_ID_SA_Scenario"
    )

    adapter_spec = get_generation_adapter_spec(
        instructions="Apa sentimen dari kalimat berikut ini?\nJawab dengan satu kata saja: Positif/Negatif/Netral",
        input_noun="Kalimat",
        output_noun="Jawaban",
        max_train_instances=5,
    )

    return RunSpec(
        name="nusax_id_sa",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_f1_metric_specs(),
        groups=["bhasa_nlu"],
    )

@run_spec_function("indicsentiment_ta_sa")
def get_indicsentiment_ta_sa_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.bhasa_scenario.IndicSentiment_TA_SA_Scenario"
    )

    adapter_spec = get_generation_adapter_spec(
        instructions="உங் களுக் கு ஒரு பத் தியும் ஒரு ேகள் வியும் தரப் படும். தரப் பட் ட பத் தியிலிருந் து ேகள் விக் கான பதிைலக் கண் டறBயவும்.  ேகள் விக் குப் பதிலளிக் கமுடியாவிட் டால் ஒற் ைற எழுத் து X இைனப் பதிலாக அளிக் கவும்.",
        input_noun="உைர",
        output_noun="பதில்",
        max_train_instances=5,
    )

    return RunSpec(
        name="indicsentiment_ta_sa",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["bhasa_nlu"],
    )

@run_spec_function("mlhsd_id_td")
def get_mlhsd_id_td_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.bhasa_scenario.MLHSD_ID_TD_Scenario"
    )

    adapter_spec = get_generation_adapter_spec(
        instructions="Anda adalah pendeteksi ujaran kebencian.\nDefinisi dari labelnya adalah sebagai berikut:\nBersih: Tidak ada ujaran kebencian\nKasar: Ada ujaran kebencian dan kata-kata kasar,namun tidak menyerang pihak tertentu\nBenci: Ada ujaran kebencian langsung atau serangan terhadap pihak tertentu\nKetika diberikan sebuah kalimat, klasifikasikan dengan satu kata:\nBersih/Kasar/Benci",
        input_noun="Kalimat",
        max_train_instances=5,
    )

    return RunSpec(
        name="mlhsd_id_td",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_f1_metric_specs(),
        groups=["bhasa_nlu"],
    )

@run_spec_function("vihsd_vi_td")
def get_vihsd_vi_td_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.bhasa_scenario.ViHSD_VI_TD_Scenario"
    )

    adapter_spec = get_generation_adapter_spec(
        instructions="Bạn là máy phát hiện phát ngôn thù ghét.\nCác nhãn được định nghĩa như sau:\nSạch: Không quấy rối\nCông kích: Bao gồm quấy rối và thậm chí chửi thề, nhưng không tấn công bất kì đối tượng cụ thể nào\nThù ghét: Trực tiếp quấy rối hay lăng mạ một đối tượng cụ thể\nCho câu dưới đây, hãy phân loại nó với 1 từ duy nhất:\nSạch/Công kích/Thù ghét",
        input_noun="Câu",
        max_train_instances=5,
    )

    return RunSpec(
        name="vihsd_vi_td",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_f1_metric_specs(),
        groups=["bhasa_nlu"],
    )

@run_spec_function("thai_toxicity_tweets_th_td")
def get_thai_toxicity_tweets_th_td_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.bhasa_scenario.Thai_Toxicity_Tweets_TH_TD_Scenario"
    )

    adapter_spec = get_generation_adapter_spec(
        instructions="คุณเป็นเครื่องตรวจจับข้อความที่เป็นพิษ\nข้อความเป็นพิษถ้ามันทําร้าย\nสร้างความเสียหายหรือสื่อเจตนาเชิงลบถึงบุคคลหรือกลุ่มบุคคล\nคุณควรคํานึงถึงการประชดประชันด้วย\nเมื่อได้รับข้อความ ให้ตอบด้วยตัวอักษร Y หากเป็นพิษ และ N หากไม่เป็นพิษ",
        input_noun="ข้อความ",
        max_train_instances=5,
    )

    return RunSpec(
        name="thai_toxicity_tweets_th_td",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_f1_metric_specs(),
        groups=["bhasa_nlu"],
    )


# NLG

# NLR

# LD