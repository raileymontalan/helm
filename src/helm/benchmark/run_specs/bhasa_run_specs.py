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

def get_bhasa_adapter_spec(
    global_prefix: str = "",
    global_suffix: str = "",
    instructions: str = "",
    input_prefix: str = "",
    input_suffix: str = "",
    output_prefix: str = "",
    output_suffix: str = "",
    max_train_instances: int = 5,
    num_outputs: int = 1,
    max_tokens: int = 100,
    temperature: float = 0.0,
) -> AdapterSpec:
    """
    [instructions]

    [input_noun]: [input]
    [output_noun]: [output]

    [input_noun]: [input]
    [output_noun]:
    """

    def format_instructions(instructions: str) -> str:
        if len(instructions) > 0:
            instructions += "\n"
        return instructions

    return AdapterSpec(
        method=ADAPT_GENERATION,
        global_prefix=global_prefix,
        global_suffix=global_suffix,
        instructions=format_instructions(instructions),
        input_prefix=input_prefix,
        input_suffix=input_suffix,
        output_prefix=output_prefix,
        output_suffix=output_suffix,
        max_train_instances=max_train_instances,
        num_outputs=num_outputs,
        max_tokens=max_tokens,
        temperature=temperature,
    )

# NLU
@run_spec_function("xquad_qa_vi_bhasa_adapter")
def get_xquad_qa_vi_bhasa_adapter_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.bhasa_scenario.XQuAD_QA_VI_Scenario"
    )

    adapter_spec = get_bhasa_adapter_spec(
        instructions="Bạn sẽ được cho một đoạn văn và một câu hỏi.\nTrả lời câu hỏi bằng cách trích xuất câu trả lời từ đoạn văn.",
        input_suffix="\n",
        output_prefix="Câu trả lời: ",
        output_suffix="\n",
    )

    return RunSpec(
        name="xquad_qa_vi_bhasa_adapter",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_f1_metric_specs(),
        groups=["bhasa_nlu"],
    )

@run_spec_function("xquad_qa_vi")
def get_xquad_qa_vi_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.bhasa_scenario.XQuAD_QA_VI_Scenario"
    )

    adapter_spec = get_generation_adapter_spec(
        instructions="Bạn sẽ được cho một đoạn văn và một câu hỏi.\nTrả lời câu hỏi bằng cách trích xuất câu trả lời từ đoạn văn.",
        output_noun="Câu trả lời",
        max_train_instances=5,
    )

    return RunSpec(
        name="xquad_qa_vi",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_f1_metric_specs(),
        groups=["bhasa_nlu"],
    )

@run_spec_function("xquad_qa_th")
def get_xquad_qa_th_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.bhasa_scenario.XQuAD_QA_TH_Scenario"
    )

    adapter_spec = get_generation_adapter_spec(
        instructions="คุณจะได้รับข้อความและคําถาม กรุณาตอบคําถาม\nโดยแยกคําตอบจากข้อความ",
        output_noun="คําตอบ",
        max_train_instances=5,
    )

    return RunSpec(
        name="xquad_qa_th",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_f1_metric_specs(),
        groups=["bhasa_nlu"],
    )

@run_spec_function("uit_vsfc_sa_vi")
def get_uit_vsfc_sa_vi_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.bhasa_scenario.UIT_VSFC_SA_VI_Scenario"
    )

    adapter_spec = get_generation_adapter_spec(
        instructions="Sắc thái của câu sau đây là gì?\nTrả lời với một từ duy nhất: Tích cực/Tiêu cực/Trung lập",
        input_noun="Câu",
        output_noun="Câu trả lời",
        max_train_instances=5,
    )

    return RunSpec(
        name="uit_vsfc_sa_vi",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_f1_metric_specs(),
        groups=["bhasa_nlu"],
    )

@run_spec_function("nusax_sa_id")
def get_nusax_sa_id_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.bhasa_scenario.NusaX_SA_ID_Scenario"
    )

    adapter_spec = get_generation_adapter_spec(
        instructions="Apa sentimen dari kalimat berikut ini?\nJawab dengan satu kata saja: Positif/Negatif/Netral",
        input_noun="Kalimat",
        output_noun="Jawaban",
        max_train_instances=5,
    )

    return RunSpec(
        name="nusax_sa_id",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_f1_metric_specs(),
        groups=["bhasa_nlu"],
    )

@run_spec_function("indicsentiment_sa_ta")
def get_indicsentiment_sa_ta_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.bhasa_scenario.IndicSentiment_SA_TA_Scenario"
    )

    adapter_spec = get_generation_adapter_spec(
        instructions="பின்வரும் வாக்கியத்தில் வெளிப்படுத்தப்படும் உணர்வு எது?\nஒரு சொல்லில் மட்டும் பதிலளிக்கவும்:\n- நேர்மறை\n- எதிர்மறை",
        input_noun="வாக்கியம்",
        output_noun="பதில்",
        max_train_instances=5,
    )

    return RunSpec(
        name="indicsentiment_sa_ta",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["bhasa_nlu"],
    )

@run_spec_function("mlhsd_td_id")
def get_mlhsd_td_id_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.bhasa_scenario.MLHSD_TD_ID_Scenario"
    )

    adapter_spec = get_generation_adapter_spec(
        instructions="Anda adalah pendeteksi ujaran kebencian.\nDefinisi dari labelnya adalah sebagai berikut:\nBersih: Tidak ada ujaran kebencian\nKasar: Ada ujaran kebencian dan kata-kata kasar,namun tidak menyerang pihak tertentu\nBenci: Ada ujaran kebencian langsung atau serangan terhadap pihak tertentu\nKetika diberikan sebuah kalimat, klasifikasikan dengan satu kata:\nBersih/Kasar/Benci",
        input_noun="Kalimat",
        max_train_instances=5,
    )

    return RunSpec(
        name="mlhsd_td_id",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_f1_metric_specs(),
        groups=["bhasa_nlu"],
    )

@run_spec_function("vihsd_td_vi")
def get_vihsd_td_vi_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.bhasa_scenario.ViHSD_TD_VI_Scenario"
    )

    adapter_spec = get_generation_adapter_spec(
        instructions="Bạn là máy phát hiện phát ngôn thù ghét.\nCác nhãn được định nghĩa như sau:\nSạch: Không quấy rối\nCông kích: Bao gồm quấy rối và thậm chí chửi thề, nhưng không tấn công bất kì đối tượng cụ thể nào\nThù ghét: Trực tiếp quấy rối hay lăng mạ một đối tượng cụ thể\nCho câu dưới đây, hãy phân loại nó với 1 từ duy nhất:\nSạch/Công kích/Thù ghét",
        input_noun="Câu",
        max_train_instances=5,
    )

    return RunSpec(
        name="vihsd_td_vi",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_f1_metric_specs(),
        groups=["bhasa_nlu"],
    )

@run_spec_function("thai_toxicity_tweets_td_th")
def get_thai_toxicity_tweets_td_th_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.bhasa_scenario.Thai_Toxicity_Tweets_TD_TH_Scenario"
    )

    adapter_spec = get_generation_adapter_spec(
        instructions="คุณเป็นเครื่องตรวจจับข้อความที่เป็นพิษ\nข้อความเป็นพิษถ้ามันทําร้าย\nสร้างความเสียหายหรือสื่อเจตนาเชิงลบถึงบุคคลหรือกลุ่มบุคคล\nคุณควรคํานึงถึงการประชดประชันด้วย\nเมื่อได้รับข้อความ ให้ตอบด้วยตัวอักษร Y หากเป็นพิษ และ N หากไม่เป็นพิษ",
        input_noun="ข้อความ",
        max_train_instances=5,
    )

    return RunSpec(
        name="thai_toxicity_tweets_td_th",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_f1_metric_specs(),
        groups=["bhasa_nlu"],
    )


# NLG

# NLR

@run_spec_function("indonli_nli_id")
def get_indonli_nli_id_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.bhasa_scenario.IndoNLI_NLI_ID_Scenario"
    )

    adapter_spec = get_generation_adapter_spec(
        instructions="Anda akan diberikan dua kalimat, X dan Y.\nTentukan mana dari pernyataan berikut ini yang paling sesuai untuk kalimat X dan Y.\nA: Kalau X benar, maka Y juga harus benar.\nB: X bertentangan dengan Y.\nC: Ketika X benar, Y mungkin benar atau mungkin tidak benar.\nJawablah hanya dengan menggunakan satu huruf A, B atau C.",
        output_noun="Jawaban",
        max_train_instances=5,
    )

    return RunSpec(
        name="indonli_nli_id",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_f1_metric_specs(),
        groups=["bhasa_nlr"],
    )

@run_spec_function("xnli_nli_vi")
def get_xnli_nli_vi_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.bhasa_scenario.XNLI_NLI_VI_Scenario"
    )

    adapter_spec = get_generation_adapter_spec(
        instructions="Bạn sẽ được cho hai câu, X và Y.\nXác định câu nào sau đây là câu phù hợp nhất cho câu X và Y.\nA: Nếu X đúng thì Y phải đúng.\nB: X mâu thuẫn với Y.\nC: Khi X đúng, Y có thể đúng hoặc không đúng.\nTrả lời với một chữ cái duy nhất A, B, hoặc C.",
        output_noun="Câu trả lời",
        max_train_instances=5,
    )

    return RunSpec(
        name="xnli_nli_vi",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_f1_metric_specs(),
        groups=["bhasa_nlr"],
    )

@run_spec_function("xnli_nli_th")
def get_xnli_nli_th_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.bhasa_scenario.XNLI_NLI_TH_Scenario"
    )

    adapter_spec = get_generation_adapter_spec(
        instructions="คุณจะได้รับสองข้อความ X และ Y\nกรุณาพิจารณาว่า ข้อความใดต่อไปนี้ใช้กับข้อความ X และ Y ได้ดีที่สุด\nA: ถ้า X เป็นจริง Y จะต้องเป็นจริง\nB: X ขัดแย้งกับ Y\nC: เมื่อ X เป็นจริง Y อาจเป็นจริงหรือไม่ก็ได้\nกรุณาตอบด้วยตัวอักษร A, B หรือ C ตัวเดียวเท่านั้น",
        output_noun="คำตอบ",
        max_train_instances=5,
    )

    return RunSpec(
        name="xnli_nli_th",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_f1_metric_specs(),
        groups=["bhasa_nlr"],
    )


# LD

@run_spec_function("lindsea_mp_id")
def get_lindsea_mp_id_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.bhasa_scenario.LINDSEA_MP_ID_Scenario"
    )

    adapter_spec = get_multiple_choice_adapter_spec(
        method=ADAPT_MULTIPLE_CHOICE_JOINT,
        instructions="System Prompt:\nAnda adalah seorang ahli bahasa Indonesia\nHuman Prompt:Kalimat mana yang lebih mungkin?\nJawablah dengan menggunakan A atau B saja.",
        input_noun=None,
        output_noun="Jawaban",
        max_train_instances=5,
    )

    return RunSpec(
        name="lindsea_mp_id",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["bhasa_ld"],
    )