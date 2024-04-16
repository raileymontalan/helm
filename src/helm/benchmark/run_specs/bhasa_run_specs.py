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
    format_instructions,
    get_completion_adapter_spec,
    get_generation_adapter_spec,
    get_language_modeling_adapter_spec,
    get_multiple_choice_adapter_spec,
    get_ranking_binary_adapter_spec,
    get_machine_translation_adapter_spec,
    get_summarization_adapter_spec,
)
from helm.benchmark.adaptation.bhasa_adapter_specs import get_bhasa_adapter_spec
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
    get_machine_translation_metric_specs,
    get_numeracy_metric_specs,
    get_open_ended_generation_metric_specs,
    get_summarization_metric_specs,
    get_basic_generation_metric_specs,
    get_basic_reference_metric_specs,
    get_generic_metric_specs,
    get_bhasa_summarization_metric_specs,
    get_bhasa_machine_translation_metric_specs,
    get_bhasa_qa_metric_specs,
)
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.runner import get_benchmark_output_path
from helm.benchmark.scenarios.scenario import ScenarioSpec, get_scenario_cache_path
from helm.common.hierarchical_logger import hlog, htrack

# NLU
@run_spec_function("indicqa_qa_ta")
def get_indicqa_qa_ta_spec(zeroshot=False) -> RunSpec:
    max_train_instances = 5
    name = "indicqa_qa_ta"

    if zeroshot:
        name += ",zeroshot=True"
        max_train_instances = 0

    adapter_spec = get_generation_adapter_spec(
        instructions="உங்களுக்கு ஒரு பத்தியும் ஒரு கேள்வியும் தரப்படும். தரப்பட்ட பத்தியிலிருந்து கேள்விக்கான பதிலைக் கண்டறியவும். கேள்விக்குப் பதிலளிக்கமுடியாவிட்டால் ஒற்றை எழுத்து X இனைப் பதிலாக அளிக்கவும்.",
        output_noun="பதில்",
        max_train_instances=max_train_instances,
        max_tokens=50,
    )

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.bhasa_scenario.IndicQA_QA_TA_Scenario"
    )

    return RunSpec(
        name=name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_bhasa_qa_metric_specs(args={
            "language": 'ta',
        }),
        groups=["bhasa_nlu"],
    )

@run_spec_function("tydiqa_goldp_qa_id")
def get_tydiqa_goldp_qa_id_spec(zeroshot=False) -> RunSpec:
    max_train_instances = 5
    name = "tydiqa_goldp_qa_id"

    if zeroshot:
        name += ",zeroshot=True"
        max_train_instances = 0

    adapter_spec = get_generation_adapter_spec(
        instructions="Anda akan diberikan sebuah paragraf dan sebuah pertanyaan. Jawablah pertanyaannya dengan mengekstrak jawaban dari paragraf tersebut.",
        output_noun="Jawaban",
        max_train_instances=max_train_instances,
        max_tokens=50,
    )

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.bhasa_scenario.TyDiQA_GoldP_QA_ID_Scenario"
    )

    return RunSpec(
        name=name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_bhasa_qa_metric_specs(args={
            "language": 'id',
        }),
        groups=["bhasa_nlu"],
    )

xquad_prompts = {
    "th": {
        "instructions": "คุณจะได้รับข้อความและคำถาม กรุณาตอบคำถามโดยแยกคำตอบจากข้อความ",
        "output_noun": "คำตอบ",
    },
    "vi": {
        "instructions": "Bạn sẽ được cho một đoạn văn và một câu hỏi. Trả lời câu hỏi bằng cách trích xuất câu trả lời từ đoạn văn.",
        "output_noun": "Câu trả lời",
    },
}

def generate_xquad_run_spec(zeroshot=False, language="th"):
    max_train_instances = 5
    name = f"xquad_qa_{language}"

    if zeroshot:
        name += ",zeroshot=True"
        max_train_instances = 0
        
    adapter_spec = get_generation_adapter_spec(
        instructions=xquad_prompts[language]['instructions'],
        output_noun=xquad_prompts[language]['output_noun'],
        max_train_instances=max_train_instances,
        max_tokens=50,
    )

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.bhasa_scenario.XQuAD_QA_Scenario",
        args={
            "language": language,
        },
    )

    return RunSpec(
        name=name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_bhasa_qa_metric_specs(args={
            "language": language,
        }),
        groups=["bhasa_nlu"],
    )
    
@run_spec_function("xquad_qa_th")
def get_xquad_qa_th_spec(zeroshot=False) -> RunSpec:
    return generate_xquad_run_spec(zeroshot, 'th')

@run_spec_function("xquad_qa_vi")
def get_xquad_qa_vi_spec(zeroshot=False) -> RunSpec:
    return generate_xquad_run_spec(zeroshot, 'vi')

@run_spec_function("indicsentiment_sa_ta")
def get_indicsentiment_sa_ta_spec(zeroshot=False) -> RunSpec:
    max_train_instances = 5
    name = "indicsentiment_sa_ta"

    if zeroshot:
        name += ",zeroshot=True"
        max_train_instances = 0
        adapter_spec = get_bhasa_adapter_spec(
            instructions="பின்வரும் வாக்கியத்தில் வெளிப்படுத்தப்படும் உணர்வு எது?",
            input_noun="வாக்கியம்",
            input_suffix="ஒரு சொல்லில் மட்டும் பதிலளிக்கவும்:\n- நேர்மறை\n- எதிர்மறை",
            output_noun="பதில்",
            max_train_instances=max_train_instances,
        )

    else:
        adapter_spec = get_generation_adapter_spec(
            instructions="பின்வரும் வாக்கியத்தில் வெளிப்படுத்தப்படும் உணர்வு எது?\nஒரு சொல்லில் மட்டும் பதிலளிக்கவும்:\n- நேர்மறை\n- எதிர்மறை",
            input_noun="வாக்கியம்",
            output_noun="பதில்",
            max_train_instances=max_train_instances,
        )

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.bhasa_scenario.IndicSentiment_SA_TA_Scenario"
    )

    return RunSpec(
        name=name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_classification_metric_specs(),
        groups=["bhasa_nlu"],
    )

@run_spec_function("nusax_sa_id")
def get_nusax_sa_id_spec(zeroshot=False) -> RunSpec:
    max_train_instances = 5
    name = "nusax_sa_id"

    if zeroshot:
        name += ",zeroshot=True"
        max_train_instances = 0
        adapter_spec = get_bhasa_adapter_spec(
            instructions="Apa sentimen dari kalimat berikut ini?",
            input_noun="Kalimat",
            input_suffix="Jawab dengan satu kata saja:\n- Positif\n- Negatif\n- Netral",
            output_noun="Jawaban",
            max_train_instances=max_train_instances,
        )

    else:
        adapter_spec = get_generation_adapter_spec(
            instructions="Apa sentimen dari kalimat berikut ini?\nJawab dengan satu kata saja:\n- Positif\n- Negatif\n- Netral",
            input_noun="Kalimat",
            output_noun="Jawaban",
            max_train_instances=max_train_instances,
        )

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.bhasa_scenario.NusaX_SA_ID_Scenario"
    )

    return RunSpec(
        name=name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_f1_metric_specs(),
        groups=["bhasa_nlu"],
    )

@run_spec_function("wisesight_sa_th")
def get_wisesight_sa_th_spec(zeroshot=False) -> RunSpec:
    max_train_instances = 5
    name = "wisesight_sa_th"

    if zeroshot:
        name += ",zeroshot=True"
        max_train_instances = 0
        adapter_spec = get_bhasa_adapter_spec(
            instructions="อารมณ์ความรู้สึกของข้อความต่อไปนี้เป็นอย่างไร?",
            input_noun="ข้อความ",
            input_suffix="โปรดตอบโดยใช้คำเดียวเท่านั้น:\n- แง่บวก\n- แง่ลบ\n- เฉยๆ",
            output_noun="คำตอบ",
            max_train_instances=max_train_instances,
        )

    else:
        adapter_spec = get_generation_adapter_spec(
            instructions="อารมณ์ความรู้สึกของข้อความต่อไปนี้เป็นอย่างไร?\nโปรดตอบโดยใช้คำเดียวเท่านั้น:\n- แง่บวก\n- แง่ลบ\n- เฉยๆ",
            input_noun="ข้อความ",
            output_noun="คำตอบ",
            max_train_instances=max_train_instances,
        )

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.bhasa_scenario.Wisesight_SA_TH_Scenario"
    )

    return RunSpec(
        name=name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_f1_metric_specs(),
        groups=["bhasa_nlu"],
    )

@run_spec_function("uit_vsfc_sa_vi")
def get_uit_vsfc_sa_vi_spec(zeroshot=False) -> RunSpec:
    max_train_instances = 5
    name = "uit_vsfc_sa_vi"

    if zeroshot:
        name += ",zeroshot=True"
        max_train_instances = 0
        adapter_spec = get_bhasa_adapter_spec(
            instructions="Sắc thái của câu sau đây là gì?",
            input_noun="Câu",
            input_suffix="Trả lời với một từ duy nhất:\n- Tích cực\n- Tiêu cực\n- Trung lập",
            output_noun="Câu trả lời",
            max_train_instances=max_train_instances,
        )

    else:
        adapter_spec = get_generation_adapter_spec(
            instructions="Sắc thái của câu sau đây là gì?\nTrả lời với một từ duy nhất:\n- Tích cực\n- Tiêu cực\n- Trung lập",
            input_noun="Câu",
            output_noun="Câu trả lời",
            max_train_instances=max_train_instances,
        )

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.bhasa_scenario.UIT_VSFC_SA_VI_Scenario"
    )

    return RunSpec(
        name=name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_f1_metric_specs(),
        groups=["bhasa_nlu"],
    )

@run_spec_function("mlhsd_td_id")
def get_mlhsd_td_id_spec(zeroshot=False) -> RunSpec:
    max_train_instances = 5
    name = "mlhsd_td_id"

    if zeroshot:
        name += ",zeroshot=True"
        max_train_instances = 0
        adapter_spec = get_bhasa_adapter_spec(
            instructions="Anda adalah pendeteksi ujaran kebencian. Definisi dari labelnya adalah sebagai berikut:\nBersih: Tidak ada ujaran kebencian.\nKasar: Ada ujaran kebencian dan kata-kata kasar, namun tidak menyerang pihak tertentu.\nBenci: Ada ujaran kebencian langsung atau serangan terhadap pihak tertentu.\nBerdasarkan definisi labelnya, klasifikasikan kalimat berikut ini:",
            input_noun="Kalimat",
            input_suffix="Jawab dengan hanya A, B atau C.\nA: Bersih\nB: Kasar\nC: Benci",
            output_noun="Jawaban",
            max_train_instances=max_train_instances,
        )

    else:
        adapter_spec = get_generation_adapter_spec(
            instructions="Anda adalah pendeteksi ujaran kebencian. Definisi dari labelnya adalah sebagai berikut:\nBersih: Tidak ada ujaran kebencian.\nKasar: Ada ujaran kebencian dan kata-kata kasar, namun tidak menyerang pihak tertentu.\nBenci: Ada ujaran kebencian langsung atau serangan terhadap pihak tertentu.\nBerdasarkan definisi labelnya, klasifikasikan kalimat berikut ini.\nJawab dengan hanya A, B atau C.\nA: Bersih\nB: Kasar\nC: Benci",
            input_noun="Kalimat",
            output_noun="Jawaban",
            max_train_instances=max_train_instances,
        )

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.bhasa_scenario.MLHSD_TD_ID_Scenario"
    )

    return RunSpec(
        name=name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_f1_metric_specs(),
        groups=["bhasa_nlu"],
    )

@run_spec_function("thai_toxicity_tweets_td_th")
def get_thai_toxicity_tweets_td_th_spec(zeroshot=False) -> RunSpec:
    max_train_instances = 5
    name = "thai_toxicity_tweets_td_th"

    if zeroshot:
        name += ",zeroshot=True"
        max_train_instances = 0
    
    adapter_spec = get_generation_adapter_spec(
        instructions="คุณเป็นเครื่องตรวจจับข้อความที่เป็นพิษ ข้อความเป็นพิษถ้ามันทำร้าย สร้างความเสียหาย หรือสื่อเจตนาเชิงลบถึงบุคคลหรือกลุ่มบุคคล คุณควรคำนึงถึงการประชดประชันด้วย เมื่อได้รับข้อความ ให้ตอบด้วยตัวอักษร Y หากเป็นพิษ และ N หากไม่เป็นพิษ\nคำตอบ: ",
        input_noun="ข้อความ",
        output_noun="คำตอบ",
        max_train_instances=max_train_instances,
    )

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.bhasa_scenario.Thai_Toxicity_Tweets_TD_TH_Scenario"
    ) 

    return RunSpec(
        name=name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_f1_metric_specs(),
        groups=["bhasa_nlu"],
    )

@run_spec_function("vihsd_td_vi")
def get_vihsd_td_vi_spec(zeroshot=False) -> RunSpec:
    max_train_instances = 5
    name = "vihsd_td_vi"

    if zeroshot:
        name += ",zeroshot=True"
        max_train_instances = 0
        adapter_spec = get_bhasa_adapter_spec(
            instructions="Bạn là máy phát hiện phát ngôn thù ghét. Các nhãn được định nghĩa như sau:\nSạch: Không quấy rối.\nCông kích: Bao gồm quấy rối và thậm chí chửi thề, nhưng không tấn công bất kì đối tượng cụ thể nào.\nThù ghét: Trực tiếp quấy rối hay lăng mạ một đối tượng cụ thể.\nVới các định nghĩa của nhãn, hãy phân loại câu dưới đây:",
            input_noun="Câu",
            input_suffix="nChỉ trả lời bằng A, B hoặc C.\nA: Sạch\nB: Công kích\nC: Thù ghét",
            output_noun="Câu trả lời",
            max_train_instances=max_train_instances,
        )
    else:
        adapter_spec = get_generation_adapter_spec(
            instructions="Bạn là máy phát hiện phát ngôn thù ghét. Các nhãn được định nghĩa như sau:\nSạch: Không quấy rối.\nCông kích: Bao gồm quấy rối và thậm chí chửi thề, nhưng không tấn công bất kì đối tượng cụ thể nào.\nThù ghét: Trực tiếp quấy rối hay lăng mạ một đối tượng cụ thể.\nVới các định nghĩa của nhãn, hãy phân loại câu dưới đây.\nChỉ trả lời bằng A, B hoặc C.\nA: Sạch\nB: Công kích\nC: Thù ghét",
            input_noun="Câu",
            output_noun="Câu trả lời",
            max_train_instances=max_train_instances,
        )

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.bhasa_scenario.ViHSD_TD_VI_Scenario"
    )

    return RunSpec(
        name=name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_f1_metric_specs(),
        groups=["bhasa_nlu"],
    )

# NLG
flores_prompts = {
    "en_id": {
        "instructions": "Terjemahkan teks berikut ini ke dalam Bahasa Indonesia.",
        "input_noun": "Teks",
        "output_noun": "Terjemahan",
    },
    "en_ta": {
        "instructions": "பின்வரும் உரையைத் தமிழ் மொழிக்கு மொழிபெயர்க்கவும்.",
        "input_noun": "உரை",
        "output_noun": "மொழிபெயர்ப்பு",
    },
    "en_th": {
        "instructions": "กรุณาแปลข้อความต่อไปนี้เป็นภาษาไทย",
        "input_noun": "ข้อความ",
        "output_noun": "คำแปล",
    },
    "en_vi": {
        "instructions": "Dịch văn bản dưới đây sang Tiếng Việt.",
        "input_noun": "Văn bản",
        "output_noun": "Bản dịch",
    },
    "id_en": {
        "instructions": "Terjemahkan teks berikut ini ke dalam Bahasa Inggris.",
        "input_noun": "Teks",
        "output_noun": "Terjemahan",
    },
    "ta_en": {
        "instructions": "பின்வரும் உரையை ஆங்கில மொழிக்கு மொழிபெயர்க்கவும்.",
        "input_noun": "உரை",
        "output_noun": "மொழிபெயர்ப்பு",
    },
    "th_en": {
        "instructions": "กรุณาแปลข้อความต่อไปนี้เป็นภาษาอังกฤษ",
        "input_noun": "ข้อความ",
        "output_noun": "คำแปล",
    },
    "vi_en": {
        "instructions": "Dịch văn bản dưới đây sang Tiếng Anh.",
        "input_noun": "Văn bản",
        "output_noun": "Bản dịch",
    },
}

def generate_flores_run_spec(zeroshot=False, pair="en_id"):
    max_train_instances = 5
    name = f"flores_mt_{pair}"

    if zeroshot:
        name += ",zeroshot=True"
        max_train_instances = 0
        
    adapter_spec = get_generation_adapter_spec(
        instructions=flores_prompts[pair]['instructions'],
        input_noun=flores_prompts[pair]['input_noun'],
        output_noun=flores_prompts[pair]['output_noun'],
        max_tokens=128,
        max_train_instances=max_train_instances,
    )

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.bhasa_scenario.Flores_MT_Scenario",
        args={
            "pair": pair,
        },
    )

    return RunSpec(
        name=name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_bhasa_machine_translation_metric_specs(),
        groups=["bhasa_nlg"],
    )
    
@run_spec_function("flores_mt_en_id")
def get_flores_mt_en_id_spec(zeroshot=False) -> RunSpec:
    return generate_flores_run_spec(zeroshot, 'en_id')

@run_spec_function("flores_mt_en_ta")
def get_flores_mt_en_ta_spec(zeroshot=False) -> RunSpec:
    return generate_flores_run_spec(zeroshot, 'en_ta')

@run_spec_function("flores_mt_en_th")
def get_flores_mt_en_th_spec(zeroshot=False) -> RunSpec:
    return generate_flores_run_spec(zeroshot, 'en_th')

@run_spec_function("flores_mt_en_vi")
def get_flores_mt_en_vi_spec(zeroshot=False) -> RunSpec:
    return generate_flores_run_spec(zeroshot, 'en_vi')

@run_spec_function("flores_mt_id_en")
def get_flores_mt_id_en_spec(zeroshot=False) -> RunSpec:
    return generate_flores_run_spec(zeroshot, 'id_en')

@run_spec_function("flores_mt_ta_en")
def get_flores_mt_ta_en_spec(zeroshot=False) -> RunSpec:
    return generate_flores_run_spec(zeroshot, 'ta_en')

@run_spec_function("flores_mt_th_en")
def get_flores_mt_th_en_spec(zeroshot=False) -> RunSpec:
    return generate_flores_run_spec(zeroshot, 'th_en')

@run_spec_function("flores_mt_vi_en")
def get_flores_mt_vi_en_spec(zeroshot=False) -> RunSpec:
    return generate_flores_run_spec(zeroshot, 'vi_en')

xlsum_prompts = {
    "id": {
        "input_noun": "Artikel",
        "input_suffix": "Rangkumkan artikel Bahasa Indonesia ini dalam 1 atau 2 kalimat. Jawabannya harus ditulis dalam Bahasa Indonesia.",
        "output_noun": "Rangkuman",
    },
    "ta": {
       "input_noun": "கட்டுரை",
        "input_suffix": "இந்தத் தமிழ்க் கட்டுரைக்கு 1 அல்லது 2 வாக்கியங்களில் பொழிப்பு எழுதவும். பதில் தமிழ் மொழியில் இருக்கவேண்டும்.",
        "output_noun": "கட்டுரைப் பொழிப்பு",
    },
    "th": {
        "input_noun": "บทความ",
        "input_suffix": "กรุณาสรุปบทความภาษาไทยฉบับนี้ใน 1 หรือ 2 ประโยค คำตอบควรเป็นภาษาไทย",
        "output_noun": "บทสรุป",
    },
    "vi": {
        "input_noun": "Bài báo",
        "input_suffix": "Tóm tắt bài báo Tiếng Việt trên với 1 hay 2 câu. Câu trả lời nên được viết bằng tiếng Việt.",
        "output_noun": "Bản tóm tắt",
    }
}

def generate_xlsum_run_spec(zeroshot=False, language="id", device="gpu"):
    max_train_instances = 5
    name = f"xlsum_as_{language}"

    if zeroshot:
        name += ",zeroshot=True"
        max_train_instances = 0
        
    adapter_spec = get_bhasa_adapter_spec(
        input_noun=xlsum_prompts[language]['input_noun'],
        input_suffix=xlsum_prompts[language]['input_suffix'],
        output_noun=xlsum_prompts[language]['output_noun'],
        max_train_instances=max_train_instances,
        temperature = 0.3,
        max_tokens=128,
    )

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.bhasa_scenario.XLSum_AS_Scenario",
        args={
            "language": language,
        },
    )

    return RunSpec(
        name=name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_bhasa_summarization_metric_specs(args={
            "language": language
        }),
        groups=["bhasa_nlg"],
    )
    
@run_spec_function("xlsum_as_id")
def get_xlsum_as_id_spec(zeroshot=False) -> RunSpec:
    return generate_xlsum_run_spec(zeroshot, 'id')

@run_spec_function("xlsum_as_ta")
def get_xlsum_as_ta_spec(zeroshot=False) -> RunSpec:
    return generate_xlsum_run_spec(zeroshot, 'ta')

@run_spec_function("xlsum_as_th")
def get_xlsum_as_th_spec(zeroshot=False) -> RunSpec:
    return generate_xlsum_run_spec(zeroshot, 'th')

@run_spec_function("xlsum_as_vi")
def get_xlsum_as_vi_spec(zeroshot=False) -> RunSpec:
    return generate_xlsum_run_spec(zeroshot, 'vi')

# NLR

@run_spec_function("indicxnli_nli_ta")
def get_indicxnli_nli_ta_spe(zeroshot=False) -> RunSpec:
    max_train_instances = 5
    name = "indicxnli_nli_ta"

    if zeroshot:
        name += ",zeroshot=True"
        max_train_instances = 0
        adapter_spec = get_bhasa_adapter_spec(
            instructions="உங்களுக்கு இரண்டு வாக்கியங்கள், X மற்றும் Y, தரப்படும்.",
            input_suffix="பின்வரும் கூற்றுகளில் எது X மற்றும் Y வாக்கியங்களுடன் மிகப் பொருந்துகிறது எனக் கண்டறியவும்.\nA: X உண்மை என்றால் Y உம் உண்மையாக இருக்க வேண்டும்.\nB: X உம் Y உம் முரண்படுகின்றன.\nC: X உண்மையாக இருக்கும்போது Y உண்மையாக இருக்கலாம் அல்லது இல்லாமல் இருக்கலாம்.\nA அல்லது B அல்லது C எழுத்தில் மட்டும் பதிலளிக்கவும்.",
            output_noun="பதில்",
            max_train_instances=max_train_instances,
        )
    
    else:
        adapter_spec = get_generation_adapter_spec(
            instructions="உங்களுக்கு இரண்டு வாக்கியங்கள், X மற்றும் Y, தரப்படும்.\nபின்வரும் கூற்றுகளில் எது X மற்றும் Y வாக்கியங்களுடன் மிகப் பொருந்துகிறது எனக் கண்டறியவும்.\nA: X உண்மை என்றால் Y உம் உண்மையாக இருக்க வேண்டும்.\nB: X உம் Y உம் முரண்படுகின்றன.\nC: X உண்மையாக இருக்கும்போது Y உண்மையாக இருக்கலாம் அல்லது இல்லாமல் இருக்கலாம்.\nA அல்லது B அல்லது C எழுத்தில் மட்டும் பதிலளிக்கவும்.",
            output_noun="பதில்",
            max_train_instances=max_train_instances,
        )

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.bhasa_scenario.IndicXNLI_NLI_TA_Scenario"
    )

    return RunSpec(
        name=name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_f1_metric_specs(),
        groups=["bhasa_nlr"],
    )

@run_spec_function("indonli_nli_id")
def get_indonli_nli_id_spec(zeroshot=False) -> RunSpec:
    max_train_instances = 5
    name = "indonli_nli_id"

    if zeroshot:
        name += ",zeroshot=True"
        max_train_instances = 0
        adapter_spec = get_bhasa_adapter_spec(
            instructions="Anda akan diberikan dua kalimat, X dan Y.",
            input_suffix="Tentukan mana dari pernyataan berikut ini yang paling sesuai untuk kalimat X dan Y.\nA: Kalau X benar, maka Y juga harus benar.\nB: X bertentangan dengan Y.\nC: Ketika X benar, Y mungkin benar atau mungkin tidak benar.\nJawablah hanya dengan menggunakan satu huruf A, B atau C.",
            output_noun="Jawaban",
            max_train_instances=max_train_instances,
        )

    else:
        adapter_spec = get_generation_adapter_spec(
            instructions="Anda akan diberikan dua kalimat, X dan Y.\nTentukan mana dari pernyataan berikut ini yang paling sesuai untuk kalimat X dan Y.\nA: Kalau X benar, maka Y juga harus benar.\nB: X bertentangan dengan Y.\nC: Ketika X benar, Y mungkin benar atau mungkin tidak benar.\nJawablah hanya dengan menggunakan satu huruf A, B atau C.",
            output_noun="Jawaban",
            max_train_instances=max_train_instances,
        )

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.bhasa_scenario.IndoNLI_NLI_ID_Scenario"
    )

    return RunSpec(
        name=name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_f1_metric_specs(),
        groups=["bhasa_nlr"],
    )

xnli_prompts = {
    "th": {
        "instructions": "คุณจะได้รับสองข้อความ X และ Y",
        "input_suffix": "กรุณาพิจารณาว่า ข้อความใดต่อไปนี้ใช้กับข้อความ X และ Y ได้ดีที่สุด\nA: ถ้า X เป็นจริง Y จะต้องเป็นจริง\nB: X ขัดแย้งกับ Y\nC: เมื่อ X เป็นจริง Y อาจเป็นจริงหรือไม่ก็ได้\nกรุณาตอบด้วยตัวอักษร A, B หรือ C ตัวเดียวเท่านั้น",
        "output_noun": "คำตอบ",
    },
    "vi": {
        "instructions": "Bạn sẽ được cho hai câu, X và Y.",
        "input_suffix": "Xác định câu nào sau đây là câu phù hợp nhất cho câu X và Y.\nA: Nếu X đúng thì Y phải đúng.\nB: X mâu thuẫn với Y.\nC: Khi X đúng, Y có thể đúng hoặc không đúng.\nTrả lời với một chữ cái duy nhất A, B, hoặc C.",
        "output_noun": "Câu trả lời",
    },
}

def generate_xnli_run_spec(zeroshot=False, language="vi"):
    max_train_instances = 5
    name = f"xnli_nli_{language}"

    if zeroshot:
        name += ",zeroshot=True"
        max_train_instances = 0
        adapter_spec = get_bhasa_adapter_spec(
            instructions=xnli_prompts[language]['instructions'],
            input_suffix=xnli_prompts[language]['input_suffix'],
            output_noun=xnli_prompts[language]['output_noun'],
            max_train_instances=max_train_instances,
        )
    
    else:
        adapter_spec = get_generation_adapter_spec(
            instructions=xnli_prompts[language]['instructions'] + '\n' + xnli_prompts[language]['input_suffix'],
            output_noun=xnli_prompts[language]['output_noun'],
            max_train_instances=max_train_instances,
        )

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.bhasa_scenario.XNLI_NLI_Scenario",
        args={
            "language": language,
        },
    )

    return RunSpec(
        name=name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_classification_metric_specs(),
        groups=["bhasa_nlr"],
    )

@run_spec_function("xnli_nli_th")
def get_xnli_nli_th_spec(zeroshot=False) -> RunSpec:
    return generate_xnli_run_spec(zeroshot, language="th")

@run_spec_function("xnli_nli_vi")
def get_xnli_nli_vi_spec(zeroshot=False) -> RunSpec:
    return generate_xnli_run_spec(zeroshot, language="vi")

xcopa_prompts = {
    "id": {
        "input_noun": "Situasi",
        "output_noun": "Jawaban",
    },
    "ta": {
        "input_noun": "சூழ்நிலை",
        "output_noun": "பதில்",
    },
    "th": {
        "input_noun": "สถานการณ์",
        "output_noun": "nคำตอบ",
    },
    "vi": {
        "input_noun": "Tình huống",
        "output_noun": "Câu trả lời",
    }
}

def generate_xcopa_run_spec(zeroshot=False, language="id"):
    max_train_instances = 5
    name = f"xcopa_cr_{language}"

    if zeroshot:
        name += ",zeroshot=True"
        max_train_instances = 0
        
    adapter_spec = get_generation_adapter_spec(
        input_noun=xcopa_prompts[language]['input_noun'],
        output_noun=xcopa_prompts[language]['output_noun'],
        max_train_instances=max_train_instances,
    )

    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.bhasa_scenario.XCOPA_CR_Scenario",
        args={
            "language": language,
        },
    )

    return RunSpec(
        name=name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_classification_metric_specs(),
        groups=["bhasa_nlr"],
    )

@run_spec_function("xcopa_cr_id")
def get_xcopa_cr_id_spec(zeroshot=False) -> RunSpec:
    return generate_xcopa_run_spec(zeroshot, 'id')

@run_spec_function("xcopa_cr_ta")
def get_xcopa_cr_ta_spec(zeroshot=False) -> RunSpec:
    return generate_xcopa_run_spec(zeroshot, 'ta')

@run_spec_function("xcopa_cr_th")
def get_xcopa_cr_th_spec(zeroshot=False) -> RunSpec:
    return generate_xcopa_run_spec(zeroshot, 'th')

@run_spec_function("xcopa_cr_vi")
def get_xcopa_cr_vi_spec(zeroshot=False) -> RunSpec:
    return generate_xcopa_run_spec(zeroshot, 'vi')
    
# LD

# lindsea_mp_prompts = {
#     "id": {
#         "instructions": "System Prompt:\nAnda adalah seorang ahli bahasa Indonesia\nHuman Prompt:",
#         "input_suffix": "Jawablah dengan menggunakan A atau B saja.",
#     },
# }

# def generate_lindsea_mp_run_spec(zeroshot=False, language="id") -> RunSpec:
#     max_train_instances = 5
#     name = f"lindsea_mp_{language}"

#     if zeroshot:
#         name += ",zeroshot=True"
#         max_train_instances = 0

#     scenario_spec = ScenarioSpec(
#         class_name="helm.benchmark.scenarios.bhasa_scenario.LINDSEA_MP_Scenario",
#         args={
#             "language": language,
#         }
#     )

#     adapter_spec = get_bhasa_adapter_spec(
#         instructions=lindsea_mp_prompts[language]['instructions'],
#         input_suffix=lindsea_mp_prompts[language]['input_suffix'],
#         max_train_instances=max_train_instances,
#     )

#     return RunSpec(
#         name=name,
#         scenario_spec=scenario_spec,
#         adapter_spec=adapter_spec,
#         metric_specs=get_exact_match_metric_specs(),
#         groups=["bhasa_ld"],
#     )

# @run_spec_function("lindsea_mp_id")
# def get_lindsea_mp_id_spec(zeroshot=False) -> RunSpec:
#     return generate_lindsea_mp_run_spec(zeroshot, 'id')