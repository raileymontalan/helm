import datasets, random, os
import pandas as pd
from typing import List

from .scenario import Scenario, Instance, Reference, TRAIN_SPLIT, VALID_SPLIT, TEST_SPLIT, CORRECT_TAG, Input, Output, PassageQuestionInput
from helm.common.general import ensure_file_downloaded

# NLU

class IndicQA_QA_TA_Scenario(Scenario):
    """
    This is a Tamil question answer scenario. The data comes from IndicQA, a manually curated cloze-style reading comprehension dataset.

    The models are prompted using the following format:

        உங்களுக்கு ஒரு பத்தியும் ஒரு கேள்வியும் தரப்படும். தரப்பட்ட பத்தியிலிருந்து கேள்விக்கான பதிலைக் கண்டறியவும். கேள்விக்குப் பதிலளிக்கமுடியாவிட்டால் ஒற்றை எழுத்து X இனைப் பதிலாக அளிக்கவும்.
        பத்தி: {text}
        கேள்வி: {question}
        பதில்: 

    Target completion:
        <answer>

    @article{Doddapaneni2022towards,
        title={Towards Leaving No Indic Language Behind: Building Monolingual Corpora, Benchmark and Models for Indic Languages},
        author={Sumanth Doddapaneni and Rahul Aralikatte and Gowtham Ramesh and Shreyansh Goyal and Mitesh M. Khapra and Anoop Kunchukuttan and Pratyush Kumar},
        journal={ArXiv},
        year={2022},
        volume={abs/2212.05409}
    }
    """

    name = "indicqa_qa_ta"
    description = "IndicQA question answering dataset"
    tags = ["question_answering"]

    def __init__(self):
        super().__init__()
        self.splits = {
            'train': TRAIN_SPLIT,
            'test': TEST_SPLIT
        }

    def get_instances(self, output_path) -> List[Instance]:
        dataset = datasets.load_dataset("ai4bharat/IndicQA", "indicqa.ta")
        dataset = dataset['test'].train_test_split(test_size=0.8)

        outputs = []
        for split in list(dataset.keys()):
            data = dataset[split].to_pandas()
            for index, row in data.iterrows():
                if len(row["answers"]["text"][0].strip()) > 0:
                    passage = row["context"].strip()
                    question = row["question"].strip()
                    input = PassageQuestionInput(
                        passage=passage,
                        question=question,
                        passage_prefix="பத்தி: ",
                        question_prefix="கேள்வி: ",
                    )
                    output = Output(text=row["answers"]["text"][0].strip())
                    references = [
                        Reference(output, tags=[CORRECT_TAG]),
                    ]
                    instance = Instance(
                        input=input,
                        references=references, 
                        split=self.splits[split]
                    )
                    outputs.append(instance)
        return outputs

class TyDiQA_GoldP_QA_ID_Scenario(Scenario):
    """
    This is a Indonesian question answer scenario. The data comes from TyDIQA, a question answering dataset covering 
    11 typologically diverse languages with 204K question-answer pairs. 

    The models are prompted using the following format:

        Anda akan diberikan sebuah paragraf dan sebuah pertanyaan. Jawablah pertanyaannya dengan mengekstrak jawaban dari paragraf tersebut.
        Paragraf: <text>
        Pertanyaan: <question>
        Jawaban: 

    Target completion:
        <answer>

    @inproceedings{ruder-etal-2021-xtreme,
        title = "{XTREME}-{R}: Towards More Challenging and Nuanced Multilingual Evaluation",
        author = "Ruder, Sebastian  and
            Constant, Noah  and
            Botha, Jan  and
            Siddhant, Aditya  and
            Firat, Orhan  and
            Fu, Jinlan  and
            Liu, Pengfei  and
            Hu, Junjie  and
            Garrette, Dan  and
            Neubig, Graham  and
            Johnson, Melvin",
        booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
        month = nov,
        year = "2021",
        address = "Online and Punta Cana, Dominican Republic",
        publisher = "Association for Computational Linguistics",
        url = "https://aclanthology.org/2021.emnlp-main.802",
        doi = "10.18653/v1/2021.emnlp-main.802",
        pages = "10215--10245",
        }
    """

    name = "tydiqa_goldp_qa_ta"
    description = "TyDiQA GoldP question answering dataset"
    tags = ["question_answering"]

    def __init__(self):
        super().__init__()
        self.splits = {
            'train': TRAIN_SPLIT,
            'validation': TEST_SPLIT
        }

    def get_instances(self, output_path) -> List[Instance]:
        dataset = datasets.load_dataset("khalidalt/tydiqa-goldp", "indonesian")

        outputs = []
        for split in self.splits:
            data = dataset[split].to_pandas()
            for index, row in data.iterrows():
                passage = row["passage_text"].strip()
                question = row["question_text"].strip()
                input = PassageQuestionInput(
                    passage=passage,
                    question=question,
                    passage_prefix="Paragraf: ",
                    question_prefix="Pertanyaan: ",
                )
                output = Output(text=row["answers"]["text"][0].strip())
                references = [
                    Reference(output, tags=[CORRECT_TAG]),
                ]
                instance = Instance(
                    input=input,
                    references=references, 
                    split=self.splits[split]
                )
                outputs.append(instance)
        return outputs

class XQuAD_QA_Scenario(Scenario):
    """
    This is a XQuAD question answer scenario. The data comes from XQuAD, and the dataset consists of a subset of
    240 paragraphs and 1190 question-answer pairs from the development set of SQuAD v1.1 together (Rajpurkar et al., 2016).

    The models are prompted using the following general format:

        You will be given a paragraph and a question. Answer the question by extracting the answer from the paragraph.
        Paragraph: {text}
        Question: {question}
        Answer:

    Target completion:
        <answer>

    @article{Artetxe:etal:2019,
      author    = {Mikel Artetxe and Sebastian Ruder and Dani Yogatama},
      title     = {On the cross-lingual transferability of monolingual representations},
      journal   = {CoRR},
      volume    = {abs/1910.11856},
      year      = {2019},
      archivePrefix = {arXiv},
      eprint    = {1910.11856}
    }
    """

    name = "xquad_qa"
    description = "XQuAD question answering dataset"
    tags = ["question_answering"]

    def __init__(self, language: str):
        super().__init__()
        self.language = language
        self.splits = {
            'train': TRAIN_SPLIT,
            'test': TEST_SPLIT
        }
        self.prefixes = {
            "th": {
                "passage_prefix": "ข้อความ: ",
                "question_prefix": "คำถาม: ",
            },
            "vi": {
                "passage_prefix": "Đoạn văn: ",
                "question_prefix": "Câu hỏi: ",
            }
        }

    def get_instances(self, output_path) -> List[Instance]:
        dataset = datasets.load_dataset("xquad", f"xquad.{self.language}")
        dataset = dataset['validation'].train_test_split(test_size=0.8)

        outputs = []
        for split in list(dataset.keys()):
            data = dataset[split].to_pandas()
            for index, row in data.iterrows():
                passage = row["context"].strip()
                question = row["question"].strip()
                input = PassageQuestionInput(
                    passage=passage,
                    question=question,
                    passage_prefix=self.prefixes[self.language]['passage_prefix'],
                    question_prefix=self.prefixes[self.language]['question_prefix'],
                )
                output = Output(text=row["answers"]["text"][0].strip())
                references = [
                    Reference(output, tags=[CORRECT_TAG]),
                ]
                instance = Instance(
                    input=input,
                    references=references, 
                    split=self.splits[split]
                )
                outputs.append(instance)
        return outputs
   
class IndicSentiment_SA_TA_Scenario(Scenario):
    """
    This is a Tamil sentiment analysis scenario. The data comes from IndicXTREME, and consists of product reviews
    that were written by annotators. Labels are positive or negative. For this scenario, the `validation` split is
    used as the `train` split for in-context examples. 

    The models are prompted using the following format:

        பின்வரும் வாக்கியத்தில் வெளிப்படுத்தப்படும் உணர்வு எது?
        வாக்கியம்: <text>
        ஒரு சொல்லில் மட்டும் பதிலளிக்கவும்:
        - நேர்மறை
        - எதிர்மறை
        பதில்: 

    Target completion:
        <sentiment> (<sentiment>:positive or negative)

    @article{Doddapaneni2022towards,
        title={Towards Leaving No Indic Language Behind: Building Monolingual Corpora, Benchmark and Models for Indic Languages},
        author={Sumanth Doddapaneni and Rahul Aralikatte and Gowtham Ramesh and Shreyansh Goyal and Mitesh M. Khapra and Anoop Kunchukuttan and Pratyush Kumar},
        journal={ArXiv},
        year={2022},
        volume={abs/2212.05409}
    }
    """

    name = "indicsentiment_sa_ta"
    description = "IndicSentiment sentiment analysis dataset"
    tags = ["sentiment_analysis"]

    def __init__(self):
        super().__init__()
        self.splits = {
            'validation': TRAIN_SPLIT,
            'test': TEST_SPLIT
        }
        self.sentiment2label = {
            'Positive': 'ேநர்மைற',
            'Negative': 'எதிர்மைற',
        }

    def get_instances(self, output_path) -> List[Instance]:
        dataset = datasets.load_dataset("ai4bharat/IndicSentiment", "translation-ta")

        outputs = []
        for split in self.splits:
            data = dataset[split].to_pandas()
            for index, row in data.iterrows():
                if not row["LABEL"]:
                    continue
                input = Input(row["INDIC REVIEW"].strip())
                output = Output(text=self.sentiment2label[row["LABEL"]])
                references = [
                    Reference(output, tags=[CORRECT_TAG]),
                ]
                instance = Instance(
                    input=input,
                    references=references, 
                    split=self.splits[split]
                )
                outputs.append(instance)
        return outputs
 
class NusaX_SA_ID_Scenario(Scenario):
    """
    This is an Indonesian sentiment analysis scenario. The data consists of comments and reviews from the 
    IndoNLU benchmark. Labels are positive, negative or neutral.

    The models are prompted using the following format:

        Apa sentimen dari kalimat berikut ini?
        Kalimat: <text>
        Jawab dengan satu kata saja:
        - Positif
        - Negatif
        - Netral
        Jawaban: 

    Target completion:
        <sentiment> (<sentiment>:positive or negative or neutral)

    @inproceedings{van2018uit,
        title={UIT-VSFC: Vietnamese students’ feedback corpus for sentiment analysis},
        author={Van Nguyen, Kiet and Nguyen, Vu Duc and Nguyen, Phu XV and Truong, Tham TH and Nguyen, Ngan Luu-Thuy},
        booktitle={2018 10th international conference on knowledge and systems engineering (KSE)},
        pages={19--24},
        year={2018},
        organization={IEEE}
    }
    """

    name = "nusax_sa_id"
    description = "NusaX-Senti sentiment analysis dataset"
    tags = ["sentiment_analysis"]

    def __init__(self):
        super().__init__()
        self.splits = {
            'train': TRAIN_SPLIT,
            'valid': VALID_SPLIT,
            'test': TEST_SPLIT
        }
        self.sentiment2label = {
            'positive': 'Positif',
            'negative': 'Negatif',
            'neutral': 'Netral',
        }

    def download_dataset(self, output_path: str):
        URLS = {
            "test": "https://raw.githubusercontent.com/IndoNLP/nusax/main/datasets/sentiment/indonesian/test.csv",
            "train": "https://raw.githubusercontent.com/IndoNLP/nusax/main/datasets/sentiment/indonesian/train.csv",
            "valid": "https://raw.githubusercontent.com/IndoNLP/nusax/main/datasets/sentiment/indonesian/valid.csv",
        }

        data = {}
        for split in list(URLS.keys()):
            data[split] = []
            target_path_file = os.path.join(output_path, split)
            ensure_file_downloaded(source_url=URLS[split], target_path=target_path_file)
            data[split] = pd.read_csv(target_path_file)
        return data

    def get_instances(self, output_path) -> List[Instance]:
        data = self.download_dataset(output_path)
        outputs = []
        for split in list(data.keys()):
            for index, row in data[split].iterrows():
                input = Input(row["text"].strip())
                output = Output(text=self.sentiment2label[row["label"]])
                references = [
                    Reference(output, tags=[CORRECT_TAG]),
                ]
                instance = Instance(
                    input=input,
                    references=references,
                    split=self.splits[split]
                )
                outputs.append(instance)
        return outputs

class Wisesight_SA_TH_Scenario(Scenario):
    """
    This is an Thai sentiment analysis scenario. The data consists of social media messages regarding
    consumer products and services. Labels are positive, negative or neutral.

    The models are prompted using the following format:

        อารมณ์ความรู้สึกของข้อความต่อไปนี้เป็นอย่างไร?
        ข้อความ: {text}
        โปรดตอบโดยใช้คำเดียวเท่านั้น:
        - แง่บวก
        - แง่ลบ
        - เฉยๆ
        คำตอบ: 

    Target completion:
        <sentiment> (<sentiment>:positive or negative or neutral)

    @software{bact_2019_3457447,
        author       = {Suriyawongkul, Arthit and
                        Chuangsuwanich, Ekapol and
                        Chormai, Pattarawat and
                        Polpanumas, Charin},
        title        = {PyThaiNLP/wisesight-sentiment: First release},
        month        = sep,
        year         = 2019,
        publisher    = {Zenodo},
        version      = {v1.0},
        doi          = {10.5281/zenodo.3457447},
        url          = {https://doi.org/10.5281/zenodo.3457447}
    }
    """

    name = "wisesight_sa_th"
    description = "Wisesight sentiment analysis dataset"
    tags = ["sentiment_analysis"]

    def __init__(self):
        super().__init__()
        self.splits = {
            'train': TRAIN_SPLIT,
            'valid': VALID_SPLIT,
            'test': TEST_SPLIT
        }
        self.sentiment2label = {
            'pos': 'แง่บวก',
            'neg': 'แง่ลบ',
            'neu': 'เฉยๆ',
        }

    def download_dataset(self, output_path: str):
        URL = "https://github.com/PyThaiNLP/wisesight-sentiment/raw/master/huggingface/data.zip"
        data_path = os.path.join(output_path, "data")
        ensure_file_downloaded(source_url=URL, target_path=data_path, unpack=True)

        data = {}
        for split in self.splits.keys():
            data[split] = []
            target_path_file = os.path.join(data_path, "data", f"{split}.jsonl")
            data[split] = pd.read_json(target_path_file, lines=True)
        return data

    def get_instances(self, output_path) -> List[Instance]:
        data = self.download_dataset(output_path)
        outputs = []
        for split in list(data.keys()):
            for index, row in data[split].iterrows():
                if row["category"].strip() == "q":
                    continue
                input = Input(row["texts"].strip())
                output = Output(text=self.sentiment2label[row["category"]])
                references = [
                    Reference(output, tags=[CORRECT_TAG]),
                ]
                instance = Instance(
                    input=input,
                    references=references,
                    split=self.splits[split]
                )
                outputs.append(instance)
        return outputs

class UIT_VSFC_SA_VI_Scenario(Scenario):
    """
    This is a Vietnamese sentiment analysis scenario. The data consists of student feedback obtained from 
    end-of-semester surveys at a Vietnamese university. Feedback is labeled as one of three sentiment 
    polarities: positive, negative or neutral.

    The models are prompted using the following format:

        Sắc thái của câu sau đây là gì?
        Câu: <text>
        Trả lời với một từ duy nhất:
        - Tích cực
        - Tiêu cực
        - Trung lập
        Câu trả lời: 

    Target completion:
        <sentiment> (<sentiment>:positive or negative or neutral)

    @inproceedings{van2018uit,
        title={UIT-VSFC: Vietnamese students’ feedback corpus for sentiment analysis},
        author={Van Nguyen, Kiet and Nguyen, Vu Duc and Nguyen, Phu XV and Truong, Tham TH and Nguyen, Ngan Luu-Thuy},
        booktitle={2018 10th international conference on knowledge and systems engineering (KSE)},
        pages={19--24},
        year={2018},
        organization={IEEE}
    }
    """

    name = "uit_vsfc_sa_vi"
    description = "BHASA Vietnamese Students' Feedback Corpus for sentiment analysis"
    tags = ["sentiment_analysis"]

    def __init__(self):
        super().__init__()
        self.splits = {
            'train': TRAIN_SPLIT,
            'valid': VALID_SPLIT,
            'test': TEST_SPLIT
        }
        self.id2label = {
            0: 'Tiêu cực',
            1: 'Trung lập',
            2: 'Tích cực',
        }

    def download_dataset(self, output_path: str):
        URLS = {
            "train": {
                "sentences": "https://drive.google.com/uc?id=1nzak5OkrheRV1ltOGCXkT671bmjODLhP&export=download",
                "sentiments": "https://drive.google.com/uc?id=1ye-gOZIBqXdKOoi_YxvpT6FeRNmViPPv&export=download",
            },
            "valid": {
                "sentences": "https://drive.google.com/uc?id=1sMJSR3oRfPc3fe1gK-V3W5F24tov_517&export=download",
                "sentiments": "https://drive.google.com/uc?id=1GiY1AOp41dLXIIkgES4422AuDwmbUseL&export=download",
            },
            "test": {
                "sentences": "https://drive.google.com/uc?id=1aNMOeZZbNwSRkjyCWAGtNCMa3YrshR-n&export=download",
                "sentiments": "https://drive.google.com/uc?id=1vkQS5gI0is4ACU58-AbWusnemw7KZNfO&export=download",
            },
        }

        data = {}
        for split in list(URLS.keys()):
            data[split] = {}
            for file in list(URLS[split].keys()):
                data[split][file] = []
                target_path_file = os.path.join(output_path, split, file)
                ensure_file_downloaded(source_url=URLS[split][file], target_path=target_path_file)
                with open(target_path_file, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        data[split][file].append(str(line).strip())
        return data

    def get_instances(self, output_path) -> List[Instance]:
        data = self.download_dataset(output_path)
        outputs = []
        for split in list(data.keys()):
            for i, r in zip(data[split]['sentences'], data[split]['sentiments']):
                input = Input(i)
                output = Output(text=self.id2label[int(r)])
                references = [
                    Reference(output, tags=[CORRECT_TAG]),
                ]
                instance = Instance(
                    input=input, 
                    references=references, 
                    split=self.splits[split]
                )
                outputs.append(instance)
        return outputs
  
class MLHSD_TD_ID_Scenario(Scenario):
    """
    This is an Indonesian toxicity detection scenario. The data comes from the Indonesian Twitter Multi-label Hate Speech and 
    Abusive Language Detection Dataset.

    The models are prompted using the following format:

        Anda adalah pendeteksi ujaran kebencian. Definisi dari labelnya adalah sebagai berikut:
        Bersih: Tidak ada ujaran kebencian.
        Kasar: Ada ujaran kebencian dan kata-kata kasar, namun tidak menyerang pihak tertentu.
        Benci: Ada ujaran kebencian langsung atau serangan terhadap pihak tertentu.
        Berdasarkan definisi labelnya, klasifikasikan kalimat berikut ini:
        Kalimat: <text>
        Jawab dengan hanya A, B atau C.
        A: Bersih
        B: Kasar
        C: Benci
        Jawaban: 

    Target completion:
        <answer>

    @inproceedings{ibrohim2019multi,
        title={Multi-label hate speech and abusive language detection in Indonesian Twitter},
        author={Ibrohim, Muhammad Okky and Budi, Indra},
        booktitle={Proceedings of the third workshop on abusive language online},
        pages={46--57},
        year={2019}
    }
    """

    name = "mlhsd_td_id"
    description = "MLHSD toxicity detection dataset"
    tags = ["toxicity_dectection"]

    def __init__(self):
        super().__init__()
        self.splits = {
            'train': TRAIN_SPLIT,
            'test': TEST_SPLIT
        }

        self.id2label = {
            'Bersih': 'A',
            'Kasar': 'B',
            'Benci': 'C',
        }

    def download_dataset(self, output_path: str):
        URL = "https://raw.githubusercontent.com/okkyibrohim/id-multi-label-hate-speech-and-abusive-language-detection/master/re_dataset.csv"
        target_path_file = os.path.join(output_path, "mlhsd")
        ensure_file_downloaded(source_url=URL, target_path=target_path_file)
        df = pd.read_csv(target_path_file, encoding="ISO-8859-1")

        split_index = int(len(df)*0.8)
        data = {}
        data['train'] = df.iloc[:split_index]
        data['test'] = df.iloc[split_index:]
        return data
    
    def get_label(self, row) -> str:
        if int(row["HS"]) == 1:
            return "C"
        elif int(row["Abusive"]) == 1:
            return "B"
        else:
            return "A"

    def get_instances(self, output_path) -> List[Instance]:
        data = self.download_dataset(output_path)
        outputs = []
        for split in list(data.keys()):
            for index, row in data[split].iterrows():
                input = Input(row["Tweet"].strip())
                output = Output(text=self.get_label(row))
                references = [
                    Reference(output, tags=[CORRECT_TAG]),
                ]
                instance = Instance(
                    input=input,
                    references=references,
                    split=self.splits[split]
                )
                outputs.append(instance)
        return outputs

class Thai_Toxicity_Tweets_TD_TH_Scenario(Scenario):
    """
    This is a Thai toxicity detection scenario. The data comes from the Thai Toxicity Tweets dataset.

    The models are prompted using the following format:

        คุณเป็นเครื่องตรวจจับข้อความที่เป็นพิษ ข้อความเป็นพิษถ้ามันทำร้าย สร้างความเสียหาย หรือสื่อเจตนาเชิงลบถึงบุคคลหรือกลุ่มบุคคล คุณควรคำนึงถึงการประชดประชันด้วย เมื่อได้รับข้อความ ให้ตอบด้วยตัวอักษร Y หากเป็นพิษ และ N หากไม่เป็นพิษ
        ข้อความ: <text>
        คำตอบ: 

    Target completion:
        <toxicity>

    @inproceedings{sirihattasak2018annotation,
        title={Annotation and classification of toxicity for Thai Twitter},
        author={Sirihattasak, Sugan and Komachi, Mamoru and Ishikawa, Hiroshi},
        booktitle={TA-COS 2018: 2nd Workshop on Text Analytics for Cybersecurity and Online Safety},
        pages={1},
        year={2018}
    }
    """

    name = "thai_toxicity_tweets_td_th"
    description = "Thai Toxicity Tweets toxicity detection dataset"
    tags = ["toxicity_detection"]

    def __init__(self):
        super().__init__()
        self.splits = {
            'train': TRAIN_SPLIT,
            'test': TEST_SPLIT
        }
        self.id2label = {
            0: 'N',
            1: 'Y',
        }

    def get_instances(self, output_path) -> List[Instance]:
        dataset = datasets.load_dataset("thai_toxicity_tweet")
        dataset = dataset['train'].train_test_split(test_size=0.8)

        outputs = []
        for split in list(dataset.keys()):
            data = dataset[split].to_pandas()
            for index, row in data.iterrows():
                if len(row["tweet_text"]) > 0 and row["tweet_text"] != "TWEET_NOT_FOUND":
                    input = Input(row["tweet_text"].strip())
                    output = Output(text=self.id2label[int(row["is_toxic"])])
                    references = [
                        Reference(output, tags=[CORRECT_TAG]),
                    ]
                    instance = Instance(
                        input=input,
                        references=references, 
                        split=self.splits[split]
                    )
                    outputs.append(instance)
        return outputs

class ViHSD_TD_VI_Scenario(Scenario):
    """
    This is a Vietnamese toxicity detection scenario. The data comes from the ViHSD dataset.

    The models are prompted using the following format:

        Bạn là máy phát hiện phát ngôn thù ghét. Các nhãn được định nghĩa như sau:
        Sạch: Không quấy rối.
        Công kích: Bao gồm quấy rối và thậm chí chửi thề, nhưng không tấn công bất kì đối tượng cụ thể nào.
        Thù ghét: Trực tiếp quấy rối hay lăng mạ một đối tượng cụ thể.
        Với các định nghĩa của nhãn, hãy phân loại câu dưới đây:
        Câu: <text>
        Chỉ trả lời bằng A, B hoặc C.
        A: Sạch
        B: Công kích
        C: Thù ghét
        Câu trả lời: 

    Target completion:
        <toxicity>

    @inproceedings{luu2021large,
        title={A large-scale dataset for hate speech detection on vietnamese social media texts},
        author={Luu, Son T and Nguyen, Kiet Van and Nguyen, Ngan Luu-Thuy},
        booktitle={Advances and Trends in Artificial Intelligence. Artificial Intelligence Practices: 34th International Conference on Industrial, Engineering and Other Applications of Applied Intelligent Systems, IEA/AIE 2021, Kuala Lumpur, Malaysia, July 26--29, 2021, Proceedings, Part I 34},
        pages={415--426},
        year={2021},
        organization={Springer}
    }
    """

    name = "vihsd_td_vi"
    description = "ViHSD toxicity detection dataset"
    tags = ["toxicity_detection"]

    def __init__(self):
        super().__init__()
        self.splits = {
            'train': TRAIN_SPLIT,
            'dev': VALID_SPLIT,
            'test': TEST_SPLIT
        }
        self.id2label = {
            0: 'A',
            1: 'B',
            2: 'C',
        }

    def download_dataset(self, output_path: str):
        URL = "https://raw.githubusercontent.com/sonlam1102/vihsd/main/data/vihsd.zip"
        data_path = os.path.join(output_path, "data")
        ensure_file_downloaded(source_url=URL, target_path=data_path, unpack=True)

        data = {}
        for split in self.splits.keys():
            data[split] = []
            target_path_file = os.path.join(data_path, "vihsd", f"{split}.csv")
            data[split] = pd.read_csv(target_path_file)
        return data

    def get_instances(self, output_path) -> List[Instance]:
        data = self.download_dataset(output_path)
        outputs = []
        for split in list(data.keys()):
            for index, row in data[split].iterrows():
                input = Input(str(row["free_text"]).strip())
                output = Output(text=self.id2label[int(row["label_id"])])
                references = [
                    Reference(output, tags=[CORRECT_TAG]),
                ]
                instance = Instance(
                    input=input, 
                    references=references, 
                    split=self.splits[split]
                )
                outputs.append(instance)
        return outputs

# NLG

class Flores_MT_Scenario(Scenario):
    """
    This is the Flores machine translation scenario.

    The models are prompted using the following general format:

        Translate the following text into XX language.
        Text: <text>
        Translation:

    Target completion:
        <translation>

    @article{nllb2022,
        author    = {NLLB Team, Marta R. Costa-jussà, James Cross, Onur Çelebi, Maha Elbayad, Kenneth Heafield, Kevin Heffernan, Elahe Kalbassi,  Janice Lam, Daniel Licht, Jean Maillard, Anna Sun, Skyler Wang, Guillaume Wenzek, Al Youngblood, Bapi Akula, Loic Barrault, Gabriel Mejia Gonzalez, Prangthip Hansanti, John Hoffman, Semarley Jarrett, Kaushik Ram Sadagopan, Dirk Rowe, Shannon Spruit, Chau Tran, Pierre Andrews, Necip Fazil Ayan, Shruti Bhosale, Sergey Edunov, Angela Fan, Cynthia Gao, Vedanuj Goswami, Francisco Guzmán, Philipp Koehn, Alexandre Mourachko, Christophe Ropers, Safiyyah Saleem, Holger Schwenk, Jeff Wang},
        title     = {No Language Left Behind: Scaling Human-Centered Machine Translation},
        year      = {2022}
    }

    """

    name = "flores_mt"
    description = "Flores machine translation dataset"
    tags = ["machine_translation"]

    def __init__(self, pair: str):
        super().__init__()
        self.pair = pair
        self.source = pair.split('_')[0]
        self.target = pair.split('_')[1]

        self.splits = {
            'dev': TRAIN_SPLIT,
            'devtest': TEST_SPLIT
        }

        self.languages = {
            "en": "eng_Latn",
            "id": "ind_Latn",
            "vi": "vie_Latn",
            "th": "tha_Thai",
            "ta": "tam_Taml",
        }

    def get_instances(self, output_path) -> List[Instance]:
        source_dataset = datasets.load_dataset("facebook/flores", self.languages[self.source])
        target_dataset = datasets.load_dataset("facebook/flores", self.languages[self.target])

        outputs = []
        for split in self.splits.keys():
            source_df = source_dataset[split].to_pandas()
            target_df = target_dataset[split].to_pandas()
            df = source_df.join(target_df, lsuffix="_source", rsuffix="_target")
            for index, row in df.iterrows():
                input = Input(row["sentence_source"].strip())
                output = Output(row["sentence_target"].strip())
                references = [
                    Reference(output, tags=[CORRECT_TAG]),
                ]
                instance = Instance(
                    input=input,
                    references=references,
                    split=self.splits[split]
                )
                outputs.append(instance)
        return outputs

class XLSum_AS_Scenario(Scenario):
    """
    This is the XLSum abstractive summarization scenario.

    The models are prompted using the following general format:

        Article: <text>
        Summarize this <language> language article in 1 or 2 sentences. The answer must be written in <language> language.
        Summary:

    Target completion:
        <summary>

    @inproceedings{hasan-etal-2021-xl,
        title = "{XL}-Sum: Large-Scale Multilingual Abstractive Summarization for 44 Languages",
        author = "Hasan, Tahmid  and
            Bhattacharjee, Abhik  and
            Islam, Md. Saiful  and
            Mubasshir, Kazi  and
            Li, Yuan-Fang  and
            Kang, Yong-Bin  and
            Rahman, M. Sohel  and
            Shahriyar, Rifat",
        booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
        month = aug,
        year = "2021",
        address = "Online",
        publisher = "Association for Computational Linguistics",
        url = "https://aclanthology.org/2021.findings-acl.413",
        pages = "4693--4703",
    }


    """

    name = "xlsum_as"
    description = "XLSUm abstractive summarization dataset"
    tags = ["abstractive_summarization"]

    def __init__(self, language: str):
        super().__init__()
        self.language = language

        self.splits = {
            'train': TRAIN_SPLIT,
            'test': TEST_SPLIT,
            'validation': VALID_SPLIT
        }

        self.languages = {
            "id": "indonesian",
            "vi": "vietnamese",
            "th": "thai",
            "ta": "tamil",
        }

    def get_instances(self, output_path) -> List[Instance]:
        dataset = datasets.load_dataset("csebuetnlp/xlsum", self.languages[self.language])

        outputs = []
        for split in self.splits.keys():
            df = dataset[split].to_pandas()
            for index, row in df.iterrows():
                input = Input(row["text"].strip())
                output = Output(row["summary"].strip())
                references = [
                    Reference(output, tags=[CORRECT_TAG]),
                ]
                instance = Instance(
                    input=input,
                    references=references,
                    split=self.splits[split]
                )
                outputs.append(instance)
        return outputs

# NLR

class IndicXNLI_NLI_TA_Scenario(Scenario):
    """
    This is a Tamil natural language inference scenario. The data was automatically translated from XNLI into 11 Indic languages.

    The models are prompted using the following format:

        பின் வரும் வாக் கியத் தில்
         ெவளிப் படுத் தப் படும்
        உணர் வு எது? <sentence>
        ஒரு ெசால் லில் மட் டும் பதிலளிக் கவும் :
         ேநர்மைற/எதிர் மைற

    Target completion:
        <sentiment> (<sentiment>:positive or negative or neutral)

    @misc{https://doi.org/10.48550/arxiv.2204.08776,
        doi = {10.48550/ARXIV.2204.08776},
        url = {https://arxiv.org/abs/2204.08776},
        author = {Aggarwal, Divyanshu and Gupta, Vivek and Kunchukuttan, Anoop},
        keywords = {Computation and Language (cs.CL), Artificial Intelligence (cs.AI), FOS: Computer and information sciences, FOS: Computer and information sciences},
        title = {IndicXNLI: Evaluating Multilingual Inference for Indian Languages}, 
        publisher = {arXiv},
        year = {2022},
        copyright = {Creative Commons Attribution 4.0 International}
    }
    """

    name = "indicxnli_nli_ta"
    description = "IndicXNLI natural language inference dataset"
    tags = ["natural_language_inference"]

    def __init__(self):
        super().__init__()
        self.splits = {
            'train': TRAIN_SPLIT,
            'validation': VALID_SPLIT,
            'test': TEST_SPLIT,
        }
        self.id2label = {
            0: "A",
            1: "B",
            2: "C"
        }

    def get_instances(self, output_path) -> List[Instance]:
        dataset = datasets.load_dataset("Divyanshu/indicxnli", "ta")

        outputs = []
        for split in self.splits:
            data = dataset[split].to_pandas()
            for index, row in data.iterrows():
                passage = "X: " + row["premise"].strip() + "\nY: " + row["hypothesis"].strip()
                input = Input(passage)
                output = Output(text=self.id2label[row["label"]])
                references = [
                    Reference(output, tags=[CORRECT_TAG]),
                ]
                instance = Instance(
                    input=input,
                    references=references, 
                    split=self.splits[split]
                )
                outputs.append(instance)
        return outputs

class IndoNLI_NLI_ID_Scenario(Scenario):
    """
    This is a IndoNLI natural language inference scenario. The data comes from IndoNLI, and incorporates various linguistic 
    phenomena such as numerical reasoning, structural changes, idioms, or temporal and spatial reasoning. Labels are
    entailment, contradiction, or neutral. 

    The models are prompted using the following format:

        Anda akan diberikan dua kalimat, X dan Y.
        X: <sentence1>
        Y: <sentence2>
        Tentukan mana dari pernyataan berikut ini yang paling sesuai untuk kalimat X dan Y.
        A: Kalau X benar, maka Y juga harus benar.
        B: X bertentangan dengan Y.
        C: Ketika X benar, Y mungkin benar atau mungkin tidak benar.
        Jawablah hanya dengan menggunakan satu huruf A, B atau C.
        Jawaban: 

    Target completion:
        <answer> (<answer>:entailment, contradiction, or neutral)

    @inproceedings{mahendra-etal-2021-indonli,
        title = "{I}ndo{NLI}: A Natural Language Inference Dataset for {I}ndonesian",
        author = "Mahendra, Rahmad and Aji, Alham Fikri and Louvan, Samuel and Rahman, Fahrurrozi and Vania, Clara",
        booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
        month = nov,
        year = "2021",
        address = "Online and Punta Cana, Dominican Republic",
        publisher = "Association for Computational Linguistics",
        url = "https://aclanthology.org/2021.emnlp-main.821",
        pages = "10511--10527",
    }
    """

    name = "indonli_nli_id"
    description = "IndoNLI natural language inference dataset"
    tags = ["textual_entailment"]

    def __init__(self):
        super().__init__()
        self.splits = {
            'train': TRAIN_SPLIT,
            'validation': VALID_SPLIT,
            'test_expert': TEST_SPLIT,
            'test_lay': TEST_SPLIT,
        }
        self.id2label = {
            0: 'A',
            1: 'B',
            2: 'C'
        }

    def get_instances(self, output_path) -> List[Instance]:
        dataset = datasets.load_dataset("indonli")

        outputs = []
        for split in list(dataset.keys()):
            data = dataset[split].to_pandas()
            for index, row in data.iterrows():
                passage = "X: " + row["premise"].strip() + "\nY: " + row["hypothesis"].strip()
                input = Input(passage)
                output = Output(self.id2label[int(row["label"])])
                references = [
                    Reference(output, tags=[CORRECT_TAG]),
                ]
                instance = Instance(
                    input=input,
                    references=references, 
                    split=self.splits[split]
                )
                outputs.append(instance)
        return outputs
        
class XNLI_NLI_Scenario(Scenario):
    """
    This is a XNLI natural language inference scenario. The data comes from XNLI, and incorporates various linguistic 
    phenomena such as numerical reasoning, structural changes, idioms, or temporal and spatial reasoning. Labels are
    entailment, neutral, or contradiction. 

    The models are prompted using the following general format:

        You will be given two sentences, X and Y.
        X: {sentence1}
        Y: {sentence2}
        Determine which of the following statements applies to sentences X and Y the best.
        A: If X is true, Y must be true.
        B: X contradicts Y.
        C: When X is true, Y may or may not be true.
        Answer strictly with a single letter A, B or C.
        Answer:

    Target completion:
        <answer> (<answer>:entailment, neutral, or contradiction)

    @InProceedings{conneau2018xnli,
        author = {Conneau, Alexis
                    and Rinott, Ruty
                    and Lample, Guillaume
                    and Williams, Adina
                    and Bowman, Samuel R.
                    and Schwenk, Holger
                    and Stoyanov, Veselin},
        title = {XNLI: Evaluating Cross-lingual Sentence Representations},
        booktitle = {Proceedings of the 2018 Conference on Empirical Methods
                    in Natural Language Processing},
        year = {2018},
        publisher = {Association for Computational Linguistics},
        location = {Brussels, Belgium},
    }
    """

    name = "xnli_nli"
    description = "XNLI natural language inference dataset"
    tags = ["textual_entailment"]

    def __init__(self, language: str):
        super().__init__()
        self.language = language
        self.splits = {
            'train': TRAIN_SPLIT,
            'validation': VALID_SPLIT,
            'test': TEST_SPLIT,
        }
        self.id2label = {
            0: 'A',
            2: 'B',
            1: 'C'
        }

    def get_instances(self, output_path) -> List[Instance]:
        dataset = datasets.load_dataset("xnli", self.language)

        outputs = []
        for split in list(dataset.keys()):
            data = dataset[split].to_pandas()
            for index, row in data.iterrows():
                passage = "X: " + row["premise"].strip() + "\nY: " + row["hypothesis"].strip()
                input = Input(passage)
                output = Output(self.id2label[int(row["label"])])
                references = [
                    Reference(output, tags=[CORRECT_TAG]),
                ]
                instance = Instance(
                    input=input,
                    references=references, 
                    split=self.splits[split]
                )
                outputs.append(instance)
        return outputs

class XCOPA_CR_Scenario(Scenario):
    """
    This is a XCOPA causal reasoning scenario. The data comes from XCOPA, a translation and reannotation of the English COPA.

    The models are prompted using the following general format:

        Situation: {premise}
        Based on the situation above, which of the following choices is most likely to be its {question}?
        A: {choice1}
        B: {choice2}
        Answer with only A or B.
        Answer:

    Target completion:
        <answer> (<answer>:entailment, neutral, or contradiction)

    @article{ponti2020xcopa,
    title={{XCOPA: A} Multilingual Dataset for Causal Commonsense Reasoning},
    author={Edoardo M. Ponti, Goran Glava
    {s}, Olga Majewska, Qianchu Liu, Ivan Vuli'{c} and Anna Korhonen},
    journal={arXiv preprint},
    year={2020},
    url={https://ducdauge.github.io/files/xcopa.pdf}
    }

    @inproceedings{roemmele2011choice,
    title={Choice of plausible alternatives: An evaluation of commonsense causal reasoning},
    author={Roemmele, Melissa and Bejan, Cosmin Adrian and Gordon, Andrew S},
    booktitle={2011 AAAI Spring Symposium Series},
    year={2011},
    url={https://people.ict.usc.edu/~gordon/publications/AAAI-SPRING11A.PDF},
    }
    """

    name = "xcopa_cr"
    description = "XCOPA causal reasoning dataset"
    tags = ["causal_reasoning"]

    def __init__(self, language: str):
        super().__init__()
        self.language = language
        self.splits = {
            'validation': TRAIN_SPLIT,
            'test': TEST_SPLIT,
        }
        self.id2label = {
            0: 'A',
            1: 'B',
        }
        self.prompt = {
            "id": {
                "cause": "sebab",
                "effect": "akibat",
                "instruction1": "Berdasarkan situasi di atas, mana dari pilihan-pilihan berikut ini yang lebih mungkin menjadi {}?",
                "instruction2": "Tanggapi dengan hanya menggunakan A atau B.",
            },
            "ta": {
                "cause": "สาเหตุ",
                "effect": "ผล",
                "instruction1": "பின்வரும் வாக்கியங்களில் பெரும்பாலும் எது தரப்பட்ட சூழ்நிலைக்குரிய {} இருக்கும்?",
                "instruction2": "A அல்லது B எழுத்தில் மட்டும் பதிலளிக்கவும்.",
            },
            "th": {
                "cause": "สาเหตุ",
                "effect": "ผล",
                "instruction1": "เมื่อพิจารณาจากสถานการณ์นี้ ตัวเลือกใดต่อไปนี้น่าจะเป็น{}มากกว่ากัน?",
                "instruction2": "กรุณาตอบด้วยตัวอักษร A หรือ B เท่านั้น",
            },
            "vi": {
                "cause": "nguyên nhân",
                "effect": "kết quả",
                "instruction1": "Với tình huống trên, lựa chọn nào dưới đây có khả năng cao là {} của nó hơn?",
                "instruction2": "Chỉ trả lời bằng chữ cái A hoặc B.",
            },
        }

    def get_instances(self, output_path) -> List[Instance]:
        language_dataset = datasets.load_dataset("xcopa", self.language)
        tamil_dataset = datasets.load_dataset("xcopa", 'ta')

        outputs = []
        for split in self.splits.keys():
            language_df = language_dataset[split].to_pandas()
            tamil_df = tamil_dataset[split].to_pandas()
            df = pd.merge(language_df, tamil_df[['question', 'idx']], on='idx') # Use the Tamil split's question column
            for index, row in df.iterrows():
                instruction1 = self.prompt[self.language]['instruction1'].format(self.prompt[self.language][row['question_y']])
                passage = "{premise}\n{instruction1}\nA: {choice1}\nB:{choice2}\n{instruction2}".format(
                    premise=row["premise"].strip(),
                    instruction1=instruction1,
                    choice1=row["choice1"].strip(),
                    choice2=row["choice2"].strip(),
                    instruction2=self.prompt[self.language]['instruction2'],
                )
                input = Input(passage)
                output = Output(self.id2label[int(row["label"])])
                references = [
                    Reference(output, tags=[CORRECT_TAG]),
                ]
                instance = Instance(
                    input=input,
                    references=references, 
                    split=self.splits[split]
                )
                outputs.append(instance)
        return outputs
    
# LD
    
class LINDSEA_MP_Scenario(Scenario):
    """
    This is a LINDSEA minimal pairs (linguistic diagnostic for syntax) scenario. The data comes from the BHASA LINDSEA dataset.

    The models are prompted using the following general format:

        System Prompt:
        You are a <language> linguist
        Human Prompt:
        Which sentence is more acceptable?
        A: <sentence1>
        B: <sentence2>
        Answer with A or B only.

    Target completion:
        <choice>

    @misc{leong2023bhasa,
        title={BHASA: A Holistic Southeast Asian Linguistic and Cultural Evaluation Suite for Large Language Models}, 
        author={Wei Qi Leong and Jian Gang Ngui and Yosephine Susanto and Hamsawardhini Rengarajan and Kengatharaiyer Sarveswaran and William Chandra Tjhi},
        year={2023},
        eprint={2309.06085},
        archivePrefix={arXiv},
        primaryClass={cs.CL}
    }
    """

    name = "lindsea_mp"
    description = "LINDSEA minimal pairs dataset"
    tags = ["minimal_pairs", "linguistic_diagnostic", "syntax"]

    def __init__(self, language: str):
        super().__init__()
        self.language = language
        self.splits = {
            'train': TRAIN_SPLIT,
            'test': TEST_SPLIT
        }
        self.prompt = {
            "id": {
                "question": "Kalimat mana yang lebih mungkin?",
            },
        }

    def download_dataset(self, output_path: str):
        URLS = {
            "npis_and_negation": f"https://raw.githubusercontent.com/aisingapore/BHASA/main/lindsea/{self.language}/syntax/NPIs_and_negation.jsonl",
            "argument_structure": f"https://raw.githubusercontent.com/aisingapore/BHASA/main/lindsea/{self.language}/syntax/argument_structure.jsonl",
            "filler_gap_dependencies": f"https://raw.githubusercontent.com/aisingapore/BHASA/main/lindsea/{self.language}/syntax/filler-gap_dependencies.jsonl",
            "morphology": f"https://raw.githubusercontent.com/aisingapore/BHASA/main/lindsea/{self.language}/syntax/morphology.jsonl",
        }
        
        data_files = {}
        for file in list(URLS.keys()):
            data_files[file] = []
            target_path_file = os.path.join(output_path, file)
            ensure_file_downloaded(source_url=URLS[file], target_path=target_path_file)
            data_files[file] = pd.read_json(target_path_file, lines=True)
        data = pd.concat(data_files)
        
        return data


    def get_instances(self, output_path) -> List[Instance]:
        data = self.download_dataset(output_path)
        dataset = datasets.Dataset.from_pandas(data).train_test_split(test_size=0.8)
        
        outputs = []
        for split in list(dataset.keys()):
            data = dataset[split].to_pandas()
            for index, row in data.iterrows():
                input = Input(text=self.prompt[self.language]['question'])
                references = [
                    Reference(Output(text=row["correct"].strip()), tags=[CORRECT_TAG]),
                    Reference(Output(text=row["wrong"].strip()), tags=[]),
                ]
                random.shuffle(references) # Shuffle order of references
                instance = Instance(
                    input=input,
                    references=references, 
                    split=self.splits[split]
                )
                outputs.append(instance)
        return outputs
    
class LINDSEA_PR_Scenario(Scenario):
    """
    This is a LINDSEA pragmatic reasoning scenario. The data comes from the BHASA LINDSEA dataset.

    For single sentence questions, the models are prompted using the following general format:

        System Prompt:
        You are a <language> linguist
        Human Prompt:
        Is the following statement true or false?
        Statement: <sentence>
        Answer only with True or False.

    For double sentence questions, the models are prompted using the following general format:

        System Prompt:
        You are a <language> linguist
        Human Prompt:
        Situation: <premise>
        Given this situation, is the following statement true or false?
        Statement: <sentence>
        Answer only with True or False.

    Target completion:
        <answer>

    @misc{leong2023bhasa,
        title={BHASA: A Holistic Southeast Asian Linguistic and Cultural Evaluation Suite for Large Language Models}, 
        author={Wei Qi Leong and Jian Gang Ngui and Yosephine Susanto and Hamsawardhini Rengarajan and Kengatharaiyer Sarveswaran and William Chandra Tjhi},
        year={2023},
        eprint={2309.06085},
        archivePrefix={arXiv},
        primaryClass={cs.CL}
    }
    """

    name = "lindsea_pr"
    description = "LINDSEA pragmatic reasoning dataset"
    tags = ["pragmatic_reasoning", "linguistic_diagnostic", "pragmatics"]

    def __init__(self, language: str):
        super().__init__()
        self.language = language
        self.splits = {
            'train': TRAIN_SPLIT,
            'test': TEST_SPLIT
        }
        self.prompt = {
            "id": {
                "question1": "Apakah pernyataan berikut ini benar atau salah?",
                "question2": "Berdasarkan situasi ini, apakah pernyataan berikut ini benar atau salah?",
                "situation": "Situasi",
                "statement": "Pernyataan",
                "True": "Benar",
                "False": "Salah",
            },
        }

    def get_mapping(self, x):
        return self.prompt[self.language][x.strip().capitalize()]

    def download_dataset(self, output_path: str):
        URLS = {
            "pragmatic_reasoning_pair": f"https://raw.githubusercontent.com/aisingapore/BHASA/main/lindsea/{self.language}/pragmatics/pragmatic_reasoning_pair.jsonl",
            # "pragmatic_reasoning_single": f"https://raw.githubusercontent.com/aisingapore/BHASA/main/lindsea/{self.language}/pragmatics/pragmatic_reasoning_single.jsonl",
        }
        
        data_files = {}
        for file in list(URLS.keys()):
            data_files[file] = []
            target_path_file = os.path.join(output_path, file)
            ensure_file_downloaded(source_url=URLS[file], target_path=target_path_file)
            data_files[file] = pd.read_json(target_path_file, lines=True)
        data = pd.concat(data_files.values(), ignore_index=True)
        data['label'] = data['label'].astype(str)
        data['label'] = data['label'].apply(self.get_mapping)
        print(data['label'])
        return data


    def get_instances(self, output_path) -> List[Instance]:
        data = self.download_dataset(output_path)
        dataset = datasets.Dataset.from_pandas(data).train_test_split(test_size=0.8)
        
        outputs = []
        for split in list(dataset.keys()):
            data = dataset[split].to_pandas()
            for index, row in data.iterrows():
                if row['question']:
                    text = self.prompt[self.language]['question1'] + "\n"
                    text += f"{self.prompt[self.language]['statement']}: " + row['text']
                    input = Input(text=text)
                else:
                    text = self.prompt[self.language]['situation'] + ": "
                    text += row['text'] + "\n"
                    text += self.prompt[self.language]['question2'] + "\n" 
                    text += f"{self.prompt[self.language]['statement']}: " + row['conclusion']
                    input = Input(text=text)
                references = [
                    Reference(Output(text=self.prompt[self.language][row["label"]]), tags=[CORRECT_TAG]),
                ]
                random.shuffle(references) # Shuffle order of references
                instance = Instance(
                    input=input,
                    references=references, 
                    split=self.splits[split]
                )
                outputs.append(instance)
        return outputs