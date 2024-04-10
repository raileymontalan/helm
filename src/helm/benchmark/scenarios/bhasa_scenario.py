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

class XQuAD_QA_VI_Scenario(Scenario):
    """
    This is a Vietnamese question answer scenario. The data comes from XQuAD, and the dataset consists of a subset of
    240 paragraphs and 1190 question-answer pairs from the development set of SQuAD v1.1 together (Rajpurkar et al., 2016).

    The models are prompted using the following format:

        Bạn sẽ được cho một đoạn văn và một câu hỏi. Trả lời câu hỏi bằng cách trích xuất câu trả lời từ đoạn văn.
        Đoạn văn: <text>
        Câu hỏi: <question>
        Câu trả lời: 

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

    name = "xquad_qa_vi"
    description = "XQuAD question answering dataset"
    tags = ["question_answering"]

    def __init__(self):
        super().__init__()
        self.splits = {
            'train': TRAIN_SPLIT,
            'test': TEST_SPLIT
        }

    def get_instances(self, output_path) -> List[Instance]:
        dataset = datasets.load_dataset("xquad", "xquad.vi")
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
                    passage_prefix="Đoạn văn: ",
                    question_prefix="Câu hỏi: ",
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

class XQuAD_QA_TH_Scenario(Scenario):
    """
    This is a Thai question answer scenario. The data comes from XQuAD, and the dataset consists of a subset of
    240 paragraphs and 1190 question-answer pairs from the development set of SQuAD v1.1 together (Rajpurkar et al., 2016).

    The models are prompted using the following format:

        คุณจะได้รับข้อความและคำถาม กรุณาตอบคำถามโดยแยกคำตอบจากข้อความ
        ข้อความ: <text>
        คำถาม: <question>
        คำตอบ: 

    Target completion:
        <toxicity>

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

    name = "xquad_qa_th"
    description = "XQuAD question answering dataset"
    tags = ["question_answering"]

    def __init__(self):
        super().__init__()
        self.splits = {
            'train': TRAIN_SPLIT,
            'test': TEST_SPLIT
        }

    def get_instances(self, output_path) -> List[Instance]:
        dataset = datasets.load_dataset("xquad", "xquad.th")
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
                    passage_prefix="ข้อความ: ",
                    question_prefix="คำถาม: ",
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
    pass

class UIT_VSFC_SA_VI_Scenario(Scenario):
    """
    This is a Vietnamese sentiment analysis scenario. The data consists of student feedback obtained from 
    end-of-semester surveys at a Vietnamese university. Feedback is labeled as one of three sentiment 
    polarities: positive, negative or neutral.

    The models are prompted using the following format:

        Sắc thái của câu sau đây là gì? <sentence>
        Trả lời với một từ duy nhất: Tích cực/Tiêu cực/Trung lập

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
            0: 'Tích cực',
            1: 'Tiêu cực',
            2: 'Trung lập',
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

        Anda adalah pendeteksi ujaran kebencian.
        Definisi dari labelnya adalah sebagai berikut:
        Bersih: Tidak ada ujaran kebencian
        Kasar: Ada ujaran kebencian dan kata-kata kasar, namun tidak menyerang pihak tertentu
        Benci: Ada ujaran kebencian langsung atau serangan terhadap pihak tertentu
        Ketika diberikan sebuah kalimat, klasifikasikan dengan satu kata:
        Bersih/Kasar/Benci
        Kalimat: <text>

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
            0: 'Bersih',
            1: 'Kasar',
            2: 'Benci',
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
    
    def get_label(self, row) -> int:
        if int(row["HS"]) == 0:
            return self.id2label[int(row["Abusive"])]
        else: 
            return self.id2label[int(row["HS"])]

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

        คุณเป็นเครื่องตรวจจับข้อความที่เป็นพิษ
        ข้อความเป็นพิษถ้ามันทําร้าย
        สร้างความเสียหายหรือสื่อเจตนาเชิงลบถึงบุคคลหรือกลุ่มบุคคล
        คุณควรคํานึงถึงการประชดประชันด้วย
        เมื่อได้รับข้อความ ให้ตอบด้วยตัวอักษร Y หากเป็นพิษ และ N หากไม่เป็นพิษ
        ข้อความ: <text>

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
                if row["tweet_text"] != "" and row["tweet_text"] != "TWEET_NOT_FOUND":
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

        Bạn là máy phát hiện phát ngôn thù ghét.
        Các nhãn được định nghĩa như sau:
        Sạch: Không quấy rối
        Công kích: Bao gồm quấy rối và thậm chí chửi thề, nhưng không tấn công bất kì đối tượng cụ thể nào
        Thù ghét: Trực tiếp quấy rối hay lăng mạ một đối tượng cụ thể
        Cho câu dưới đây, hãy phân loại nó với 1 từ duy nhất:
        Sạch/Công kích/Thù ghét
        Câu: <text>

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
            0: 'Sạch',
            1: 'Công kích',
            2: 'Thù ghét',
        }

    def download_dataset(self, output_path: str):
        URL = "https://raw.githubusercontent.com/sonlam1102/vihsd/main/data/vihsd.zip"
        out_path = os.path.join(output_path, "vihsd")
        ensure_file_downloaded(source_url=URL, target_path=out_path, unpack=True)

        data = {}
        for split in self.splits.keys():
            data[split] = []
            target_path_file = os.path.join(output_path, "vihsd", "vihsd", split+".csv")
            data[split] = pd.read_csv(target_path_file)
        return data

    def get_instances(self, output_path) -> List[Instance]:
        data = self.download_dataset(output_path)
        outputs = []
        for split in list(data.keys()):
            for index, row in data[split].iterrows():
                input = Input(row["free_text"].strip())
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

# NLR

class IndicXNLI_NLI_TA_Scenario(Scenario):
    """
    This is a Tamil natural language inference scenario. The data comes from IndicXTREME, and consists of product reviews
    that were written by annotators. Labels are positive or negative. For this scenario, the `validation` split is
    used as the `train` split for in-context examples. 

    The models are prompted using the following format:

        பின் வரும் வாக் கியத் தில்
         ெவளிப் படுத் தப் படும்
        உணர் வு எது? <sentence>
        ஒரு ெசால் லில் மட் டும் பதிலளிக் கவும் :
         ேநர்மைற/எதிர் மைற

    Target completion:
        <sentiment> (<sentiment>:positive or negative or neutral)

    @article{Doddapaneni2022towards,
        title={Towards Leaving No Indic Language Behind: Building Monolingual Corpora, Benchmark and Models for Indic Languages},
        author={Sumanth Doddapaneni and Rahul Aralikatte and Gowtham Ramesh and Shreyansh Goyal and Mitesh M. Khapra and Anoop Kunchukuttan and Pratyush Kumar},
        journal={ArXiv},
        year={2022},
        volume={abs/2212.05409}
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
            for index, row in split.iterrows():
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
        X: {sentence1}
        Y: {sentence2}
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
    
class XNLI_NLI_VI_Scenario(Scenario):
    """
    This is a XNLI natural language inference scenario. The data comes from XNLI, and incorporates various linguistic 
    phenomena such as numerical reasoning, structural changes, idioms, or temporal and spatial reasoning. Labels are
    entailment, neutral, or contradiction. 

    The models are prompted using the following format:

        Bạn sẽ được cho hai câu, X và Y.
        X: {sentence1}
        Y: {sentence2}
        Xác định câu nào sau đây là câu phù hợp nhất cho câu X và Y.
        A: Nếu X đúng thì Y phải đúng.
        B: X mâu thuẫn với Y.
        C: Khi X đúng, Y có thể đúng hoặc không đúng.
        Trả lời với một chữ cái duy nhất A, B, hoặc C.
        Câu trả lời: 

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

    name = "xnli_nli_vi"
    description = "XNLI natural language inference dataset"
    tags = ["textual_entailment"]

    def __init__(self):
        super().__init__()
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
        dataset = datasets.load_dataset("xnli", "vi")

        outputs = []
        for split in list(dataset.keys()):
            data = dataset[split].to_pandas()
            for index, row in data.iterrows():
                passage = "X: " + row["premise"] + "\nY: " + row["hypothesis"]
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
    
class XNLI_NLI_TH_Scenario(Scenario):
    """
    This is a XNLI natural language inference scenario. The data comes from XNLI, and incorporates various linguistic 
    phenomena such as numerical reasoning, structural changes, idioms, or temporal and spatial reasoning. Labels are
    entailment, neutral, or contradiction. 

    The models are prompted using the following format:

        Bạn sẽ được cho hai câu, X và Y.
        X: {sentence1}
        Y: {sentence2}
        Xác định câu nào sau đây là câu phù hợp nhất cho câu X và Y.
        A: Nếu X đúng thì Y phải đúng.
        B: X mâu thuẫn với Y.
        C: Khi X đúng, Y có thể đúng hoặc không đúng.
        Trả lời với một chữ cái duy nhất A, B, hoặc C.
        Câu trả lời: 

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

    name = "xnli_nli_th"
    description = "XNLI natural language inference dataset"
    tags = ["textual_entailment"]

    def __init__(self):
        super().__init__()
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
        dataset = datasets.load_dataset("xnli", "th")

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

# LD
    
class LINDSEA_MP_ID_Scenario(Scenario):
    """
    This is a Indonesian minimal pairs (linguistic diagnostic for syntax) scenario. The data comes from the BHASA LINDSEA dataset.

    The models are prompted using the following format:

        System Prompt:
        Anda adalah seorang ahli bahasa Indonesia
        Human Prompt:
        Kalimat mana yang lebih mungkin?
        A: {SENTENCE 1}
        B: {SENTENCE 2}
        Jawablah dengan menggunakan A atau B saja.

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

    name = "lindsea_mp_id"
    description = "Indonesian minimal pairs dataset"
    tags = ["minimal_pairs", "linguistic_diagnostic", "syntax"]

    def __init__(self):
        super().__init__()
        self.splits = {
            'train': TRAIN_SPLIT,
            'test': TEST_SPLIT
        }

    def download_dataset(self, output_path: str):
        URLS = {
            "npis_and_negation": "https://raw.githubusercontent.com/aisingapore/BHASA/main/lindsea/id/syntax/NPIs_and_negation.jsonl",
            "argument_structure": "https://raw.githubusercontent.com/aisingapore/BHASA/main/lindsea/id/syntax/argument_structure.jsonl",
            "filler_gap_dependencies": "https://raw.githubusercontent.com/aisingapore/BHASA/main/lindsea/id/syntax/filler-gap_dependencies.jsonl",
            "morphology": "https://raw.githubusercontent.com/aisingapore/BHASA/main/lindsea/id/syntax/morphology.jsonl",
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
                input = Input(text="Kalimat mana yang lebih mungkin?")
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