import datasets, random, os
import pandas as pd
from typing import List

from .scenario import Scenario, Instance, Reference, TRAIN_SPLIT, VALID_SPLIT, TEST_SPLIT, CORRECT_TAG, Input, Output, PassageQuestionInput
from helm.common.general import ensure_file_downloaded

# NLU

class XQuAD_VI_QA_Scenario(Scenario):
    """
    This is a Vietnamese question answer scenario. The data comes from XQuAD, and the dataset consists of a subset of
    240 paragraphs and 1190 question-answer pairs from the development set of SQuAD v1.1 together (Rajpurkar et al., 2016).

    The models are prompted using the following format:

        Bạn sẽ được cho một đoạn văn và một câu hỏi.
        Trả lời câu hỏi bằng cách trích xuất câu trả lời từ đoạn văn
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

    name = "xquad_vi_qa"
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
                passage = row["context"]
                question = row["question"]
                input = PassageQuestionInput(
                    passage=passage,
                    question=question,
                    passage_prefix="Đoạn văn: ",
                    question_prefix="Câu hỏi: ",
                )
                output = Output(text=row["answers"]["text"][0])
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

class XQuAD_TH_QA_Scenario(Scenario):
    """
    This is a Thai question answer scenario. The data comes from XQuAD, and the dataset consists of a subset of
    240 paragraphs and 1190 question-answer pairs from the development set of SQuAD v1.1 together (Rajpurkar et al., 2016).

    The models are prompted using the following format:

        คุณจะได้รับข้อความและคําถาม กรุณาตอบคําถาม
        โดยแยกคําตอบจากข้อความ
        ข้อความ: <text>
        คําถาม: <question>
        คําตอบ:

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

    name = "xquad_th_qa"
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
                passage = row["context"]
                question = row["question"]
                input = PassageQuestionInput(
                    passage=passage,
                    question=question,
                    passage_prefix="ข้อความ: ",
                    question_prefix="คําถาม: ",
                )
                output = Output(text=row["answers"]["text"][0])
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

class UIT_VSFC_VI_SA_Scenario(Scenario):
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

    name = "uit_vsfc_vi_sa"
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
            0: 'positive',
            1: 'negative',
            2: 'neutral',
        }

        self.label2id = {
            'positive': 0,
            'negative': 1,
            'neutral': 2,
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
                        data[split][file].append(line)
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

class NusaX_ID_SA_Scenario(Scenario):
    """
    This is an Indonesian sentiment analysis scenario. The data consists of comments and reviews from the 
    IndoNLU benchmark. Labels are positive, negative or neutral.

    The models are prompted using the following format:

        Apa sentimen dari kalimat berikut ini? <sentence>
        Jawab dengan satu kata saja: Positif/Negatif/Netral

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

    name = "nusax_id_sa"
    description = "NusaX-Senti sentiment analysis dataset"
    tags = ["sentiment_analysis"]

    def __init__(self):
        super().__init__()
        self.splits = {
            'train': TRAIN_SPLIT,
            'valid': VALID_SPLIT,
            'test': TEST_SPLIT
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
                input = Input(row["text"])
                output = Output(text=row["label"])
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
    
class IndicSentiment_TA_SA_Scenario(Scenario):
    """
    This is a Tamil sentiment analysis scenario. The data comes from IndicXTREME, and consists of product reviews
    that were written by annotators. Labels are positive or negative. For this scenario, the `validation` split is
    used as the `train` split for in-context examples. 

    The models are prompted using the following format:

        பின் வரும் வாக் கியத் தில்
         ெவளிப் படுத் தப் படும்
        உணர் வு எது? <sentence>
        ஒரு ெசால் லில் மட் டும் பதிலளிக் கவும் :
         ேநர் மைற/எதிர் மைற

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

    name = "indicsentiment_ta_sa"
    description = "IndicSentiment sentiment analysis dataset"
    tags = ["sentiment_analysis"]

    def __init__(self):
        super().__init__()
        self.splits = {
            'train': TRAIN_SPLIT,
            'test': TEST_SPLIT
        }

    def download_dataset(self, output_path: str):
        # For this scenario, the `validation` split is used as the `train` split for in-context examples. 
        URLS = {
            "test": "https://huggingface.co/datasets/ai4bharat/IndicSentiment/resolve/main/data/test/ta.json",
            "train": "https://huggingface.co/datasets/ai4bharat/IndicSentiment/resolve/main/data/validation/ta.json",
        }

        data = {}
        for split in list(URLS.keys()):
            data[split] = []
            target_path_file = os.path.join(output_path, split)
            ensure_file_downloaded(source_url=URLS[split], target_path=target_path_file)
            data[split] = pd.read_json(target_path_file, lines=True)
        return data

    def get_instances(self, output_path) -> List[Instance]:
        data = self.download_dataset(output_path)
        outputs = []
        for split in list(data.keys()):
            for index, row in data[split].iterrows():
                input = Input(row["INDIC REVIEW"])
                output = Output(text=row["LABEL"])
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
      
class MLHSD_ID_TD_Scenario(Scenario):
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

    name = "mlhsd_id_td"
    description = "MLHSD toxicity detection dataset"
    tags = ["toxicity_dectection"]

    def __init__(self):
        super().__init__()
        self.splits = {
            'train': TRAIN_SPLIT,
            'test': TEST_SPLIT
        }

        self.id2label = {
            0: 'CLEAN',
            1: 'OFFENSIVE',
            2: 'HATE',
        }

        self.label2id = {
            'CLEAN': 0,
            'OFFENSIVE': 1,
            'HATE': 2,
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
                input = Input(row["Tweet"])
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

class ViHSD_VI_TD_Scenario(Scenario):
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

    name = "vihsd_vi_td"
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
            0: 'CLEAN',
            1: 'OFFENSIVE',
            2: 'HATE',
        }

        self.label2id = {
            'CLEAN': 0,
            'OFFENSIVE': 1,
            'HATE': 2,
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
                input = Input(row["free_text"])
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

class Thai_Toxicity_Tweets_TH_TD_Scenario(Scenario):
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

    name = "thai_toxicity_tweets_th_td"
    description = "Thai Toxicity Tweets toxicity detection dataset"
    tags = ["toxicity_detection"]

    def __init__(self):
        super().__init__()
        self.splits = {
            'train': TRAIN_SPLIT,
            'test': TEST_SPLIT
        }

        self.id2label = {
            0: 'neg',
            1: 'pos',
        }

        self.label2id = {
            'neg': 0,
            'pos': 1,
        }

    def get_instances(self, output_path) -> List[Instance]:
        dataset = datasets.load_dataset("thai_toxicity_tweet")
        dataset = dataset['train'].train_test_split(test_size=0.8)

        outputs = []
        for split in list(dataset.keys()):
            data = dataset[split].to_pandas()
            for index, row in data.iterrows():
                if row["tweet_text"] != "" and row["tweet_text"] != "TWEET_NOT_FOUND":
                    input = Input(row["tweet_text"])
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



# NLG

# NLR

# LD