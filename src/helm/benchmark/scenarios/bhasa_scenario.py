import datasets
import os
import random
from typing import List

import pandas as pd

from helm.benchmark.scenarios.scenario import (
    Input, Instance, Output, PassageQuestionInput, Reference, Scenario,
    CORRECT_TAG, TEST_SPLIT, TRAIN_SPLIT
)
from helm.common.general import ensure_file_downloaded

# BHASA Scenarios
#   A. Natural Language Understanding
#   B. Natural Language Generation
#   C. Natural Language Reasoning
#   D. Linguistic Diagnostics

# A. Natural Language Understanding
#   1. Question Answering
#   2. Sentiment Analysis
#   3. Toxicity Detection/Classification

# 1. Question Answering
# 1.1 Indonesian: TyDiQA
class TyDiQAScenario(Scenario):
    """
    TyDiQA is is an open-book question answering scenario for 11 typologically-diverse languages.
    The questions are written by people who want to know the answer, but do not know the answer yet,
    and the data is collected directly in each language without the use of translation.

    This scenario only uses the Indonesian subset of the data, and uses the Gold Passage (GoldP) task,
    which requires the tested system to extract a span from the given passage to answer a given question.
    There are no unanswerable questions.

    The models are prompted using the following format:

        Anda akan diberikan sebuah paragraf dan sebuah pertanyaan. Jawablah pertanyaannya dengan mengekstrak jawaban dari paragraf tersebut.

        Paragraf: <text>
        Pertanyaan: <question>
        Jawaban: <answer>

        ...

        Paragraf: <text>
        Pertanyaan: <question>
        Jawaban:


    Target completion:
        <answer>

    @article{clark-etal-2020-tydi,
        title = "{T}y{D}i {QA}: A Benchmark for Information-Seeking Question Answering in Typologically Diverse Languages",
        author = "Clark, Jonathan H.  and
        Choi, Eunsol  and
        Collins, Michael  and
        Garrette, Dan  and
        Kwiatkowski, Tom  and
        Nikolaev, Vitaly  and
        Palomaki, Jennimaria",
        editor = "Johnson, Mark  and
        Roark, Brian  and
        Nenkova, Ani",
        journal = "Transactions of the Association for Computational Linguistics",
        volume = "8",
        year = "2020",
        address = "Cambridge, MA",
        publisher = "MIT Press",
        url = "https://aclanthology.org/2020.tacl-1.30",
        doi = "10.1162/tacl_a_00317",
        pages = "454--470",
    }
    """

    name = "tydiqa_id"
    description = "Indonesian Open-book Question Answering task"
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
        for split in self.splits.keys():
            df = dataset[split].to_pandas()

            if split == "train":
                # Select only bottom 20th percentile by length for in-context examples as examples are very long
                data = df[df["passage_text"].apply(len) < df["passage_text"].apply(len).quantile(.2)]
            else:
                # Sample 100 examples for test
                data = df.sample(n=100, random_state=5018)

            for _, row in data.iterrows():
                passage = row["passage_text"].strip()
                question = row["question_text"].strip()
                input = PassageQuestionInput(
                    passage=passage,
                    question=question,
                    passage_prefix="Paragraf: ",
                    question_prefix="Pertanyaan: ",
                )
                references = []
                for answer in row["answers"]["text"]:
                    output = Output(text=answer.strip())
                    references.append(Reference(output, tags=[CORRECT_TAG]))
                instance = Instance(
                    input=input,
                    references=references,
                    split=self.splits[split]
                )
                outputs.append(instance)
        return outputs

# 1.2 Vietnamese & Thai: XQuAD
class XQuADScenario(Scenario):
    """
    XQuAD is an open-book question answering scenario that is parallel across 10 languages.
    The dataset consists of a subset of 240 paragraphs and 1190 question-answer pairs from the
    development set of SQuAD v1.1 (Rajpurkar et al., 2016) together with their professional translations.

    This scenario only uses the Vietnamese and Thai subsets of the data and there are no
    unanswerable questions.

    The models are prompted using the following general format:

        You will be given a paragraph and a question. Answer the question by extracting the answer from the paragraph.

        Paragraph: <text>
        Question: <question>
        Answer: <answer>

        ...

        Paragraph: <text>
        Question: <question>
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

    name = "xquad"
    description = "Vietnamese and Thai Open-book Question Answering task"
    tags = ["question_answering"]

    def __init__(self, language: str):
        super().__init__()
        self.language = language
        self.splits = {
            'train': TRAIN_SPLIT,
            'test': TEST_SPLIT
        }
        self.map = {
            "th": {
                "passage_prefix": "ข้อความ: ",
                "question_prefix": "คำถาม: ",
                "random_state": 4520,
            },
            "vi": {
                "passage_prefix": "Đoạn văn: ",
                "question_prefix": "Câu hỏi: ",
                "random_state": 4502,
            }
        }

    def get_instances(self, output_path) -> List[Instance]:
        dataset = datasets.load_dataset("xquad", f"xquad.{self.language}", split="validation")
        df = dataset.to_pandas()

        # Sample 100 examples for test
        df_test = df.sample(n=100, random_state=self.map[self.language]["random_state"])

        # In-context examples to be drawn from remaining examples (since there is no train data)
        df_train = df[~df.index.isin(df_test.index)]

        # Select only bottom 20th percentile by length for in-context examples as examples are very long
        df_train = df_train[df_train["context"].apply(len) < df_train["context"].apply(len).quantile(.2)]
        dataset = {
            'train': df_train,
            'test': df_test,
        }

        outputs = []
        for split in self.splits.keys():
            data = dataset[split]
            for _, row in data.iterrows():
                passage = row["context"].strip()
                question = row["question"].strip()
                input = PassageQuestionInput(
                    passage=passage,
                    question=question,
                    passage_prefix=self.map[self.language]['passage_prefix'],
                    question_prefix=self.map[self.language]['question_prefix'],
                )
                references = []
                for answer in row["answers"]["text"]:
                    output = Output(text=answer.strip())
                    references.append(Reference(output, tags=[CORRECT_TAG]))
                instance = Instance(
                    input=input,
                    references=references,
                    split=self.splits[split]
                )
                outputs.append(instance)
        return outputs

# 1.3 Tamil: IndicQA
class IndicQAScenario(Scenario):
    """
    IndicQA is an open-book question answering scenario for 11 Indic languages.
    Answers to questions are to be extracted from the text provided. The data is taken from
    Wikipedia articles across various domains and questions and answers were manually created
    by native speakers.

    This scenario only uses the Tamil subset of the data and unanswerable questions
    are removed from the dataset in order to be consistent with the question answering
    scenarios for Indonesian, Vietnamese and Thai.

    The models are prompted using the following format:

        உங்களுக்கு ஒரு பத்தியும் ஒரு கேள்வியும் தரப்படும். தரப்பட்ட பத்தியிலிருந்து கேள்விக்கான பதிலைக் கண்டறியவும்.

        பத்தி: <text>
        கேள்வி: <question>
        பதில்: <answer>

        ...

        பத்தி: <text>
        கேள்வி: <question>
        பதில்:

    Target completion:
        <answer>

    @inproceedings{doddapaneni-etal-2023-towards,
        title = "Towards Leaving No {I}ndic Language Behind: Building Monolingual Corpora, Benchmark and Models for {I}ndic Languages",
        author = "Doddapaneni, Sumanth  and
            Aralikatte, Rahul  and
            Ramesh, Gowtham  and
            Goyal, Shreya  and
            Khapra, Mitesh M.  and
            Kunchukuttan, Anoop  and
            Kumar, Pratyush",
        editor = "Rogers, Anna  and
            Boyd-Graber, Jordan  and
            Okazaki, Naoaki",
        booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
        month = jul,
        year = "2023",
        address = "Toronto, Canada",
        publisher = "Association for Computational Linguistics",
        url = "https://aclanthology.org/2023.acl-long.693",
        doi = "10.18653/v1/2023.acl-long.693",
        pages = "12402--12426",
    }
    """

    name = "indicqa"
    description = "Tamil Open-book Question Answering task"
    tags = ["question_answering"]

    def __init__(self):
        super().__init__()
        self.splits = {
            'train': TRAIN_SPLIT,
            'test': TEST_SPLIT
        }

    def get_instances(self, output_path) -> List[Instance]:
        dataset = datasets.load_dataset("ai4bharat/IndicQA", "indicqa.ta", split="test")
        df = dataset.to_pandas()

        # Remove unanswerable questions (answer is an empty string)
        df = df[df["answers"].apply(lambda x: len(x["text"][0].strip()) > 0)]

        # Sample 100 examples for test
        df_test = df.sample(n=100, random_state=7900)

        # In-context examples to be drawn from remaining examples (since there is no train/dev data)
        df_train = df[~df.index.isin(df_test.index)]

        # Select only bottom 20th percentile by length for in-context examples as examples are very long
        df_train = df_train[df_train["context"].apply(len) < df_train["context"].apply(len).quantile(.2)]
        dataset = {
            'train': df_train,
            'test': df_test,
        }

        outputs = []
        for split in self.splits.keys():
            data = dataset[split]
            for _, row in data.iterrows():
                passage = row["context"].strip()
                question = row["question"].strip()
                input = PassageQuestionInput(
                    passage=passage,
                    question=question,
                    passage_prefix="பத்தி: ",
                    question_prefix="கேள்வி: ",
                )
                references = []
                for answer in row["answers"]["text"]:
                    output = Output(text=answer.strip())
                    references.append(Reference(output, tags=[CORRECT_TAG]))
                instance = Instance(
                    input=input,
                    references=references,
                    split=self.splits[split]
                )
                outputs.append(instance)
        return outputs

# 2. Sentiment Analysis
# 2.1 Indonesian: NusaX Sentiment
class NusaXScenario(Scenario):
    """
    This is an Indonesian sentiment analysis scenario. The data consists of comments and reviews from the
    IndoNLU benchmark. Labels are positive, negative or neutral.

    The models are prompted using the following format:

        Apa sentimen dari kalimat berikut ini?
        Jawablah dengan satu kata saja:
        - Positif
        - Negatif
        - Netral

        Kalimat: <text>
        Jawaban: <sentiment>

        ...

        Kalimat: <text>
        Jawaban:

    Target completion:
        <sentiment>

    @inproceedings{winata-etal-2023-nusax,
        title = "{N}usa{X}: Multilingual Parallel Sentiment Dataset for 10 {I}ndonesian Local Languages",
        author = "Winata, Genta Indra  and
            Aji, Alham Fikri  and
            Cahyawijaya, Samuel  and
            Mahendra, Rahmad  and
            Koto, Fajri  and
            Romadhony, Ade  and
            Kurniawan, Kemal  and
            Moeljadi, David  and
            Prasojo, Radityo Eko  and
            Fung, Pascale  and
            Baldwin, Timothy  and
            Lau, Jey Han  and
            Sennrich, Rico  and
            Ruder, Sebastian",
        editor = "Vlachos, Andreas  and
            Augenstein, Isabelle",
        booktitle = "Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics",
        month = may,
        year = "2023",
        address = "Dubrovnik, Croatia",
        publisher = "Association for Computational Linguistics",
        url = "https://aclanthology.org/2023.eacl-main.57",
        doi = "10.18653/v1/2023.eacl-main.57",
        pages = "815--834",
    }
    """

    name = "nusax"
    description = "NusaX-Senti sentiment analysis dataset"
    tags = ["sentiment_analysis"]

    def __init__(self):
        super().__init__()
        self.splits = {
            'train': TRAIN_SPLIT,
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
        }

        dataset = {}
        for split in self.splits.keys():
            target_path_file = os.path.join(output_path, split)
            ensure_file_downloaded(source_url=URLS[split], target_path=target_path_file)
            data = pd.read_csv(target_path_file)
            dataset[split] = data
        return dataset

    def get_instances(self, output_path) -> List[Instance]:
        dataset = self.download_dataset(output_path)
        outputs = []
        for split in self.splits.keys():
            data = dataset[split]
            for _, row in data.iterrows():
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

# 2.2 Vietnamese: UIT-VSFC
class UITVSFCScenario(Scenario):
    """
    This is a Vietnamese sentiment analysis scenario. The data consists of student feedback obtained from
    end-of-semester surveys at a Vietnamese university. Feedback is labeled as one of three sentiment
    polarities: positive, negative or neutral.

    The models are prompted using the following format:

        Sắc thái của câu sau đây là gì?
        Trả lời với một từ duy nhất:
        - Tích cực
        - Tiêu cực
        - Trung lập

        Câu văn: <text>
        Câu trả lời: <sentiment>

        ...

        Câu văn: <text>
        Câu trả lời:

    Target completion:
        <sentiment>

    @inproceedings{van2018uit,
        title={UIT-VSFC: Vietnamese students’ feedback corpus for sentiment analysis},
        author={Van Nguyen, Kiet and Nguyen, Vu Duc and Nguyen, Phu XV and Truong, Tham TH and Nguyen, Ngan Luu-Thuy},
        booktitle={2018 10th international conference on knowledge and systems engineering (KSE)},
        pages={19--24},
        year={2018},
        organization={IEEE},
        url={https://ieeexplore.ieee.org/document/8573337},
    }
    """

    name = "uitvsfc"
    description = "BHASA Vietnamese Students' Feedback Corpus for sentiment analysis"
    tags = ["sentiment_analysis"]

    def __init__(self):
        super().__init__()
        self.splits = {
            'train': TRAIN_SPLIT,
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
            "test": {
                "sentences": "https://drive.google.com/uc?id=1aNMOeZZbNwSRkjyCWAGtNCMa3YrshR-n&export=download",
                "sentiments": "https://drive.google.com/uc?id=1vkQS5gI0is4ACU58-AbWusnemw7KZNfO&export=download",
            },
        }

        dataset = {}
        for split in list(URLS.keys()):
            dataset[split] = {}
            for file in list(URLS[split].keys()):
                dataset[split][file] = []
                target_path_file = os.path.join(output_path, split, file)
                ensure_file_downloaded(source_url=URLS[split][file], target_path=target_path_file)
                with open(target_path_file, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        dataset[split][file].append(str(line).strip())
            data = pd.DataFrame({
                "text": dataset[split]['sentences'],
                "label": dataset[split]['sentiments']
            })
            dataset[split] = data
        return dataset

    def get_instances(self, output_path) -> List[Instance]:
        dataset = self.download_dataset(output_path)
        outputs = []
        for split in self.splits.keys():
            data = dataset[split]
            for _, row in data.iterrows():
                input = Input(row['text'])
                output = Output(text=self.id2label[int(row['label'])])
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

# 2.3 Thai: Wisesight Sentiment
class WisesightScenario(Scenario):
    """
    This is an Thai sentiment analysis scenario. The data consists of social media messages regarding
    consumer products and services. Labels are positive, negative or neutral.

    The models are prompted using the following format:

        อารมณ์ความรู้สึกของข้อความต่อไปนี้เป็นอย่างไร?
        กรุณาตอบโดยใช้คำเดียวเท่านั้น:
        - แง่บวก
        - แง่ลบ
        - เฉยๆ

        ข้อความ: <text>
        คำตอบ: <sentiment>

        ...

        ข้อความ: <text>
        คำตอบ:

    Target completion:
        <sentiment>

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

    name = "wisesight"
    description = "Wisesight sentiment analysis dataset"
    tags = ["sentiment_analysis"]

    def __init__(self):
        super().__init__()
        self.splits = {
            'train': TRAIN_SPLIT,
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

        dataset = {}
        for split in self.splits.keys():
            target_path_file = os.path.join(data_path, "data", f"{split}.jsonl")
            df = pd.read_json(target_path_file, lines=True)
            df = df[df["category"] != "q"]
            if split == 'test':
                dataset[split] = df.groupby("category", group_keys=False).apply(lambda x: x.sample(frac=1000/len(df), random_state=4183))
            else:
                dataset[split] = df
        return dataset

    def get_instances(self, output_path) -> List[Instance]:
        dataset = self.download_dataset(output_path)
        outputs = []
        for split in self.splits.keys():
            data = dataset[split]
            for _, row in data.iterrows():
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

# 2.4 Tamil: IndicSentiment
class IndicSentimentScenario(Scenario):
    """
    This is a Tamil sentiment analysis scenario. The data comes from IndicXTREME, and consists of product reviews
    that were written by annotators. Labels are positive or negative.

    The models are prompted using the following format:

        பின்வரும் வாக்கியத்தில் வெளிப்படுத்தப்படும் உணர்வு எது?
        ஒரு சொல்லில் மட்டும் பதிலளிக்கவும்:
        - நேர்மறை
        - எதிர்மறை

        வாக்கியம்: <text>
        பதில்:

        ...

        வாக்கியம்: <text>
        பதில்: <answer>

    Target completion:
        <sentiment> (<sentiment>:positive or negative)

    @inproceedings{doddapaneni-etal-2023-towards,
        title = "Towards Leaving No {I}ndic Language Behind: Building Monolingual Corpora, Benchmark and Models for {I}ndic Languages",
        author = "Doddapaneni, Sumanth  and
            Aralikatte, Rahul  and
            Ramesh, Gowtham  and
            Goyal, Shreya  and
            Khapra, Mitesh M.  and
            Kunchukuttan, Anoop  and
            Kumar, Pratyush",
        editor = "Rogers, Anna  and
            Boyd-Graber, Jordan  and
            Okazaki, Naoaki",
        booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
        month = jul,
        year = "2023",
        address = "Toronto, Canada",
        publisher = "Association for Computational Linguistics",
        url = "https://aclanthology.org/2023.acl-long.693",
        doi = "10.18653/v1/2023.acl-long.693",
        pages = "12402--12426",
    }
    """

    name = "indicsentiment"
    description = "IndicSentiment sentiment analysis dataset"
    tags = ["sentiment_analysis"]

    def __init__(self):
        super().__init__()
        self.splits = {
            'validation': TRAIN_SPLIT,
            'test': TEST_SPLIT
        }
        self.sentiment2label = {
            'Positive': 'நேர்மறை',
            'Negative': 'எதிர்மறை',
        }

    def get_instances(self, output_path) -> List[Instance]:
        dataset = datasets.load_dataset("ai4bharat/IndicSentiment", "translation-ta")

        outputs = []
        for split in self.splits.keys():
            data = dataset[split].to_pandas()
            data["LABEL"] = data["LABEL"].fillna("Positive")
            for _, row in data.iterrows():
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

# 3. Toxicity Detection/Classification
# 3.1 Indonesian: Multi-Label Hate Speech Detection
class MLHSDScenario(Scenario):
    """
    This is an Indonesian toxicity detection scenario. The data comes from the Indonesian Twitter Multi-label Hate Speech and
    Abusive Language Detection Dataset.

    The models are prompted using the following format:

        Anda adalah pendeteksi ujaran kebencian. Definisi dari labelnya adalah sebagai berikut:
        Bersih: Tidak ada ujaran kebencian.
        Kasar: Ada ujaran kebencian dan kata-kata kasar, namun tidak menyerang pihak tertentu.
        Benci: Ada ujaran kebencian atau serangan langsung terhadap pihak tertentu.
        Berdasarkan definisi labelnya, klasifikasikan kalimat berikut ini dengan satu kata saja:
        - Bersih
        - Kasar
        - Benci

        Kalimat: <text>
        Jawaban: <answer>

        ...

        Kalimat: <text>
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

    name = "mlhsd"
    description = "MLHSD toxicity detection dataset"
    tags = ["toxicity_dectection"]

    def __init__(self):
        super().__init__()
        self.splits = {
            'train': TRAIN_SPLIT,
            'test': TEST_SPLIT
        }

    def download_dataset(self, output_path: str):
        URL = "https://raw.githubusercontent.com/okkyibrohim/id-multi-label-hate-speech-and-abusive-language-detection/master/re_dataset.csv"
        target_path_file = os.path.join(output_path, "mlhsd")
        ensure_file_downloaded(source_url=URL, target_path=target_path_file)
        df = pd.read_csv(target_path_file, encoding="ISO-8859-1")
        df['label'] = df.apply(lambda x: self.get_label(x), axis=1)
        df_test = df.groupby("label", group_keys=False).apply(lambda x: x.sample(frac=1000/len(df), random_state=7123))
        df_train = df[~df.apply(tuple,1).isin(df_test.apply(tuple,1))]
        dataset = {
            'train': df_train,
            'test': df_test,
        }
        return dataset

    def get_label(self, row) -> str:
        if int(row["HS"]) == 1:
            return "Benci"
        elif int(row["Abusive"]) == 1:
            return "Kasar"
        else:
            return "Bersih"

    def get_instances(self, output_path) -> List[Instance]:
        dataset = self.download_dataset(output_path)
        outputs = []
        for split in self.splits.keys():
            data = dataset[split]
            for _, row in data.iterrows():
                input = Input(row["Tweet"].strip())
                output = Output(text=row['label'])
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

# 3.2 Vietnamese: ViHSD
class ViHSDScenario(Scenario):
    """
    This is a Vietnamese toxicity detection scenario. The data comes from the ViHSD dataset.

    The models are prompted using the following format:

        Bạn là máy phát hiện phát ngôn thù ghét. Các nhãn được định nghĩa như sau:
        Sạch: Không quấy rối.
        Công kích: Bao gồm quấy rối và thậm chí chửi thề, nhưng không tấn công bất kì đối tượng cụ thể nào.
        Thù ghét: Trực tiếp quấy rối hay lăng mạ một đối tượng cụ thể.
        Với các định nghĩa của nhãn, hãy phân loại câu dưới đây với một từ duy nhất:
        - Sạch
        - Công kích
        - Thù ghét


        Câu văn: <text>
        Câu trả lời: <toxicity>

        ...

        Câu văn: <text>
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

    name = "vihsd"
    description = "ViHSD toxicity detection dataset"
    tags = ["toxicity_detection"]

    def __init__(self):
        super().__init__()
        self.splits = {
            'train': TRAIN_SPLIT,
            'test': TEST_SPLIT
        }
        self.id2label = {
            0: 'Sạch',
            1: 'Công kích',
            2: 'Thù ghét',
        }

    def download_dataset(self, output_path: str):
        URL = "https://raw.githubusercontent.com/sonlam1102/vihsd/main/data/vihsd.zip"
        data_path = os.path.join(output_path, "data")
        ensure_file_downloaded(source_url=URL, target_path=data_path, unpack=True)

        dataset = {}
        for split in self.splits.keys():
            target_path_file = os.path.join(data_path, "vihsd", f"{split}.csv")
            df = pd.read_csv(target_path_file)
            data = df.groupby("label_id", group_keys=False).apply(lambda x: x.sample(frac=1000/len(df), random_state=4878))
            dataset[split] = data
        return dataset

    def get_instances(self, output_path) -> List[Instance]:
        dataset = self.download_dataset(output_path)
        outputs = []
        for split in self.splits.keys():
            data = dataset[split]
            for _, row in data.iterrows():
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

# 3.3 Thai: Thai Toxicity Tweets
class ThaiToxicityTweetsScenario(Scenario):
    """
    This is a Thai toxicity detection scenario. The data comes from the Thai Toxicity Tweets dataset.

    The models are prompted using the following format:

        คุณเป็นเครื่องตรวจจับข้อความที่เป็นพิษ
        ข้อความเป็นพิษถ้ามันทำร้าย สร้างความเสียหาย หรือสื่อเจตนาเชิงลบถึงบุคคลหรือกลุ่มบุคคล และคุณควรคำนึงถึงการประชดประชันด้วย
        เมื่อได้รับข้อความ ให้ตอบด้วยตัวอักษร Y หากเป็นพิษ และ N หากไม่เป็นพิษ

        ข้อความ: <text>
        คำตอบ: <toxicity>

        ...

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

    name = "thaitoxicitytweets"
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
        df = dataset['train'].to_pandas()
        df = df[df["tweet_text"].str.len() > 0]
        df = df[df["tweet_text"]!= "TWEET_NOT_FOUND"]
        df_test = df.groupby("is_toxic", group_keys=False).apply(lambda x: x.sample(frac=1000/len(df), random_state=4156))
        df_train = df[~df.apply(tuple,1).isin(df_test.apply(tuple,1))]
        dataset = {
            'train': df_train,
            'test': df_test,
        }

        outputs = []
        for split in self.splits.keys():
            data = dataset[split]
            for _, row in data.iterrows():
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

# B. Natural Language Generation
#   1. Machine Translation
#   2. Abstractive Summarization

# 1. Machine Translation: FLoRes-200
class FloresScenario(Scenario):
    """
    This is the Flores machine translation scenario.

    The models are prompted using the following general format:

        Translate the following text into <language> language.

        Text: <text>
        Translation: <translation>

        ...

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

    name = "flores"
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
            data = source_df.join(target_df, lsuffix="_source", rsuffix="_target")
            for _, row in data.iterrows():
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

# 2. Abstractive Summarization: XL-Sum
class XLSumScenario(Scenario):
    """
    This is the XLSum abstractive summarization scenario.

    The models are prompted using the following general format:

        Summarize this <language> language article in 1 or 2 sentences. The answer must be written in <language> language.

        Article: <text>
        Summary: <summary>

        ...

        Article: <text>
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

    name = "xlsum"
    description = "XLSum abstractive summarization dataset"
    tags = ["abstractive_summarization"]

    def __init__(self, language: str):
        super().__init__()
        self.language = language

        self.splits = {
            'train': TRAIN_SPLIT,
            'test': TEST_SPLIT,
        }

        self.map = {
            "id": {
                "lang": "indonesian",
                "random_state": 6036,
            },
            "vi": {
                "lang": "vietnamese",
                "random_state": 8801,
            },
            "th": {
                "lang": "thai",
                "random_state": 10736,
            },
            "ta": {
                "lang": "tamil",
                "random_state": 5291,
            },
        }

    def get_instances(self, output_path) -> List[Instance]:
        dataset = datasets.load_dataset("csebuetnlp/xlsum", self.map[self.language]['lang'])

        outputs = []
        for split in self.splits.keys():
            df = dataset[split].to_pandas()
            if split == 'train':
                data = df[df["text"].apply(len) < df["text"].apply(len).quantile(.2)]
            else:
                data = df.sample(n=100, random_state=self.map[self.language]['random_state'])
            for _, row in data.iterrows():
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

# C. Natural Language Reasoning
#   1. Natural Language Inference
#   2. Causal Reasoning

# 1. Natural Language Inference
# 1.1 Indonesian: IndoNLI
class IndoNLIScenario(Scenario):
    """
    This is a IndoNLI natural language inference scenario. The data comes from IndoNLI, and incorporates various linguistic
    phenomena such as numerical reasoning, structural changes, idioms, or temporal and spatial reasoning. Labels are
    entailment, contradiction, or neutral.

    The models are prompted using the following format:

        Anda akan diberikan dua kalimat, X dan Y.
        Tentukan mana dari pernyataan berikut ini yang paling sesuai untuk kalimat X dan Y.
        A: Kalau X benar, maka Y juga harus benar.
        B: X bertentangan dengan Y.
        C: Ketika X benar, Y mungkin benar atau mungkin tidak benar.
        Jawablah dengan satu huruf saja, A, B atau C.

        X: <sentence1>
        Y: <sentence2>
        Jawaban: <entailment>

        ...

        X: <sentence1>
        Y: <sentence2>
        Jawaban:

    Target completion:
        <entailment>

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

    name = "indonli"
    description = "IndoNLI natural language inference dataset"
    tags = ["textual_entailment"]

    def __init__(self):
        super().__init__()
        self.splits = {
            'train': TRAIN_SPLIT,
            'test': TEST_SPLIT,
        }
        self.id2label = {
            'e': 'A',
            'c': 'B',
            'n': 'C'
        }

    def download_dataset(self, output_path: str):
        URLS = {
            'train': "https://raw.githubusercontent.com/ir-nlp-csui/indonli/main/data/indonli/train.jsonl",
            'test': "https://raw.githubusercontent.com/ir-nlp-csui/indonli/main/data/indonli/test_lay.jsonl"
        }

        dataset = {}
        for split in self.splits.keys():
            target_path_file = os.path.join(output_path, split)
            ensure_file_downloaded(source_url=URLS[split], target_path=target_path_file)
            df = pd.read_json(target_path_file, lines=True)
            if split == 'test':
                dataset[split] = df.groupby("label", group_keys=False).apply(lambda x: x.sample(frac=1000/len(df), random_state=4685))
            else:
                dataset[split] = df
        return dataset

    def get_instances(self, output_path) -> List[Instance]:
        dataset = self.download_dataset(output_path)
        outputs = []
        for split in self.splits.keys():
            data = dataset[split]
            for _, row in data.iterrows():
                passage = "X: " + row["premise"].strip() + "\nY: " + row["hypothesis"].strip()
                input = Input(passage)
                output = Output(self.id2label[row["label"]])
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

# 1.2 Vietnamese & Thai: XNLI
class XNLIScenario(Scenario):
    """
    This is a XNLI natural language inference scenario. The data comes from XNLI, and incorporates various linguistic
    phenomena such as numerical reasoning, structural changes, idioms, or temporal and spatial reasoning. Labels are
    entailment, neutral, or contradiction.

    The models are prompted using the following general format:

        You will be given two sentences, X and Y.
        Determine which of the following statements applies to sentences X and Y the best.
        A: If X is true, Y must be true.
        B: X contradicts Y.
        C: When X is true, Y may or may not be true.
        Answer strictly with a single letter A, B or C.

        X: <sentence1>
        Y: <sentence2>
        Answer: <entailment>

        ...

        X: <sentence1>
        Y: <sentence2>
        Answer:

    Target completion:
        <entailment>

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

    name = "xnli"
    description = "XNLI natural language inference dataset"
    tags = ["textual_entailment"]

    def __init__(self, language: str):
        super().__init__()
        self.language = language
        self.splits = {
            'validation': TRAIN_SPLIT,
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
        for split in self.splits.keys():
            df = dataset[split].to_pandas()
            if split == 'train':
                data = df
            else:
                data = df.groupby("label", group_keys=False).apply(lambda x: x.sample(frac=1000/len(df), random_state=4156))
                diff = df[~df.apply(tuple,1).isin(data.apply(tuple,1))]
                data = pd.concat([data, diff[diff["label"]==1].iloc[0].to_frame().transpose()], axis=0, ignore_index=True)
            for _, row in data.iterrows():
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

# 1.3 Tamil: IndicXNLI
class IndicXNLIScenario(Scenario):
    """
    This is a Tamil natural language inference scenario. The data was automatically translated from XNLI into 11 Indic languages.

    The models are prompted using the following format:

        உங்களுக்கு இரண்டு வாக்கியங்கள், X மற்றும் Y, தரப்படும்.
        பின்வரும் கூற்றுகளில் எது X மற்றும் Y வாக்கியங்களுடன் மிகப் பொருந்துகிறது எனக் கண்டறியவும்.
        A: X உண்மை என்றால் Y உம் உண்மையாக இருக்க வேண்டும்.
        B: X உம் Y உம் முரண்படுகின்றன.
        C: X உண்மையாக இருக்கும்போது Y உண்மையாக இருக்கலாம் அல்லது இல்லாமல் இருக்கலாம்.
        A அல்லது B அல்லது C என்ற ஒறே எழுத்தில் மட்டும் பதிலளிக்கவும்.

        X: <premise>
        Y: <hypothesis>
        பதில்: <entailment>

        ...

        X: <premise>
        Y: <hypothesis>
        பதில்:

    Target completion:
        <entailment>

    @inproceedings{aggarwal-etal-2022-indicxnli,
        title = "{I}ndic{XNLI}: Evaluating Multilingual Inference for {I}ndian Languages",
        author = "Aggarwal, Divyanshu  and
            Gupta, Vivek  and
            Kunchukuttan, Anoop",
        editor = "Goldberg, Yoav  and
            Kozareva, Zornitsa  and
            Zhang, Yue",
        booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
        month = dec,
        year = "2022",
        address = "Abu Dhabi, United Arab Emirates",
        publisher = "Association for Computational Linguistics",
        url = "https://aclanthology.org/2022.emnlp-main.755",
        doi = "10.18653/v1/2022.emnlp-main.755",
        pages = "10994--11006",
    }
    """

    name = "indicxnli"
    description = "IndicXNLI natural language inference dataset"
    tags = ["natural_language_inference"]

    def __init__(self):
        super().__init__()
        self.splits = {
            'train': TRAIN_SPLIT,
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
        for split in self.splits.keys():
            df = dataset[split].to_pandas()
            if split == 'train':
                data = df
            else:
                data = df.groupby("label", group_keys=False).apply(lambda x: x.sample(frac=1000/len(df), random_state=4156))
                diff = df[~df.apply(tuple,1).isin(data.apply(tuple,1))]
                data = pd.concat([data, diff[diff["label"]==2].iloc[0].to_frame().transpose()], axis=0, ignore_index=True)
            for _, row in data.iterrows():
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

# 2. Causal Reasoning: XCOPA
class XCOPAScenario(Scenario):
    """
    This is a XCOPA causal reasoning scenario. The data comes from XCOPA, a translation and reannotation of the English COPA.

    The models are prompted using the following general format:

        Based on the following situation, which of the following choices is most likely to be its {question}?
        Answer with only A or B.

        Situation: <premise>
        A: <choice1>
        B: <choice2>
        Answer: <answer>

        ...

        Situation: <premise>
        A: <choice1>
        B: <choice2>
        Answer:

    Target completion:
        <answer>

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

    name = "xcopa"
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
                "instruction2": "Jawablah dengan satu huruf A atau B saja.",
            },
            "ta": {
                "cause": "காரணமாக",
                "effect": "விளைவாக",
                "instruction1": "பின்வரும் வாக்கியங்களில் பெரும்பாலும் எது தரப்பட்ட சூழ்நிலைக்குரிய {} இருக்கும்?",
                "instruction2": "A அல்லது B என்ற ஒறே எழுத்தில் மட்டும் பதிலளிக்கவும்.",
            },
            "th": {
                "cause": "สาเหตุ",
                "effect": "ผล",
                "instruction1": "เมื่อพิจารณาจากสถานการณ์นี้ ตัวเลือกใดต่อไปนี้น่าจะเป็น{}มากกว่ากัน?",
                "instruction2": "กรุณาตอบด้วยตัวอักษร A หรือ B ตัวเดียวเท่านั้น",
            },
            "vi": {
                "cause": "nguyên nhân",
                "effect": "kết quả",
                "instruction1": "Với tình huống trên, lựa chọn nào dưới đây có khả năng cao là {} của nó hơn?",
                "instruction2": "Trả lời với một chữ cái duy nhất A hoặc B.",
            },
        }

    def get_instances(self, output_path) -> List[Instance]:
        language_dataset = datasets.load_dataset("xcopa", self.language)
        tamil_dataset = datasets.load_dataset("xcopa", 'ta')

        outputs = []
        for split in self.splits.keys():
            language_df = language_dataset[split].to_pandas()
            tamil_df = tamil_dataset[split].to_pandas()
            data = pd.merge(language_df, tamil_df[['question', 'idx']], on='idx') # Use the Tamil split's question column
            for _, row in data.iterrows():
                instruction1 = self.prompt[self.language]['instruction1'].format(self.prompt[self.language][row['question_y']])
                passage = "{premise}\n{instruction1}\nA: {choice1}\nB: {choice2}\n{instruction2}".format(
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

# D. Linguistic Diagnostics (LINDSEA)
#   1. Syntax: Minimal Pairs
#   2. Semantics: Pragmatic Reasoning (single sentece)
#   3. Semantics: Pragmatic Reasoning (sentence pair)

# 1. Syntax: LINDSEA Minimal Pairs
class LINDSEASyntaxMinimalPairsScenario(Scenario):
    """
    This is a LINDSEA minimal pairs (linguistic diagnostic for syntax) scenario. The data comes from the BHASA LINDSEA dataset.

    The models are prompted using the following general format:

        Which sentence is more acceptable?
        <sentence>

    Target completion:
        <sentence>

    @misc{leong2023bhasa,
        title={BHASA: A Holistic Southeast Asian Linguistic and Cultural Evaluation Suite for Large Language Models},
        author={Wei Qi Leong and Jian Gang Ngui and Yosephine Susanto and Hamsawardhini Rengarajan and Kengatharaiyer Sarveswaran and William Chandra Tjhi},
        year={2023},
        eprint={2309.06085},
        archivePrefix={arXiv},
        primaryClass={cs.CL}
    }
    """

    name = "lindseaminimalpairs"
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
        dataset = pd.concat(data_files)

        return dataset


    def get_instances(self, output_path) -> List[Instance]:
        data = self.download_dataset(output_path)
        dataset = datasets.Dataset.from_pandas(data).train_test_split(test_size=0.9, seed=5018)

        outputs = []
        for split in list(dataset.keys()):
            data = dataset[split].to_pandas()
            for _, row in data.iterrows():
                input = Input(text=self.prompt[self.language]['question'] + "\n\n")
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

# 2. Pragmatics: LINDSEA Pragmatic Reasoning (single sentence)
class LINDSEAPragmaticsPragmaticReasoningSingleScenario(Scenario):
    """
    This is a LINDSEA single-sentence pragmatic reasoning (linguistic diagnostic for pragmatics) scenario. The data comes from the BHASA LINDSEA dataset.

    The models are prompted using the following general format:

        Is the following statement true or false?
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

    name = "lindseapragmaticreasoningsingle"
    description = "LINDSEA pragmatic reasoning single sentence dataset"
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
                "question": "Apakah pernyataan berikut ini {}?",
                "instruction": "Jawablah dengan {} saja.",
            },
        }

    def download_dataset(self, output_path: str):
        URL = f"https://raw.githubusercontent.com/aisingapore/BHASA/main/lindsea/{self.language}/pragmatics/pragmatic_reasoning_single.jsonl"
        file = "pragmatic_reasoning_single"
        target_path_file = os.path.join(output_path, file)
        ensure_file_downloaded(source_url=URL, target_path=target_path_file)
        dataset = pd.read_json(target_path_file, lines=True)
        return dataset


    def get_instances(self, output_path) -> List[Instance]:
        data = self.download_dataset(output_path)
        dataset = datasets.Dataset.from_pandas(data).train_test_split(test_size=0.9, seed=5018)

        outputs = []
        for split in list(dataset.keys()):
            data = dataset[split].to_pandas()
            for _, row in data.iterrows():
                passage = "{question}\nPernyataan: {text}\n{instruction}".format(
                    question=self.prompt[self.language]['question'].format(row["question_translated"]),
                    text=row["text"],
                    instruction=self.prompt[self.language]['instruction'].format(row["choices_translated"]),
                )
                input = Input(text=passage)

                choices = row["choices"].split()
                choices_translated = row["choices_translated"].split()
                label2choice = {
                    choices[0]: choices_translated[0],
                    choices[2]: choices_translated[2],
                }
                references = [
                    Reference(Output(text=label2choice[row["label"].strip()]), tags=[CORRECT_TAG]),
                ]
                instance = Instance(
                    input=input,
                    references=references,
                    split=self.splits[split]
                )
                outputs.append(instance)
        return outputs

# 3. Pragmatics: LINDSEA Pragmatic Reasoning (sentence pair)
class LINDSEAPragmaticsPragmaticReasoningPairScenario(Scenario):
    """
    This is a LINDSEA pair-sentence pragmatic reasoning (linguistic diagnostic for syntax) scenario. The data comes from the BHASA LINDSEA dataset.

    The models are prompted using the following general format:

        Situation: <premise>
        Given this situation, is the following statement true or false?
        Statement: <hypothesis>
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

    name = "lindseapragmaticreasoningpair"
    description = "LINDSEA pragmatic reasoning sentence pair dataset"
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
                "question": "Berdasarkan situasi ini, apakah pernyataan berikut ini benar atau salah?",
                "instruction": "Jawablah dengan Benar atau Salah saja.",
                True: "Benar",
                False: "Salah",
            },
        }

    def download_dataset(self, output_path: str):
        URL = f"https://raw.githubusercontent.com/aisingapore/BHASA/main/lindsea/{self.language}/pragmatics/pragmatic_reasoning_pair.jsonl"
        file = "pragmatic_reasoning_pair"
        target_path_file = os.path.join(output_path, file)
        ensure_file_downloaded(source_url=URL, target_path=target_path_file)
        dataset = pd.read_json(target_path_file, lines=True)
        return dataset


    def get_instances(self, output_path) -> List[Instance]:
        data = self.download_dataset(output_path)
        dataset = datasets.Dataset.from_pandas(data).train_test_split(test_size=0.9, seed=5018)

        outputs = []
        for split in list(dataset.keys()):
            data = dataset[split].to_pandas()
            for _, row in data.iterrows():
                passage = "Situasi: {premise}\n{question}\nPernyataan: {conclusion}\n{instruction}".format(
                    premise=row["text"],
                    question=self.prompt[self.language]['question'],
                    conclusion=row["conclusion"],
                    instruction=self.prompt[self.language]['instruction'],
                )
                input = Input(text=passage)
                references = [
                    Reference(Output(text=self.prompt[self.language][row["label"]]), tags=[CORRECT_TAG]),
                ]
                instance = Instance(
                    input=input,
                    references=references,
                    split=self.splits[split]
                )
                outputs.append(instance)
        return outputs