import random, os
from typing import List

from helm.common.general import ensure_file_downloaded
from scenario import Scenario, Instance, Reference, TRAIN_SPLIT, VALID_SPLIT, TEST_SPLIT, CORRECT_TAG, Input, Output


class UITVSFCSentimentAnalysisScenario(Scenario):
    """
    This is a Vietnamese sentiment analysis scenario. The data consists of student feedback obtained from 
    end-of-semester surveys at a Vietnamese university. Feedback is labeled as one of three sentiment 
    polarities: positive, negative or neutral.

    The models are prompted using the following format:

        <passage>
        Sentiment:

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

    name = "uit_vsfc_sa"
    description = "BHASA Vietnamese Students' Feedback Corpus for sentiment analysis"
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
            for file in list(URLS[split].keys()):
                target_path_file = os.path.join(output_path, split, str(file + '.txt'))
                ensure_file_downloaded(source_url=URLS[split][file], target_path=target_path_file)
                with open(target_path_file, 'r') as f:
                    data[split][file] = f.readlines()
        return data
    
    def generate_instance(self, split, data):
        input = Input(data['sentence'])
        references = [Reference(Output(text=data['sentiment']), tags=[CORRECT_TAG])]
        return Instance(Input(text=input), references=references, split=self.splits[split]) 
                
    def get_instances(self, output_path) -> List[Instance]:

        data = self.download_dataset(output_path)
        output = []
        for split in list(data.keys()):
            output += [self.generate_instance(split, data[split])]
        
        return output


if __name__ == "__main__":
    print("DOWNLOADING")
    scenario = UITVSFCSentimentAnalysisScenario()
    scenario.download_dataset('data')