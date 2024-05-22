import pandas as pd
import torch
from torch.utils.data import DataLoader
from typing import List, Union
from utils.utils import dir_path, Encoder
from utils.constant import strings as s
import logging

logging.basicConfig(format='(%(asctime)s) - [%(levelname)s] : %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class ContentsBasedRecommendation(object):
    def __int__(self,
                load_path: str,
                model_name: str,
                model,
                user: list,
                queries: Union[str, List[str]],
                threshold: float = 0.6,
                rank: int = 10
                ):

        dataset = ContentsBasedRecommendationDataset(
            model=model,
            user=user,
            queries=queries,
            threshold=threshold,
            rank=rank
        )
        self.load_path = load_path
        self.model_name = model_name
        self.dataloader = DataLoader(dataset)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.response = dataset.response

    def topk_recommendation(self):
        path = dir_path(self.load_path) + '/' + self.model_name
        model = torch.load(path)
        model.eval()
        predicts = list()

        with torch.no_grad():
            for fields in self.dataloader:
                fields = fields.to(self.device)
                y = model(fields)
                predicts.extend(y.tolist())

        result = pd.merge(
            self.response,
            pd.DataFrame(predicts, columns=[s.PREDICT]),
            left_index=True,
            right_index=True
        )

        top = result.nlargest(2, s.PREDICT)
        mask = ~result.index.isin(top.index)
        result = pd.concat([top, result[mask]])
        logger.info(result.to_numpy())

        return result.to_numpy()

class ContentsBasedRecommendationDataset(torch.utils.data.Dataset):
    """
    The sequence of input parameters should follow the rule to the code.
    """
    def __init__(self,
                 model,
                 user: list,
                 queries: Union[str, List[str]],
                 threshold: float,
                 rank: int
                 ):
        """
        :param model: Similarity Model
        :param user: User inputs : ['parent_key', 'age', 'gender']
        :param queries: Content inputs : ['recipe_name' or 'title']
        """
        super().__init__()
        self.user = user
        self.similarity = model.search(queries=queries, threshold=threshold, rank=rank)  # (title, key, category, timecost, level, score)
        self.items = self.__preprocess_input()
        self.response = self.response()

    def __len__(self):
        return self.items.shape[0]

    def __getitem__(self, index):
        return self.items[index]

    def __preprocess_input(self):
        concat = pd.DataFrame(
            [self.user + i[1:-1] for i in self.similarity],
            columns=[s.PARENT_KEY, s.AGE, s.GENDER, s.KEY, s.CATEGORY, s.TIMECOST, s.LEVEL]
        )
        data = Encoder(df=concat, save=False)
        return data.to_numpy().astype(int)

    def response(self):
        return pd.DataFrame(
            [self.user + i for i in self.similarity],
            columns=[s.PARENT_KEY, s.AGE, s.GENDER, s.TITLE, s.KEY, s.CATEGORY, s.TIMECOST, s.LEVEL, s.SCORE]
        )