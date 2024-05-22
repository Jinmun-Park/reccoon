import logging
import torch
from sentence_transformers import SentenceTransformer, util
from datetime import datetime
from typing import List, Union

logging.basicConfig(format='(%(asctime)s) - [%(levelname)s] : %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class RunSBERT(object):
    def __int__(self,
                model_path: str,
                batch_size: int
                ):
        logger.info("Retrieving Sentence BERT Tokenizer and Model")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SentenceTransformer(model_path)
        self.batch_size = batch_size

    def corpus_embeddings(self,
                          sentence: Union[str, List[str], torch.Tensor]
                          ):
        self.sentence = sentence
        self.embeddings = self.model.encode(
            sentence[0],
            convert_to_tensor=True,
            batch_size=self.batch_size
        )

    def rank_similarity(self,
                        data:  Union[str, List[str], List[torch.Tensor]],
                        score,
                        threshold: float,
                        rank: int
                        ):
        """
        :param data: Import data where data[0] is 'title' data[1] is 'id'.
        :param score: Any scores that has same embedding size with data.
        :param threshold: cut-off similarity scores
        :param rank: Number of ranks
        :return: data[1]

        @ Citation: Sentence-Transformer
        """
        top_result = torch.topk(score, k=min(rank, len(data[0])))
        output = []

        for score, idx in zip(top_result[0], top_result[1]):
            if score > threshold:
                output.append([
                    data[0][idx],  # sentences | feature #1
                    data[1][idx],  # id        | feature #2
                    data[2][idx],  # category  | feature #3
                    data[3][idx],  # timecost  | feature #4
                    data[4][idx],  # level     | feature #5
                    '{:.2f}'.format(score.item())
                ])
        return output

    def search(self,
               queries: Union[str, List[str]],
               threshold: float = 0.6,
               rank: int = 5
               ):

        start = datetime.now()

        if isinstance(queries, list) and len(queries) > 1:
            combined_results = []
            for query in queries:
                results = self.search(queries=query)
                combined_results.append(results)
            return combined_results
        elif isinstance(queries, list):
            queries = queries[0]

        query_embedding = self.model.encode(queries, batch_size=self.batch_size)  # 768
        cos_scores = util.cos_sim(query_embedding, self.embeddings)[0]  # (768,) (1022, 768) => [1, 1022] => [1022]
        results = self.rank_similarity(data=self.sentence, score=cos_scores, threshold=threshold, rank=rank)

        end = datetime.now()
        time = end - start

        logger.info(results)
        logger.info("Time Spent : " + str(time) + " seconds")

        return results