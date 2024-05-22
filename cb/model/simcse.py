import logging
import torch
from numpy import ndarray
from tqdm import tqdm
from datetime import datetime
from typing import List, Union
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(format='(%(asctime)s) - [%(levelname)s] : %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class RunSimCSE(object):
    def __init__(self,
                 model_path: str,
                 batch_size: int
                 ):

        logger.info('Retrieving SimCSE Tokenizer and Model')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size

    def encode(self,
               sentence: Union[str, List[str]],
               return_numpy: bool = False,
               normlize_to_unit: bool = True,
               keepdim: bool = False
               ):
        """
        @ Description :
        output = (last_hidden_state, * pooled_output, hidden_states, attentions)
            * pooled_output :
                - Last layer hidden-state of the first token of the sequence(CLS) further processed by a linear layer
                    and a Tanh activation function.
                - The linear layer weights are trained from the next sentence prediction (classification) objective
                    during pre-training.
        truncation = <PAD> to max in batch
        """
        # Corpus & Query Identification
        single_sentence = False
        if isinstance(sentence, str):
            sentence = [sentence]
            single_sentence = True

        # Batch & Mini-batch
        embedding_list = []

        with torch.no_grad():
            total_batch = len(sentence) // self.batch_size + (1 if len(sentence) % self.batch_size > 0 else 0)
            for batch_id in tqdm(range(total_batch)):
                input_corpus = self.tokenizer(
                    sentence[batch_id * self.batch_size:(batch_id + 1) * self.batch_size],
                    padding=True,
                    truncation=True,
                    return_tensors='pt'
                )
                inputs = {k: v for k, v in input_corpus.items()}  # (input_ids, token_type_ids, attention_mask)
                outputs = self.model(**inputs, return_dict=True)  # (last_hs, pooled_output, hs, attentions)
                embeddings = outputs.pooler_output  # (b, 768)

                if normlize_to_unit:
                    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)

                embedding_list.append(embeddings.cpu())
            embeddings = torch.cat(embedding_list, 0)

            if single_sentence and not keepdim:
                embeddings = embeddings[0]  # (1, 768) => (768)

            if return_numpy and not isinstance(embeddings, ndarray):
                return embeddings.detach().numpy()  # => ndarray

            return embeddings

    def corpus_embeddings(self,
                          sentence: Union[str, List[str]]
                          ):
        embeddings = self.encode(sentence[0], normlize_to_unit=True, return_numpy=True)
        self.index = {
            'sentences': sentence[0],
            'id': sentence[1],
            'category': sentence[2],
            'timecost': sentence[3],
            'level': sentence[4],
            'embeddings': embeddings
        }

    def similarity(self,
                   queries: Union[str, List[str]],
                   keys: Union[str, List[str], ndarray]) -> Union[float, ndarray]:
        """
        @ Example:
            key_vecs = key  # (1022, 768)
            query_vecs = self.encode(queries, return_numpy=True)  # (768)
            query_vecs = query_vecs.reshape(1, -1)  # (1, 768)
            similarities = cosine_similarity(query_vecs, key_vecs)  # (1, 768) (1022, 768) => (1, 1022)
            similarities = similarities[0]  # (1022)
        :return: (N * M) similarity array
        """
        query_vecs = self.encode(queries, return_numpy=True)  # N

        if not isinstance(keys, ndarray):
            key_vecs = self.encode(keys, return_numpy=True)  # M
        else:
            key_vecs = keys  # M

        single_query, single_key = len(query_vecs.shape) == 1, len(key_vecs.shape) ==1
        if single_query:
            query_vecs = query_vecs.reshape(1, -1)
        if single_key:
            key_vecs = key_vecs.reshape(1, -1)

        similarities = cosine_similarity(query_vecs, key_vecs)

        if single_query:
            similarities = similarities[0]
            if single_key:
                similarities = float(similarities[0])

        return similarities

    def search(self,
               queries: Union[str, List[str]],
               threshold: float,
               rank: int
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

        similarities = self.similarity(queries, self.index['embeddings']).tolist()
        id_and_score = []
        for i, s in enumerate(similarities):
            if s >= threshold:
                id_and_score.append((i, s))
        id_and_score = sorted(id_and_score, key=lambda x: x[1], reverse=True)[:rank]
        results = [[
            self.index['sentences'][idx],  # sentences | feature #1
            self.index['id'][idx],         # id        | feature #2
            self.index['category'][idx],   # category  | feature #3
            self.index['timecost'][idx],   # timecost  | feature #4
            self.index['level'][idx],      # level     | feature #5
            '{:.2f}'.format(score)
        ] for idx, score in id_and_score]

        end = datetime.now()
        time = end - start

        logger.info('Searching is completed')
        logger.info(results)
        logger.info('Time Spent : ' + str(time) + ' seconds')

        return results