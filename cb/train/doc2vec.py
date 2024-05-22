from dataset.example import content_search
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from numpy import ndarray
from typing import List, Union
import logging

def TrainDoc2Vec(
        key: Union[str, List[str], ndarray],
        doc: Union[str, List[str], ndarray],
        vector_size: int = 100,
        min_count: int = 2,
        workers: int = 2,
        window: int = 4,
        epoch: int = 10
):
    document = []
    for i, k in zip(key, doc):
        document.append(TaggedDocument(tags=[str(i)], words=k))

    model = Doc2Vec(
        vector_size=vector_size,
        min_count=min_count,
        workers=workers,
        window=window
    )
    model.build_vocab(document)
    model.train(document, total_examples=model.corpus_count, epochs=epoch)
    return model

def RunDoc2Vec(model,
               key: str,
               top: int = 5,
               output = True
               ):

    logging.basicConfig(format='(%(asctime)s) - [%(levelname)s] : %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Model Output Format
    if not output:
        result = model.dv.most_similar(key, topn=top)
        logger.info(result)
        return result

    # Extract output from data
    ids = []
    for i in model.dv.most_similar(key, topn=top):
        ids.append(i[0])
    result = content_search(search_key=ids)
    logger.info(result.to_string(index=False))