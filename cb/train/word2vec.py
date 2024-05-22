from utils.w2v_tokenization import tokenization
from gensim.models import Word2Vec
from dataset.example import content_search
from utils.constant import strings as s

recipes = content_search(query=[s.TITLE, s.KEY, s.CATEGORY, s.TIMECOST, s.LEVEL])
titles = tokenization(recipes[0], token='Okt')
model = Word2Vec(
    sentences=titles,
    vector_size=100,
    window=1,
    min_count=0,
    workers=2,
    sg=0
)
# model.wv.vectors.shape
# model.wv.most_similar('쌀', topn=5)