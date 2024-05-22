from model.sbert import RunSBERT
from dataset.example import SimilarityDatasetExample
import argparse

class PreserveWhitespace(argparse.Action):
    """
    Preserve whitespace from argparser inputs. Ex) '애호박 볶음'
    """
    def __call__(self, parser, namespace, values, option_strings = None):
        setattr(namespace, self.dest, ' '.join(values))

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries",
                        nargs='+',
                        action=PreserveWhitespace,
                        help='List of queries to extract similarity from corpus'
                        )
    parser.add_argument("--batch_size", type=int, default=64, help='Default: 64')
    parser.add_argument("--rank", type=int, default=5, help='Top n ranks in similarity score')
    parser.add_argument("--threshold", type=float, default=0.6, help='Cosine similarity score threshold')
    args = parser.parse_args()

    # recipes = SimilarityDatasetExample(query=[TITLE, KEY, CATEGORY, TIMECOST, LEVEL])
    model = RunSBERT('./plm/KR-SBERT-V40K-klueNLI-augSTS', batch_size=args.batch_size)
    model.corpus_embeddings(sentence=recipes())
    return model.search(queries=args.queries, threshold=args.threshold, rank=args.rank)

if __name__ == '__main__':
    run()