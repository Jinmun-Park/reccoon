import os
import pickle
from sklearn.preprocessing import LabelEncoder
from utils.constant import strings as s

def dir_path(path: str):
    dir_path = os.path.dirname(os.path.realpath('__file__'))
    return dir_path + path


def pickle_load(name):
    file = open('save/' + name, 'rb')
    encoder = pickle_load(file)
    file.close()
    return encoder


def pickle_save(name, encoder):
    file = open('save/' + name, 'wb')
    pickle.dump(encoder, file)
    file.close()


def Encoder(df, save: bool = True):
    """
    :param df: Example) [parent_key, age, gender, key, count]
    :param save: if 'False', load the file only.
    """
    if save:
        age_encoder = LabelEncoder()
        category_encoder = LabelEncoder()
        timecost_encoder = LabelEncoder()
        level_encoder = LabelEncoder()

        df[s.AGE] = age_encoder.fit_transform(df[s.AGE])
        df[s.CATEGORY] = category_encoder.fit_transform(df[s.CATEGORY])
        df[s.TIMECOST] = timecost_encoder.fit_transform(df[s.TIMECOST])
        df[s.LEVEL] = level_encoder.fit_transform(df[s.LEVEL])

        pickle_save(s.AGE_ENCODER, age_encoder)
        pickle_save(s.CATEGORY_ENCODER, category_encoder)
        pickle_save(s.TIMECOST_ENCODER, timecost_encoder)
        pickle_save(s.LEVEL_ENCODER, level_encoder)

    else:
        age_encoder = pickle_load(s.AGE_ENCODER)
        category_encoder = pickle_load(s.CATEGORY_ENCODER)
        timecost_encoder = pickle_load(s.TIMECOST_ENCODER)
        level_encoder = pickle_load(s.LEVEL_ENCODER)

        df[s.AGE] = age_encoder.transform(df[s.AGE])
        df[s.CATEGORY] = category_encoder.transform(df[s.CATEGORY])
        df[s.TIMECOST] = timecost_encoder.transform(df[s.TIMECOST])
        df[s.LEVEL] = level_encoder.transform(df[s.LEVEL])

    return df


def mean_pooling(model_output, attention_mask):
    """
    Calculates the average of token embeddings across all tokens in the sentence.

    :param model_output: [*last_hs, hs, attentions]
    :param attention_mask: shape(n_sentences, length)

    @ Input:
        mean_pooling(model_outputs, encoded_input['attention_mask']
    @ Example:
        outputs = model(**encoded_input)  # [*last_hs, hs, attentions]
        attention_mask = encoded_input['attention_mask'].shape  # (sentences, length)
        sentence_embeddings = mean_pooling(outputs, encoded_input['attention_mask'])

        # model_output[0]  # shape(2,9,768)
        # attention_mask  # shap(2,9)
    """
    token_embeddings = model_output[0]
    input_mask_expand = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expand, 1)
    sum_mask = torch.clamp(input_mask_expand.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def calculate_cosine_similarity_score(x, y, ratio=1):
    """
    Calculate similarity score based on 'BERT' output/

    :param x: sentence_embeddings[i] for all i and j.
    :param y: sentence_embeddings[j] for all i and j.
    :param ratio: ratio can set to 100. Default is 1.

    @ Input:
        (sentence_embeddings, sentence_embeddings)
    @ Example:
        x, y = (sentence_embeddings, sentence_embeddings)
            * sentence_embeddings = mean_pooling(model_outputs, encoded_input['attention_mask'])
        or
        x, y = (embeddings[0][0], embeddings[1][0]) or ....(embeddings[0][768], embeddings[1][768])
            * embeddings, _ = model(**encoded_input, return_dict=False)  # last_hs
    """
    if len(x.shape) == 1: x = x.unsqueeze(0)  # shape(1,768)
    if len(y.shape) == 1: y = y.unsqueeze(0)  # shape(1,768)
    xnorm = x.norm(dim=1)  # shape(1), L2 NORM
    ynorm = y.norm(dim=1)  # shape(1), L2 NORM
    x_norm = x / xnorm.unsqueeze(1)
    y_norm = y / ynorm.unsqueeze(1)
    return torch.mm(x_norm, y_norm.transpose(0, 1)) * ratio