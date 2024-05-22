from sklearn.preprocessing import LabelEncoder
from utils.utils import dir_path
import numpy as np
import pandas as pd
import torch.utils.data

class MovieLens1MDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path: str):
        """
        :param dataset_path: insert folder directory only. ex) 'src/data/ml-1m'

        @Description
            * data: 'user_id', 'movie_id', 'gender', 'occupation', 'age', 'rating'
            * items: 'user_id', 'movie_id', 'gender', 'occupation', 'age'
            * targets: 'rating'
            * field_dims: max('user_id', 'movie_id', 'gender', 'occupation', 'age')
        """
        data = self.__preprocess_data(dir_path(dataset_path))

        self.items = np.concatenate((data[:, :2].astype(int) - 1, data[:, 2:self.col-1].astype(int)), axis=1)  # -1 because ID begins from 1
        self.targets = self.__preprocess_binary(data[:, self.col-1]).astype(np.float32)
        self.field_dims = np.max(self.items, axis=0) + 1

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        return self.items[index], self.targets[index]
    def __preprocess_data(self, dataset_path, sep='::', engine='python', header=None):
        ratings = pd.read_csv(dataset_path + '/ratings.dat', sep=sep, engine=engine, header=header)
        ratings.columns = ['user_id', 'movie_id', 'rating', 'timestamp']
        users = pd.read_csv(dataset_path + '/users.dat', sep=sep, engine=engine, header=header)
        users.columns = ['user_id', 'gender', 'age', 'occupation', 'zip code']

        dataset = ratings.merge(users, on='user_id', how='right')
        dataset = dataset[['user_id', 'movie_id', 'gender', 'occupation', 'age', 'rating']]
        self.col = len(dataset.columns)

        gender_encoder = LabelEncoder()
        occupations_encoder = LabelEncoder()
        age_encode = LabelEncoder()

        dataset['gender'] = gender_encoder.fit_transform(dataset['gender'])
        dataset['occupation'] = occupations_encoder.fit_transform(dataset['occupation'])
        dataset['age'] = age_encode.fit_transform(dataset['age'])

        return dataset.to_numpy()

    def __preprocess_binary(self, target):
        target[target <= 3] = 0
        target[target > 3] = 1
        return target