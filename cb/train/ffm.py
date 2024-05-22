import logging
import torch
import tqdm
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from dataset.movielens import MovieLens1MDataset
from dataset.example import FamiliarityDatasetExample
from train.earlystop import HardEarlyStopper
from model.ffm import FieldAwareFactorizationMachineModel

logging.basicConfig(format='(%(asctime)s) - [%(levelname)s] : %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def get_dataset(name, path):
    if name == 'movielens1M':
        return MovieLens1MDataset(path)
    elif name == 'example':
        return FamiliarityDatasetExample(path)
    else:
        raise ValueError('Invalid dataset name: ' + name)

def get_model(name, dataset, embed_dim):
    field_dims = dataset.field_dims
    if name == 'ffm':
        return FieldAwareFactorizationMachineModel(field_dims, embed_dim=embed_dim)
    else:
        raise ValueError('Invalid model name: ' + name)


def train(model, optimizer, data_loader, criterion, device, log_interval=100):
    model.train()
    total_loss = 0
    tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    for i, (fields, target) in enumerate(tk0):
        fields, target = fields.to(device), target.to(device)
        y = model(fields)  # model(X_train_continuous_tensor, X_train_categorical_tensor.long().to(device))
        loss = criterion(y, target.float())
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (i + 1) % log_interval == 0:
            tk0.set_postfix(loss=total_loss / log_interval)
            total_loss = 0


def test(model, data_loader, device):
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        for fields, target in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            fields, target = fields.to(device), target.to(device)
            y = model(fields)
            targets.extend(target.tolist())
            predicts.extend(y.tolist())
    return roc_auc_score(targets, predicts)


def main(dataset_name,
         dataset_path,
         model_name,
         epoch,
         learning_rate,
         batch_size,
         weight_decay,
         embed_dim,
         save_path,
         path_exist):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = get_dataset(dataset_name, dataset_path)
    train_length = int(len(dataset) * 0.8)
    valid_length = int(len(dataset) * 0.1)
    test_length = len(dataset) - train_length - valid_length
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset, (train_length, valid_length, test_length))
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=8)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8)
    model = get_model(model_name, dataset, embed_dim).to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    early_stopper = HardEarlyStopper(num_trials=2, save_path=f'{save_path}/{model_name}.pt', path_exist=path_exist)
    for epoch_i in range(epoch):
        train(model, optimizer, train_data_loader, criterion, device)
        auc = test(model, valid_data_loader, device)
        logger.info(f'Epoch:, {epoch_i}, Validation: AUC:, {auc}')
        if not early_stopper.is_continuable(model, auc):
            logger.info(f'Validation: Best AUC: {early_stopper.best_accuracy}')
            break
    auc = test(model, test_data_loader, device)
    logger.info(f'Test AUC: {auc}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='movielens1M')
    parser.add_argument('--dataset_path', default='src/example', help='src/**filename')
    parser.add_argument('--model_name', default='ffm')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--embed_dim', type=int, default=16, help='fm=16, ffm=4, xdfm=16')
    parser.add_argument('--save_path', default='save')
    parser.add_argument('--path_exist', action='store_true')
    args = parser.parse_args()
    main(args.dataset_name,
         args.dataset_path,
         args.model_name,
         args.epoch,
         args.learning_rate,
         args.batch_size,
         args.weight_decay,
         args.embed_dim,
         args.save_path,
         args.path_exist
         )


# --dataset_name "movielens1M" --dataset_path "data/ml-1m" --embed_dim 4 --model_name "ffm" --epoch 20
# --dataset_name "movielens1Mcustom" --dataset_path "data/ml-1m" --embed_dim 4 --model_name "ffm" --epoch 20