import torch
import torch.nn as nn
from utils import load_pickle
from tqdm.auto import tqdm
from models import BaselineModel
from parse_dataset import WikiDataset
from hyperparameters import HyperParameters


def decode_output(logits: torch.Tensor, idx2label):
    # print(logits.shape) # shape = (batch_size, max_len)
    max_indices = torch.argmax(logits, -1).tolist()
    predictions = list()
    for indices in max_indices:
        predictions.append([idx2label[i] for i in indices])
    return predictions


def predict(model: nn.Module, data_x):
    # Model predictions
    model.eval()
    with torch.no_grad():
        logits = model(data_x)
        predictions = torch.argmax(logits, -1)
    return logits, predictions


def tokenize_outputs(model_path, test_x, train_dataset, output_path):
    # init hyperparameters
    hyperparams = HyperParameters()
    hyperparams.vocab_size = train_dataset.vocab_size
    hyperparams.num_classes = train_dataset.out_vocab_size

    # Load model
    model = BaselineModel(hyperparams)
    model.load_state_dict(torch.load(model_path))

    # compute predictions
    y_pred = []
    for data_x in test_x:
        logits, _ = predict(model, data_x.unsqueeze(0))
        pred_y = WikiDataset.decode_output(logits, train_dataset.idx2label)[0]
        y_pred.append(pred_y)

    # Save to text file
    with open('predictions.txt', encoding='utf-8', mode='w+') as outputs_file:
        for prediciton in tqdm(y_pred, desc='Writing predictions'):
            outputs_file.write(f"{''.join(prediciton)}\n")
