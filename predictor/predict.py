import os
import torch
import torch.nn as nn

from tqdm.auto import tqdm
from model import BaselineModel
from data_loader import WikiDataset
from model import HyperParameters
from utilities import load_pickle


def predict(model: nn.Module, data_x):
    # Model predictions
    model.eval()
    with torch.no_grad():
        logits = model(data_x)
        predictions = torch.argmax(logits, -1)
    return logits, predictions


def tokenize_outputs(model_path, test_x, output_path):
    RESOURCES_PATH = os.path.join(os.getcwd(), 'resources')
    char2idx_path = os.path.join(RESOURCES_PATH, 'char2idx.pkl')
    idx2label_path = os.path.join(RESOURCES_PATH, 'idx2label.pkl')

    char2idx = load_pickle(char2idx_path)
    idx2label = load_pickle(idx2label_path)

    vocab_size = len(char2idx.items())
    out_vocab_size = len(idx2label.items())

    # init hyperparameters
    hyperparams = HyperParameters()
    hyperparams.vocab_size = vocab_size
    hyperparams.num_classes = out_vocab_size

    # Load model
    model = BaselineModel(hyperparams)
    try:
        model.load_state_dict(torch.load(model_path))
    except RuntimeError:
        model.load_state_dict(torch.load(
            model_path, map_location=torch.device('cpu')))

    # compute predictions
    y_pred = []
    for data_x in tqdm(test_x, desc='Computing predictions'):
        data_x_ = [char2idx.get(char, 1) for char in data_x]
        data_x = torch.LongTensor(data_x_)
        logits, _ = predict(model, data_x.unsqueeze(0))
        pred_y = WikiDataset.decode_output(logits, idx2label)[0]
        y_pred.append(pred_y)

    # Save to text file
    with open(output_path, encoding='utf-8', mode='w+') as outputs_file:
        for prediction in tqdm(y_pred, desc='Writing predictions'):
            outputs_file.write(f"{''.join(prediction)}\n")
