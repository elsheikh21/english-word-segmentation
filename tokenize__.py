import logging
import os
import pickle
import random
import sys
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support,
                             precision_score)
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    parser = ArgumentParser(description="English Words Segmentation")
    parser.add_argument('path')
    return vars(parser.parse_args())


def configure_workspace(SEED=1873337):
    """
    Seed everything to maintain reproducibility,
    configure logging
    """
    random.seed(SEED)
    np.random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s",
                        datefmt='%H:%M:%S', level=logging.INFO)


def save_pickle(save_to, save_what):
    with open(save_to, mode='wb') as f:
        pickle.dump(save_what, f)


def load_pickle(load_from):
    with open(load_from, 'rb') as f:
        return pickle.load(f)


class WikiDataset(Dataset):
    """
    It reads files given as input_file and gold_file, it parses file up till a sequence specified by the arg max_char_len
    trims to max_char_len or pads the input till it reach the length specified
    gets unigrams to create the vocabulary, and labels dict, and vectorize the data (sequence of ints) then converts to tensors
    """
    def __init__(self, input_file_path: str, gold_file_path: str, max_char_len: int = 256, TASK: str = 'BIS'):
        configure_workspace()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = max_char_len
        self.parse_dataset(input_file_path, gold_file_path)
        self.get_unigrams()
        self.create_vocabulary()
        self.encode_labels(TASK)
        self.encoded_data = None
        self.vocab_size = len(self.char2idx.keys())
        self.out_vocab_size = len(self.label2idx.keys())

    def parse_dataset(self, input_file_path, gold_file_path):
        with open(input_file_path, encoding='utf-8', mode='r') as file_:
            lines = file_.readlines()
        self.data_x = [line.strip() for line in lines]

        with open(gold_file_path, encoding='utf-8', mode='r') as gold_file_:
            lines_ = gold_file_.readlines()

        self.data_y = [line.strip() for line in lines_]

    def encode_labels(self, TASK='BIS'):
        if TASK == 'BIS':
            self.label2idx = {'<PAD>': 0, 'B': 1, 'I': 2, 'S': 3}
            self.idx2label = {0: '<PAD>', 1: 'B', 2: 'I', 3: 'S'}
        elif TASK == 'BI':
            self.label2idx = {'<PAD>': 0, 'B': 1, 'I': 2}
            self.idx2label = {0: '<PAD>', 1: 'B', 2: 'I'}
        else:
            raise NotImplementedError

    def get_unigrams(self):
        chars = []
        for sentence in self.data_x:
            chars.extend([word for word in sentence])
        self.unigrams = sorted(list(set(chars)))

    def create_vocabulary(self):
        self.char2idx = dict()
        self.char2idx['<PAD>'] = 0
        self.char2idx['<UNK>'] = 1
        self.char2idx.update(
            {val: key for (key, val) in enumerate(self.unigrams, start=2)})
        self.idx2char = {val: key for (key, val) in self.char2idx.items()}

    def char_padding(self, sentence_):
        if len(sentence_) > self.max_len:
            sentence_ = sentence_[:self.max_len]
        else:
            for _ in range(self.max_len - len(sentence_)):
                sentence_.append(self.label2idx.get('<PAD>'))
        return sentence_

    def vectorize_data(self):
        """
        Converts data_x from a seq of tokens (str) to a seq of indices (int)
        """
        train_x = []
        for sentence in self.data_x:
            sentence_ = [self.char2idx.get(char, 1) for char in sentence]
            train_x.append(self.char_padding(sentence_))
        self.train_x = train_x

        train_y = []
        for sentence in self.data_y:
            sentence_ = [self.label2idx.get(label) for label in sentence]
            train_y.append(self.char_padding(sentence_))
        self.train_y = train_y
        self.encode_data()

    def encode_data(self):
        self.encoded_data = list()
        # data_x_shape >> [samples_num, max_chars_sentence]
        assert len(self.train_x) == len(self.train_y)
        for i in range(len(self.train_x)):
            train_x = torch.LongTensor(self.train_x[i]).to(self.device)
            try:
                train_y = torch.LongTensor(self.train_y[i]).to(self.device)
            except:
                print(i)
                print(self.train_y[i])
            self.encoded_data.append({"inputs": train_x, "outputs": train_y})

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        if self.encoded_data is None:
            raise RuntimeError("Trying to retrieve elements, but dataset is not vectorized yet")
        return self.encoded_data[idx]

    @staticmethod
    def decode_output(logits: torch.Tensor, idx2label):
        """given logits output labels for them by taking argument_max of the logit"""
        max_indices = torch.argmax(logits, -1).tolist()  # shape >> (batch_size, max_len)
        predictions = list()
        for indices in max_indices:
            predictions.append([idx2label[i] for i in indices])
        return predictions

    @staticmethod
    def decode_data(data: torch.Tensor, idx2label):
        data_ = data.tolist()
        return [idx2label.get(idx, None) for idx in data_]


class HyperParameters():
    """Specify the model hyperparameters"""
    hidden_dim = 256
    embedding_dim = 300
    bidirectional = True
    num_layers = 2
    dropout = 0.2
    embeddings = None
    batch_size = 128


class BaselineModel(nn.Module):
    """ Builds the model"""
    def __init__(self, hparams):
        super(BaselineModel, self).__init__()
        self.word_embedding = nn.Embedding(hparams.vocab_size, hparams.embedding_dim)
        if hparams.embeddings is not None:
            print("initializing embeddings from pretrained")
            self.word_embedding.weight.data.copy_(hparams.embeddings)

        self.lstm = nn.LSTM(hparams.embedding_dim, hparams.hidden_dim,
                            bidirectional=hparams.bidirectional,
                            num_layers=hparams.num_layers,
                            dropout=hparams.dropout if hparams.num_layers > 1 else 0)
        lstm_output_dim = hparams.hidden_dim if hparams.bidirectional is False else hparams.hidden_dim * 2

        self.dropout = nn.Dropout(hparams.dropout)
        self.classifier = nn.Linear(lstm_output_dim, hparams.num_classes)

    def forward(self, x):
        embeddings = self.word_embedding(x)
        embeddings = self.dropout(embeddings)
        o, _ = self.lstm(embeddings)
        o = self.dropout(o)
        output = self.classifier(o)
        return output


class EarlyStopping(object):
    """Training Callback function to prevent overfitting based on validation loss"""
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                        best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                        best * min_delta / 100)


def save_checkpoint(state, is_best, filename='/output/checkpoint.pth.tar'):
    if is_best:
        print("Saving a new best model")
        torch.save(state, filename)  # save checkpoint


def load_checkpoint(resume_weights_path, hyperparams):
    cuda = torch.cuda.is_available()
    if cuda:
        checkpoint = torch.load(resume_weights_path)
    else:
        raise RuntimeError('Cuda is not available')

    start_epoch = checkpoint['epoch']
    best_val_loss = checkpoint['best_val_loss']
    model = BaselineModel(hyperparams)
    model.load_state_dict(checkpoint['state_dict'])
    print(f"Checkpoint loaded, trained for {start_epoch} epochs, val_loss: {best_val_loss}")
    return model


class Trainer:
    """ Trainer object to train, predict, evaluate the model"""
    def __init__(self, model: nn.Module, loss_function, optimizer, label_vocab, log_steps: int = 10_000,
                 log_level: int = 2):
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer

        self.label_vocab = label_vocab
        self.log_steps = log_steps
        self.log_level = log_level
        self.label_vocab = label_vocab
        self.writer = SummaryWriter()

    def train(self, train_dataset: Dataset, valid_dataset: Dataset, epochs: int = 1):
        best_val_loss = torch.FloatTensor([1e4])
        assert epochs > 1 and isinstance(epochs, int)
        if self.log_level > 0:
            print('Training ...')
        train_loss = 0.0

        scheduler = ReduceLROnPlateau(self.optimizer, verbose=False, mode='min', patience=2, min_lr=1e-12)

        epoch, epoch_, step = 0, 0, 0
        for epoch in range(epochs):
            logs = {}
            if self.log_level > 0:
                print(f' Epoch {epoch + 1:03d}')

            epoch_loss = 0.0
            self.model.train()

            for step, sample in enumerate(train_dataset):
                inputs = sample['inputs']
                labels = sample['outputs']
                self.optimizer.zero_grad()

                predictions = self.model(inputs)
                predictions = predictions.view(-1, predictions.shape[-1])
                labels = labels.view(-1)

                # Compute accuracy
                _, argmax = torch.max(predictions, 1)
                acc = (labels == argmax.squeeze()).float().mean()

                sample_loss = self.loss_function(predictions, labels)
                sample_loss.backward()
                self.optimizer.step()

                epoch_loss += sample_loss.tolist()

                if self.log_level > 1 and step % self.log_steps == self.log_steps - 1:
                    print(f'\t[Epoch # {epoch:2d} @ step #{step}] Curr avg loss = {epoch_loss / (step + 1):0.4f}')

            avg_epoch_loss = epoch_loss / len(train_dataset)
            train_loss += avg_epoch_loss

            epoch_ += 1

            valid_loss, valid_acc = self.evaluate(valid_dataset)
            logs['loss'] = avg_epoch_loss
            logs['valid_loss'] = valid_loss
            # Get bool not ByteTensor
            is_best = bool(torch.FloatTensor([valid_loss]) > best_val_loss)
            # Get greater Tensor to keep track best validation loss
            best_valid_loss = torch.FloatTensor([min(valid_loss, best_val_loss)])
            # Save checkpoint if is a new best
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'best_val_loss': best_valid_loss
            }, is_best)

            self.writer.add_scalar('Loss', epoch_loss, epoch_)
            self.writer.add_scalar('Loss', valid_loss, epoch_)

            if self.log_level > 0:
                print(f'\tEpoch #: {epoch:2d} [loss: {avg_epoch_loss:0.4f}, val_loss: {valid_loss:0.4f}]')

            scheduler.step(valid_loss)

        if self.log_level > 0:
            print('... Done!')

        avg_epoch_loss = train_loss / epochs
        self.writer.close()
        return avg_epoch_loss

    def evaluate(self, valid_dataset):
        valid_loss = 0.0
        # set dropout to 0!! Needed when we are in inference mode.
        self.model.eval()
        with torch.no_grad():
            for sample in valid_dataset:
                inputs = sample['inputs']
                labels = sample['outputs']

                predictions = self.model(inputs)
                predictions = predictions.view(-1, predictions.shape[-1])
                labels = labels.view(-1)
                sample_loss = self.loss_function(predictions, labels)
                valid_loss += sample_loss.tolist()

                # Compute accuracy
                _, argmax = torch.max(predictions, 1)
                acc = (labels == argmax.squeeze()).float().mean()

        return valid_loss / len(valid_dataset), acc / len(valid_dataset)

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)
            predictions = torch.argmax(logits, -1)
            return logits, predictions


def compute_scores(model: nn.Module, l_dataset: DataLoader):
    all_predictions = list()
    all_labels = list()
    for indexed_elem in l_dataset:
        indexed_in = indexed_elem["inputs"]
        indexed_labels = indexed_elem["outputs"]
        predictions = model(indexed_in)
        predictions = torch.argmax(predictions, -1).view(-1)
        labels = indexed_labels.view(-1)
        valid_indices = labels != 0

        valid_predictions = predictions[valid_indices]
        valid_labels = labels[valid_indices]

        all_predictions.extend(valid_predictions.tolist())
        all_labels.extend(valid_labels.tolist())
    micro_precision_recall_fscore = precision_recall_fscore_support(all_labels, all_predictions,
                                                                    average="micro",
                                                                    zero_division=0)

    macro_precision_recall_fscore = precision_recall_fscore_support(all_labels, all_predictions,
                                                                    average="macro",
                                                                    zero_division=0)

    per_class_precision = precision_score(all_labels, all_predictions, average=None, zero_division=0)

    return {"macro_precision_recall_fscore": macro_precision_recall_fscore,
            "micro_precision_recall_fscore": micro_precision_recall_fscore,
            "per_class_precision": per_class_precision,
            "confusion_matrix": confusion_matrix(all_labels, all_predictions, normalize='true')}


def pprint_confusion_matrix(conf_matrix, num_classes):
    """ Plot confusion matrix as an image. """
    df_cm = pd.DataFrame(conf_matrix, range(num_classes), range(num_classes))
    plt.figure(figsize=(10, 7))
    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})  # font size
    plt.show()


def predict(model: nn.Module, data_x):
    """ Created only for predicting test_x using model -given as an arg- """
    # Model predictions
    model.eval()
    with torch.no_grad():
        logits = model(data_x)
        predictions = torch.argmax(logits, -1)
    return logits, predictions


def parse_file(input_file_path):
    """ Created only for parsing test file """
    with open(input_file_path, encoding='utf-8', mode='r') as file_:
        lines = file_.readlines()
    return [line.strip() for line in lines]


def tokenize_outputs(model, char2idx, idx2label, test_x):
    """ 
    Takes test_x to vectorize it using char2idx and compute model predictions then use
    the idx2label to de-vectorize the test_y output from model and then print them
    """
    y_pred = []
    for data_x in test_x:
        data_x_ = [char2idx.get(char, 1) for char in data_x]
        data_x = torch.LongTensor(data_x_).to('cuda')
        logits, _ = predict(model, data_x.unsqueeze(0))
        pred_y = WikiDataset.decode_output(logits, idx2label)[0]
        y_pred.append(pred_y)

    for prediction in y_pred:
        print(f"{''.join(prediction)}")


if __name__ == '__main__':
    # Get parser arguments (args)
    args = parse_args()
    path = args['path']

    # prepare to print to file
    sys.stdout.flush()
    # define task based on the path_arg
    task = 'BI' if 'merged' in path else 'BIS'
    # define file names
    file_path = path + '.sentences.train'
    gold_file_path = path + '.gold.train'
    dev_file_path = path + '.sentences.dev'
    dev_gold_file_path = path + '.gold.dev'
    test_file_path = path + '.sentences.test'

    # Build train and dev dataset then vectorize both
    train_dataset = WikiDataset(file_path, gold_file_path, TASK=task)
    train_dataset.vectorize_data()

    dev_dataset = WikiDataset(dev_file_path, dev_gold_file_path, TASK=task)
    dev_dataset.char2idx = train_dataset.char2idx
    dev_dataset.idx2char = train_dataset.idx2char
    dev_dataset.label2idx = train_dataset.label2idx
    dev_dataset.idx2label = train_dataset.idx2label
    dev_dataset.vectorize_data()

    # Define model hyperparams
    hyperparams = HyperParameters()
    # Add missing hyperparams
    hyperparams.vocab_size = train_dataset.vocab_size
    hyperparams.num_classes = train_dataset.out_vocab_size

    # Build the model using hyperparams
    baseline_model = BaselineModel(hyperparams).cuda()

    # Create trainer object
    trainer = Trainer(
        model=baseline_model,
        loss_function=nn.CrossEntropyLoss(ignore_index=train_dataset.label2idx['<PAD>']),
        optimizer=optim.Adam(baseline_model.parameters(), lr=1e-6, weight_decay=1e-5),
        label_vocab=train_dataset.label2idx,
        log_level=0
    )

    # Create dataloader instances with defined batch size shuffle the train_Dataset
    train_dataset_ = DataLoader(train_dataset, batch_size=hyperparams.batch_size, shuffle=True)
    dev_dataset_ = DataLoader(dev_dataset, batch_size=hyperparams.batch_size, shuffle=False)

    # Train the model
    trainer.train(train_dataset_, dev_dataset_, epochs=100)

    # Parse test_x from test_file_path specified earlier
    test_x = parse_file(test_file_path)

    # Use the model (passed as input) to predict test_y given test_x and necessary dictionaries
    tokenize_outputs(baseline_model, train_dataset.char2idx, train_dataset.idx2label, test_x)
