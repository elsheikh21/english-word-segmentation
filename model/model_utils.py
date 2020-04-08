import io
import numpy as np
from tqdm.auto import tqdm
import torch
from model import BaselineModel


def save_checkpoint(state, is_best, filename='/output/checkpoint.pth.tar'):
    if is_best:
        print("Saving a new best model")
        torch.save(state, filename)  # save checkpoint


def load_checkpoint(resume_weights_path, hyperparams):
    cuda = torch.cuda.is_available()
    if cuda:
        checkpoint = torch.load(resume_weights_path)

    start_epoch = checkpoint['epoch']
    best_validation_loss = checkpoint['best_val_loss']
    model = BaselineModel(hyperparams)
    model.load_state_dict(checkpoint['state_dict'])
    print(
        f"loaded checkpoint '{checkpoint}' (trained for {start_epoch} epochs, val loss: {best_validation_loss})")
    return model


def load_pretrained_embeddings(fname, char2idx, embeddings_size):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in tqdm(fin, desc=f'Reading data from {fname}'):
        tokens = line.rstrip().split(' ')
        # Get only chars in word2idx, char embeddings_vector
        if len(tokens[0]) == 1:
            data[tokens[0]] = np.array(tokens[1:], dtype=np.float)
        else:
            continue

    pretrained_embeddings = torch.randn(len(char2idx), embeddings_size)
    initialised = 0
    for idx, char in enumerate(data):
        if char in char2idx:
            initialised += 1
            vector_ = torch.from_numpy(data[char])
            pretrained_embeddings[char2idx.get(char)] = vector_

    pretrained_embeddings[char2idx["<PAD>"]] = torch.zeros(embeddings_size)
    pretrained_embeddings[char2idx["<UNK>"]] = torch.zeros(embeddings_size)
    print(f'Loaded {initialised} vectors and instantiated random embeddings for {len(char2idx) - initialised}')
    return pretrained_embeddings