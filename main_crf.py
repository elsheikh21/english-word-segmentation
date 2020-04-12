import os
from data_loader import WikiDataset
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from model import HyperParameters, load_pretrained_embeddings, BiLSTM_CRF
from utilities import configure_workspace


if __name__ == "__main__":
    configure_workspace()
    DATA_PATH = os.path.join(os.getcwd(), 'data')
    RESOURCES_PATH = os.path.join(os.getcwd(), 'resources')

    file_path = os.path.join(DATA_PATH, 'en.wiki.sentences.train')
    dev_file_path = os.path.join(DATA_PATH, 'en.wiki.sentences.dev')
    gold_file_path = os.path.join(DATA_PATH, 'en.wiki.gold.train')
    dev_gold_file_path = os.path.join(DATA_PATH, 'en.wiki.gold.dev')

    print('========== Training Dataset ==========')
    train_dataset = WikiDataset(file_path, gold_file_path, is_crf=True)
    train_dataset.vectorize_data()

    train_x = torch.LongTensor(train_dataset.train_x)
    print(f'train_x shape is: {train_x.shape}')
    # x.shape = [number of samples, max characters/sentence] = [31_553, 256]
    train_y = torch.LongTensor(train_dataset.train_y)
    print(f'train_y shape is: {train_y.shape}')
    # y.shape = [number of samples, max characters/sentence] = [31_553, 256]

    dev_dataset = WikiDataset(dev_file_path, dev_gold_file_path, is_crf=True)
    dev_dataset.char2idx = train_dataset.char2idx
    dev_dataset.idx2char = train_dataset.idx2char
    dev_dataset.label2idx = train_dataset.label2idx
    dev_dataset.idx2label = train_dataset.idx2label
    dev_dataset.vectorize_data()

    hyperparams = HyperParameters()
    hyperparams.vocab_size = train_dataset.vocab_size
    hyperparams.num_classes = train_dataset.out_vocab_size

    # mask tensor with shape (batch_size, max_sent_size)
    # mask = (x_tags != Const.PAD_TAG_ID).float()

    # see bilstm_crf.py
    model = BiLSTM_CRF(hyperparams)
    opt = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
    train_dataset_ = DataLoader(train_dataset, batch_size=hyperparams.batch_size, shuffle=True)
    model.train_(optimizer=opt, epochs=1, train_dataset=train_dataset_)

    save_model_path = os.path.join(RESOURCES_PATH, 'bilstm_crf_model.pt')
    torch.save(model.state_dict(), save_model_path)
    # model.load_state_dict(torch.load(save_model_path))

    model.predict(dev_dataset.train_x, train_dataset.idx2label)
